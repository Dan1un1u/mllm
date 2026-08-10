// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

namespace {

constexpr int kContextLength = 1024;
constexpr int kHiddenSize = 2048;
constexpr int kHeadDim = 128;
constexpr int kKVHeads = 8;
constexpr int kVocabSize = 151936;

struct TimingResult {
  int part = 0;
  int seq_len = 0;
  int block_count = 0;
  std::string graph;
  double median_us = 0.0;
  double p95_us = 0.0;
  double mean_us = 0.0;
  int64_t min_us = 0;
  int64_t max_us = 0;
  std::vector<int64_t> samples_us;
  std::vector<uint64_t> output_fnv1a;
};

template <typename T>
void fillDeterministic(Tensor& tensor, uint32_t seed) {
  auto* data = tensor.ptr<T>();
  uint32_t state = seed;
  for (size_t i = 0; i < tensor.numel(); ++i) {
    state = state * 1664525U + 1013904223U;
    data[i] = static_cast<T>((state >> 16U) & 0xFFU);
  }
}

uint64_t fnv1a(const void* data, size_t bytes) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= ptr[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t checksumOutput(size_t output_index, Tensor& tensor) {
  // Every split graph emits UInt16 hidden/logits first, followed by UInt8 K/V
  // cache tensors. Part 1 only emits the UInt16 embedding output.
  if (output_index == 0) {
    return fnv1a(tensor.ptr<uint16_t>(), tensor.numel() * sizeof(uint16_t));
  }
  return fnv1a(tensor.ptr<uint8_t>(), tensor.numel());
}

int blockCountForPart(int part) {
  if (part == 2 || part == 3) { return 10; }
  if (part == 4) { return 8; }
  return 0;
}

std::vector<Tensor> makeInputs(int part, int seq_len) {
  std::vector<Tensor> inputs;
  if (part == 1) {
    auto sequence = Tensor::empty({1, seq_len}, mllm::kInt32, mllm::kQNN).alloc();
    auto* data = sequence.ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i) { data[i] = 1 + i; }
    inputs.push_back(sequence);
    return inputs;
  }

  const int block_count = blockCountForPart(part);
  const int past_len = kContextLength - seq_len;
  auto hidden = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto sin = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto cos = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto mask = Tensor::empty({1, 1, seq_len, kContextLength}, mllm::kUInt16, mllm::kQNN).alloc();
  fillDeterministic<uint16_t>(hidden, 0x6101U + static_cast<uint32_t>(seq_len));
  fillDeterministic<uint16_t>(sin, 0x6102U + static_cast<uint32_t>(seq_len));
  fillDeterministic<uint16_t>(cos, 0x6103U + static_cast<uint32_t>(seq_len));
  std::fill(mask.ptr<uint16_t>(), mask.ptr<uint16_t>() + mask.numel(), UINT16_MAX);
  inputs = {hidden, sin, cos, mask};

  for (int i = 0; i < block_count; ++i) {
    auto past_key = Tensor::empty({1, kKVHeads, kHeadDim, past_len}, mllm::kUInt8, mllm::kQNN).alloc();
    fillDeterministic<uint8_t>(past_key, 0x6200U + static_cast<uint32_t>(seq_len) + i);
    inputs.push_back(past_key);
    auto past_value = Tensor::empty({1, kKVHeads, past_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc();
    fillDeterministic<uint8_t>(past_value, 0x6300U + static_cast<uint32_t>(seq_len) + i);
    inputs.push_back(past_value);
  }
  return inputs;
}

std::vector<Tensor> makeOutputs(int part, int seq_len) {
  std::vector<Tensor> outputs;
  if (part == 1) {
    outputs.push_back(Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc());
    return outputs;
  }

  const int block_count = blockCountForPart(part);
  if (part == 4) {
    outputs.push_back(
        Tensor::empty({1, 1, seq_len, kVocabSize}, mllm::kUInt16, mllm::kQNN).alloc());
  } else {
    outputs.push_back(Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc());
  }
  for (int i = 0; i < block_count; ++i) {
    outputs.push_back(Tensor::empty({1, kKVHeads, kHeadDim, seq_len}, mllm::kUInt8, mllm::kQNN).alloc());
  }
  for (int i = 0; i < block_count; ++i) {
    outputs.push_back(Tensor::empty({1, kKVHeads, seq_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
  }
  return outputs;
}

double percentile(const std::vector<int64_t>& sorted, double quantile) {
  const auto index = static_cast<size_t>(std::ceil(quantile * static_cast<double>(sorted.size()))) - 1;
  return static_cast<double>(sorted[std::min(index, sorted.size() - 1)]);
}

TimingResult runGraph(int part, int seq_len, int warmup, int iterations) {
  const std::string graph = "model.0.s" + std::to_string(seq_len);
  QnnAOTModule module(graph);
  module.to(mllm::kQNN);
  auto inputs = makeInputs(part, seq_len);
  auto outputs = makeOutputs(part, seq_len);
  module.setOutputTensors(outputs);

  for (int i = 0; i < warmup; ++i) { outputs = module(inputs); }
  std::vector<int64_t> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    outputs = module(inputs);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  }

  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                      static_cast<double>(samples.size());
  std::vector<uint64_t> checksums;
  checksums.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    checksums.push_back(checksumOutput(i, outputs[i]));
  }
  return {
      .part = part,
      .seq_len = seq_len,
      .block_count = blockCountForPart(part),
      .graph = graph,
      .median_us = percentile(sorted, 0.5),
      .p95_us = percentile(sorted, 0.95),
      .mean_us = mean,
      .min_us = sorted.front(),
      .max_us = sorted.back(),
      .samples_us = std::move(samples),
      .output_fnv1a = std::move(checksums),
  };
}

void writeJson(const std::string& path, const std::string& context, int part, int warmup, int iterations,
               const std::vector<TimingResult>& results) {
  std::ofstream stream(path);
  if (!stream.is_open()) { throw std::runtime_error("cannot open output JSON: " + path); }
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"contract\": \"qwen3_split_context_device_probe_v1\",\n"
         << "  \"context\": \"" << context << "\",\n"
         << "  \"part\": " << part << ",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"timing_boundary\": \"host wall time around QnnAOTModule/QnnGraph_execute\",\n"
         << "  \"results\": [\n";
  for (size_t ri = 0; ri < results.size(); ++ri) {
    const auto& result = results[ri];
    stream << "    {\n"
           << "      \"graph\": \"" << result.graph << "\",\n"
           << "      \"seq_len\": " << result.seq_len << ",\n"
           << "      \"block_count\": " << result.block_count << ",\n"
           << "      \"median_us\": " << result.median_us << ",\n"
           << "      \"p95_us\": " << result.p95_us << ",\n"
           << "      \"mean_us\": " << result.mean_us << ",\n"
           << "      \"min_us\": " << result.min_us << ",\n"
           << "      \"max_us\": " << result.max_us << ",\n"
           << "      \"output_fnv1a\": [";
    for (size_t i = 0; i < result.output_fnv1a.size(); ++i) {
      if (i != 0) { stream << ", "; }
      stream << "\"" << result.output_fnv1a[i] << "\"";
    }
    stream << "]\n    }";
    if (ri + 1 != results.size()) { stream << ','; }
    stream << '\n';
  }
  stream << "  ]\n}\n";
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context = Argparse::add<std::string>("-m|--model").help("Split QNN context path.");
  auto& part = Argparse::add<int>("--part").def(1).help("Split part: 1, 2, 3, or 4.");
  auto& graph = Argparse::add<std::string>("--graph").def("both").help("s1, s32, or both.");
  auto& warmup = Argparse::add<int>("--warmup").def(2).help("Warmup executions per graph.");
  auto& iterations = Argparse::add<int>("--iterations").def(5).help("Measured executions per graph.");
  auto& output = Argparse::add<std::string>("-o|--output").def("qwen3_split_device_probe.json");
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--model is required"); }
  if (part.get() < 1 || part.get() > 4) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--part must be 1, 2, 3, or 4");
  }
  if (graph.get() != "s1" && graph.get() != "s32" && graph.get() != "both") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--graph must be s1, s32, or both");
  }
  if (warmup.get() < 0 || iterations.get() <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "warmup must be non-negative and iterations must be positive");
  }

  mllm::initQnnBackend(context.get());
  std::vector<TimingResult> results;
  if (graph.get() == "s1" || graph.get() == "both") {
    results.push_back(runGraph(part.get(), 1, warmup.get(), iterations.get()));
  }
  if (graph.get() == "s32" || graph.get() == "both") {
    results.push_back(runGraph(part.get(), 32, warmup.get(), iterations.get()));
  }
  writeJson(output.get(), context.get(), part.get(), warmup.get(), iterations.get(), results);
  for (const auto& result : results) {
    fmt::print("part{} {}: median={:.3f} ms p95={:.3f} ms mean={:.3f} ms outputs={}\n",
               result.part, result.graph, result.median_us / 1000.0, result.p95_us / 1000.0,
               result.mean_us / 1000.0, result.output_fnv1a.size());
    for (size_t i = 0; i < result.output_fnv1a.size(); ++i) {
      fmt::print("  output[{}] fnv1a={:016x}\n", i, result.output_fnv1a[i]);
    }
  }
  fmt::print("wrote {}\n", output.get());
  return 0;
});
