// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
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

struct TimingResult {
  std::string workload;
  std::string graph;
  int seq_len;
  int past_len;
  double median_us;
  double p95_us;
  double mean_us;
  int64_t min_us;
  int64_t max_us;
  std::vector<int64_t> samples_us;
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

template <typename T>
void dumpTensor(const std::string& path, Tensor& tensor) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream.is_open()) { throw std::runtime_error("cannot open tensor dump: " + path); }
  stream.write(reinterpret_cast<const char*>(tensor.ptr<T>()),
               static_cast<std::streamsize>(tensor.numel() * sizeof(T)));
  if (!stream.good()) { throw std::runtime_error("cannot write tensor dump: " + path); }
}

void dumpGraphTensors(const std::string& prefix, const std::string& workload, int seq_len, int past_len,
                      std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (prefix.empty()) { return; }
  const auto root = prefix + "." + workload;
  dumpTensor<uint16_t>(root + ".hidden_u16.bin", inputs[0]);
  dumpTensor<uint16_t>(root + ".sin_u16.bin", inputs[1]);
  dumpTensor<uint16_t>(root + ".cos_u16.bin", inputs[2]);
  dumpTensor<uint16_t>(root + ".mask_u16.bin", inputs[3]);
  dumpTensor<uint8_t>(root + ".past_key_u8.bin", inputs[4]);
  dumpTensor<uint8_t>(root + ".past_value_u8.bin", inputs[5]);
  dumpTensor<uint16_t>(root + ".hidden_out_u16.bin", outputs[0]);
  dumpTensor<uint8_t>(root + ".present_key_u8.bin", outputs[1]);
  dumpTensor<uint8_t>(root + ".present_value_u8.bin", outputs[2]);

  std::ofstream stream(root + ".json");
  if (!stream.is_open()) { throw std::runtime_error("cannot open tensor dump metadata: " + root + ".json"); }
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"contract\": \"qwen3_layer5_block_raw_quantized_v1\",\n"
         << "  \"workload\": \"" << workload << "\",\n"
         << "  \"seq_len\": " << seq_len << ",\n"
         << "  \"past_len\": " << past_len << ",\n"
         << "  \"context_length\": " << kContextLength << ",\n"
         << "  \"hidden_size\": " << kHiddenSize << ",\n"
         << "  \"head_dim\": " << kHeadDim << ",\n"
         << "  \"kv_heads\": " << kKVHeads << ",\n"
         << "  \"seeds\": {\n"
         << "    \"hidden\": " << (0x5101U + seq_len) << ",\n"
         << "    \"sin\": " << (0x5102U + seq_len) << ",\n"
         << "    \"cos\": " << (0x5103U + seq_len) << ",\n"
         << "    \"past_key\": " << (0x5104U + seq_len) << ",\n"
         << "    \"past_value\": " << (0x5105U + seq_len) << "\n"
         << "  },\n"
         << "  \"root\": \"" << root << "\",\n"
         << "  \"tensors\": [\n"
         << "    {\"name\": \"hidden\", \"file_suffix\": \".hidden_u16.bin\", \"dtype\": \"uint16\", \"shape\": [1, " << seq_len << ", " << kHiddenSize << "]},\n"
         << "    {\"name\": \"sin\", \"file_suffix\": \".sin_u16.bin\", \"dtype\": \"uint16\", \"shape\": [1, " << seq_len << ", " << kHeadDim << "]},\n"
         << "    {\"name\": \"cos\", \"file_suffix\": \".cos_u16.bin\", \"dtype\": \"uint16\", \"shape\": [1, " << seq_len << ", " << kHeadDim << "]},\n"
         << "    {\"name\": \"mask\", \"file_suffix\": \".mask_u16.bin\", \"dtype\": \"uint16\", \"shape\": [1, 1, " << seq_len << ", " << kContextLength << "]},\n"
         << "    {\"name\": \"past_key\", \"file_suffix\": \".past_key_u8.bin\", \"dtype\": \"uint8\", \"shape\": [1, " << kKVHeads << ", " << kHeadDim << ", " << past_len << "]},\n"
         << "    {\"name\": \"past_value\", \"file_suffix\": \".past_value_u8.bin\", \"dtype\": \"uint8\", \"shape\": [1, " << kKVHeads << ", " << past_len << ", " << kHeadDim << "]},\n"
         << "    {\"name\": \"hidden_out\", \"file_suffix\": \".hidden_out_u16.bin\", \"dtype\": \"uint16\", \"shape\": [1, " << seq_len << ", " << kHiddenSize << "]},\n"
         << "    {\"name\": \"present_key\", \"file_suffix\": \".present_key_u8.bin\", \"dtype\": \"uint8\", \"shape\": [1, " << kKVHeads << ", " << kHeadDim << ", " << seq_len << "]},\n"
         << "    {\"name\": \"present_value\", \"file_suffix\": \".present_value_u8.bin\", \"dtype\": \"uint8\", \"shape\": [1, " << kKVHeads << ", " << seq_len << ", " << kHeadDim << "]}\n"
         << "  ]\n"
         << "}\n";
}

double percentile(const std::vector<int64_t>& sorted, double quantile) {
  const auto index = static_cast<size_t>(std::ceil(quantile * static_cast<double>(sorted.size()))) - 1;
  return static_cast<double>(sorted[std::min(index, sorted.size() - 1)]);
}

TimingResult runGraph(int seq_len, int warmup, int iterations, const std::string& dump_prefix) {
  const std::string workload = seq_len == 1 ? "s1_decode" : "s32_prefill_chunk";
  const std::string graph = "model.0.s" + std::to_string(seq_len);
  const int past_len = kContextLength - seq_len;

  QnnAOTModule module(graph);
  module.to(mllm::kQNN);

  auto hidden = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto sin = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto cos = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto mask = Tensor::empty({1, 1, seq_len, kContextLength}, mllm::kUInt16, mllm::kQNN).alloc();
  auto past_key = Tensor::empty({1, kKVHeads, kHeadDim, past_len}, mllm::kUInt8, mllm::kQNN).alloc();
  auto past_value = Tensor::empty({1, kKVHeads, past_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc();

  fillDeterministic<uint16_t>(hidden, 0x5101U + seq_len);
  fillDeterministic<uint16_t>(sin, 0x5102U + seq_len);
  fillDeterministic<uint16_t>(cos, 0x5103U + seq_len);
  std::fill(mask.ptr<uint16_t>(), mask.ptr<uint16_t>() + mask.numel(), UINT16_MAX);
  fillDeterministic<uint8_t>(past_key, 0x5104U + seq_len);
  fillDeterministic<uint8_t>(past_value, 0x5105U + seq_len);

  std::vector<Tensor> inputs = {hidden, sin, cos, mask, past_key, past_value};
  auto hidden_out = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto present_key = Tensor::empty({1, kKVHeads, kHeadDim, seq_len}, mllm::kUInt8, mllm::kQNN).alloc();
  auto present_value = Tensor::empty({1, kKVHeads, seq_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc();
  std::vector<Tensor> outputs = {hidden_out, present_key, present_value};
  module.setOutputTensors(outputs);

  for (int i = 0; i < warmup; ++i) { outputs = module(inputs); }

  std::vector<int64_t> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    outputs = module(inputs);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  }

  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  dumpGraphTensors(dump_prefix, workload, seq_len, past_len, inputs, outputs);
  return {
      .workload = workload,
      .graph = graph,
      .seq_len = seq_len,
      .past_len = past_len,
      .median_us = percentile(sorted, 0.5),
      .p95_us = percentile(sorted, 0.95),
      .mean_us = mean,
      .min_us = sorted.front(),
      .max_us = sorted.back(),
      .samples_us = std::move(samples),
  };
}

void writeJson(const std::string& path, const std::string& context, const std::string& variant, int warmup,
               int iterations, const std::vector<TimingResult>& results) {
  std::ofstream stream(path);
  if (!stream.is_open()) { throw std::runtime_error("cannot open output JSON: " + path); }
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"context\": \"" << context << "\",\n"
         << "  \"variant\": \"" << variant << "\",\n"
         << "  \"layer\": 5,\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"timing_boundary\": \"host wall time around QnnAOTModule/QnnGraph_execute\",\n"
         << "  \"results\": [\n";
  for (size_t result_index = 0; result_index < results.size(); ++result_index) {
    const auto& result = results[result_index];
    stream << "    {\n"
           << "      \"workload\": \"" << result.workload << "\",\n"
           << "      \"graph\": \"" << result.graph << "\",\n"
           << "      \"seq_len\": " << result.seq_len << ",\n"
           << "      \"past_len\": " << result.past_len << ",\n"
           << "      \"median_us\": " << result.median_us << ",\n"
           << "      \"p95_us\": " << result.p95_us << ",\n"
           << "      \"mean_us\": " << result.mean_us << ",\n"
           << "      \"min_us\": " << result.min_us << ",\n"
           << "      \"max_us\": " << result.max_us << ",\n"
           << "      \"samples_us\": [";
    for (size_t i = 0; i < result.samples_us.size(); ++i) {
      if (i != 0) { stream << ", "; }
      stream << result.samples_us[i];
    }
    stream << "]\n    }";
    if (result_index + 1 != results.size()) { stream << ','; }
    stream << '\n';
  }
  stream << "  ]\n}\n";
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context = Argparse::add<std::string>("-m|--model").help("Standalone Layer 5 QNN context path.");
  auto& graph = Argparse::add<std::string>("--graph").def("both").help("s1, s32, or both.");
  auto& warmup = Argparse::add<int>("--warmup").def(10).help("Warmup executions per graph.");
  auto& iterations = Argparse::add<int>("--iterations").def(100).help("Measured executions per graph.");
  auto& output = Argparse::add<std::string>("-o|--output").def("qwen3_layer5_block_timing.json");
  auto& variant = Argparse::add<std::string>("--variant").def("unknown");
  Argparse::parse(argc, argv);
  const auto* dump_prefix_env = std::getenv("MLLM_QWEN3_DUMP_PREFIX");
  const std::string dump_prefix = dump_prefix_env == nullptr ? "" : dump_prefix_env;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--model is required"); }
  if (graph.get() != "s1" && graph.get() != "s32" && graph.get() != "both") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--graph must be s1, s32, or both");
  }
  if (warmup.get() < 0 || iterations.get() <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "warmup must be non-negative and iterations must be positive");
  }

  mllm::initQnnBackend(context.get());
  std::vector<TimingResult> results;
  if (graph.get() == "s1" || graph.get() == "both") {
    results.push_back(runGraph(1, warmup.get(), iterations.get(), dump_prefix));
  }
  if (graph.get() == "s32" || graph.get() == "both") {
    results.push_back(runGraph(32, warmup.get(), iterations.get(), dump_prefix));
  }
  writeJson(output.get(), context.get(), variant.get(), warmup.get(), iterations.get(), results);
  for (const auto& result : results) {
    fmt::print("{} ({}): median={:.3f} ms p95={:.3f} ms mean={:.3f} ms\n", result.workload, result.graph,
               result.median_us / 1000.0, result.p95_us / 1000.0, result.mean_us / 1000.0);
  }
  fmt::print("wrote {}\n", output.get());
  return 0;
});
