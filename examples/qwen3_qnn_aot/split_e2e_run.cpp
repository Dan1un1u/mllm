// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>

#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::QNNAllocator;
using mllm::qnn::QNNBackend;

namespace {

constexpr int kContextLength = 1024;
constexpr int kHiddenSize = 2048;
constexpr int kHeadDim = 128;
constexpr int kKVHeads = 8;
constexpr int kVocabSize = 151936;

struct ChainResult {
  int seq_len = 0;
  std::string graph;
  std::vector<int64_t> samples_us;
  std::vector<uint64_t> boundary_fnv1a;
};

uint64_t fnv1a(const void* data, size_t bytes) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= ptr[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <typename T>
void fillDeterministic(Tensor& tensor, uint32_t seed) {
  auto* data = tensor.ptr<T>();
  uint32_t state = seed;
  for (size_t i = 0; i < tensor.numel(); ++i) {
    state = state * 1664525U + 1013904223U;
    data[i] = static_cast<T>((state >> 16U) & 0xFFU);
  }
}

uint16_t quantizeRope(double value) {
  // The split compiler uses uint16 asymmetric QDQ with scale 1/32768 and
  // zero-point 32768 for both sin and cos inputs.
  const auto rounded = std::nearbyint(value * 32768.0 + 32768.0);
  const auto clamped = std::max(0.0, std::min(65535.0, rounded));
  return static_cast<uint16_t>(clamped);
}

void fillRopeAndMask(Tensor& sin, Tensor& cos, Tensor& mask, int seq_len, int past_tokens = 0, int position_offset = 0) {
  constexpr double kRopeTheta = 1000000.0;
  constexpr int kHalfDim = kHeadDim / 2;
  for (int position = 0; position < seq_len; ++position) {
    for (int d = 0; d < kHalfDim; ++d) {
      const double inv_freq = std::pow(kRopeTheta, -static_cast<double>(d) / kHalfDim);
      const auto absolute_position = position_offset + position;
      const auto sin_value = quantizeRope(std::sin(absolute_position * inv_freq));
      const auto cos_value = quantizeRope(std::cos(absolute_position * inv_freq));
      sin.ptr<uint16_t>()[position * kHeadDim + d] = sin_value;
      sin.ptr<uint16_t>()[position * kHeadDim + d + kHalfDim] = sin_value;
      cos.ptr<uint16_t>()[position * kHeadDim + d] = cos_value;
      cos.ptr<uint16_t>()[position * kHeadDim + d + kHalfDim] = cos_value;
    }
  }

  std::fill(mask.ptr<uint16_t>(), mask.ptr<uint16_t>() + mask.numel(), uint16_t{0});
  for (int row = 0; row < seq_len; ++row) {
    auto* mask_row = mask.ptr<uint16_t>() + row * kContextLength;
    std::fill(mask_row, mask_row + past_tokens, uint16_t{65535});
    mask_row[kContextLength - seq_len + row] = uint16_t{65535};
  }
}

Tensor makePastKey(int seq_len, uint32_t seed) {
  (void)seed;
  const int past_len = kContextLength - seq_len;
  auto tensor = Tensor::empty({1, kKVHeads, kHeadDim, past_len}, mllm::kUInt8, mllm::kQNN).alloc();
  std::fill(tensor.ptr<uint8_t>(), tensor.ptr<uint8_t>() + tensor.numel(), uint8_t{0});
  return tensor;
}

Tensor makePastValue(int seq_len, uint32_t seed) {
  (void)seed;
  const int past_len = kContextLength - seq_len;
  auto tensor = Tensor::empty({1, kKVHeads, past_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc();
  std::fill(tensor.ptr<uint8_t>(), tensor.ptr<uint8_t>() + tensor.numel(), uint8_t{0});
  return tensor;
}

std::vector<Tensor> makePastInputs(int first_layer, int block_count, int seq_len) {
  std::vector<Tensor> caches;
  caches.reserve(static_cast<size_t>(2 * block_count));
  for (int i = 0; i < block_count; ++i) {
    const auto layer = static_cast<uint32_t>(first_layer + i);
    caches.push_back(makePastKey(seq_len, 0x7200U + layer));
    caches.push_back(makePastValue(seq_len, 0x7300U + layer));
  }
  return caches;
}

std::vector<Tensor> makePastOutputs(int block_count, int seq_len) {
  std::vector<Tensor> caches;
  caches.reserve(static_cast<size_t>(2 * block_count));
  for (int i = 0; i < block_count; ++i) {
    caches.push_back(Tensor::empty({1, kKVHeads, kHeadDim, seq_len}, mllm::kUInt8, mllm::kQNN).alloc());
  }
  for (int i = 0; i < block_count; ++i) {
    caches.push_back(Tensor::empty({1, kKVHeads, seq_len, kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
  }
  return caches;
}

double percentile(const std::vector<int64_t>& samples, double quantile) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const auto index = static_cast<size_t>(quantile * static_cast<double>(sorted.size()));
  return static_cast<double>(sorted[std::min(index, sorted.size() - 1)]);
}

struct ChainBuffers {
  std::vector<Tensor> part1_inputs;
  std::vector<Tensor> part2_inputs;
  std::vector<Tensor> part3_inputs;
  std::vector<Tensor> part4_inputs;
  std::vector<Tensor> part1_outputs;
  std::vector<Tensor> part2_outputs;
  std::vector<Tensor> part3_outputs;
  std::vector<Tensor> part4_outputs;
};

ChainBuffers makeBuffers(int seq_len) {
  ChainBuffers buffers;
  auto sequence = Tensor::empty({1, seq_len}, mllm::kInt32, mllm::kQNN).alloc();
  for (int i = 0; i < seq_len; ++i) { sequence.ptr<int32_t>()[i] = 1 + i; }
  buffers.part1_inputs.push_back(sequence);

  auto hidden1 = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto hidden2 = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto hidden3 = Tensor::empty({1, seq_len, kHiddenSize}, mllm::kUInt16, mllm::kQNN).alloc();
  auto logits = Tensor::empty({1, 1, seq_len, kVocabSize}, mllm::kUInt16, mllm::kQNN).alloc();
  buffers.part1_outputs.push_back(hidden1);
  buffers.part2_outputs.push_back(hidden2);
  buffers.part3_outputs.push_back(hidden3);
  buffers.part4_outputs.push_back(logits);

  auto sin = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto cos = Tensor::empty({1, seq_len, kHeadDim}, mllm::kUInt16, mllm::kQNN).alloc();
  auto mask = Tensor::empty({1, 1, seq_len, kContextLength}, mllm::kUInt16, mllm::kQNN).alloc();
  fillRopeAndMask(sin, cos, mask, seq_len);

  buffers.part2_inputs = {hidden1, sin, cos, mask};
  buffers.part3_inputs = {hidden2, sin, cos, mask};
  buffers.part4_inputs = {hidden3, sin, cos, mask};

  auto part2_past = makePastInputs(0, 10, seq_len);
  auto part3_past = makePastInputs(10, 10, seq_len);
  auto part4_past = makePastInputs(20, 8, seq_len);
  buffers.part2_inputs.insert(buffers.part2_inputs.end(), part2_past.begin(), part2_past.end());
  buffers.part3_inputs.insert(buffers.part3_inputs.end(), part3_past.begin(), part3_past.end());
  buffers.part4_inputs.insert(buffers.part4_inputs.end(), part4_past.begin(), part4_past.end());

  auto part2_cache_outputs = makePastOutputs(10, seq_len);
  auto part3_cache_outputs = makePastOutputs(10, seq_len);
  auto part4_cache_outputs = makePastOutputs(8, seq_len);
  buffers.part2_outputs.insert(buffers.part2_outputs.end(), part2_cache_outputs.begin(), part2_cache_outputs.end());
  buffers.part3_outputs.insert(buffers.part3_outputs.end(), part3_cache_outputs.begin(), part3_cache_outputs.end());
  buffers.part4_outputs.insert(buffers.part4_outputs.end(), part4_cache_outputs.begin(), part4_cache_outputs.end());
  return buffers;
}

void executeChain(const std::vector<std::shared_ptr<QNNBackend>>& backends, int seq_len, ChainBuffers& buffers) {
  const auto graph = "model.0.s" + std::to_string(seq_len);
  auto allocatorFor = [&](size_t index) {
    return std::static_pointer_cast<QNNAllocator>(backends.at(index)->allocator()).get();
  };
  backends.at(0)->graphExecute(graph, buffers.part1_inputs, buffers.part1_outputs, allocatorFor(0));
  backends.at(1)->graphExecute(graph, buffers.part2_inputs, buffers.part2_outputs, allocatorFor(1));
  backends.at(2)->graphExecute(graph, buffers.part3_inputs, buffers.part3_outputs, allocatorFor(2));
  backends.at(3)->graphExecute(graph, buffers.part4_inputs, buffers.part4_outputs, allocatorFor(3));
}

ChainResult runChain(const std::vector<std::shared_ptr<QNNBackend>>& backends, int seq_len, int warmup,
                     int iterations) {
  auto buffers = makeBuffers(seq_len);
  for (int i = 0; i < warmup; ++i) { executeChain(backends, seq_len, buffers); }

  ChainResult result;
  result.seq_len = seq_len;
  result.graph = "model.0.s" + std::to_string(seq_len);
  result.samples_us.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    executeChain(backends, seq_len, buffers);
    const auto end = std::chrono::steady_clock::now();
    result.samples_us.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  }

  result.boundary_fnv1a = {
      fnv1a(buffers.part1_outputs.at(0).ptr<uint16_t>(),
            buffers.part1_outputs.at(0).numel() * sizeof(uint16_t)),
      fnv1a(buffers.part2_outputs.at(0).ptr<uint16_t>(),
            buffers.part2_outputs.at(0).numel() * sizeof(uint16_t)),
      fnv1a(buffers.part3_outputs.at(0).ptr<uint16_t>(),
            buffers.part3_outputs.at(0).numel() * sizeof(uint16_t)),
      fnv1a(buffers.part4_outputs.at(0).ptr<uint16_t>(),
            buffers.part4_outputs.at(0).numel() * sizeof(uint16_t)),
  };
  return result;
}

void writeJson(const std::string& path, const std::vector<std::string>& contexts, int warmup, int iterations,
               const std::vector<ChainResult>& results) {
  std::ofstream stream(path);
  if (!stream.is_open()) { throw std::runtime_error("cannot open output JSON: " + path); }
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"contract\": \"qwen3_split_e2e_chain_probe_v1\",\n"
         << "  \"gate_status\": \"probe_only_not_full_model_gate\",\n"
         << "  \"timing_boundary\": \"host wall time around four direct QNNBackend::graphExecute calls\",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"contexts\": [";
  for (size_t i = 0; i < contexts.size(); ++i) {
    if (i != 0) { stream << ", "; }
    stream << "\"";
    stream << contexts[i];
    stream << "\"";
  }
  stream << "],\n  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    stream << "    {\n"
           << "      \"graph\": \"" << result.graph << "\",\n"
           << "      \"seq_len\": " << result.seq_len << ",\n"
           << "      \"median_us\": " << percentile(result.samples_us, 0.5) << ",\n"
           << "      \"p95_us\": " << percentile(result.samples_us, 0.95) << ",\n"
           << "      \"samples_us\": [";
    for (size_t j = 0; j < result.samples_us.size(); ++j) {
      if (j != 0) { stream << ", "; }
      stream << result.samples_us[j];
    }
    stream << "],\n      \"boundary_fnv1a\": [";
    for (size_t j = 0; j < result.boundary_fnv1a.size(); ++j) {
      if (j != 0) { stream << ", "; }
      stream << "\"";
      stream << fmt::format("{:016x}", result.boundary_fnv1a[j]);
      stream << "\"";
    }
    stream << "]\n    }";
    if (i + 1 != results.size()) { stream << ','; }
    stream << '\n';
  }
  stream << "  ]\n}\n";
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& part1 = Argparse::add<std::string>("--part1").help("Embedding context path.");
  auto& part2 = Argparse::add<std::string>("--part2").help("Layers 0-9 context path.");
  auto& part3 = Argparse::add<std::string>("--part3").help("Layers 10-19 context path.");
  auto& part4 = Argparse::add<std::string>("--part4").help("Layers 20-27 plus lm_head context path.");
  auto& graph = Argparse::add<std::string>("--graph").def("both").help("s1, s32, or both.");
  auto& warmup = Argparse::add<int>("--warmup").def(1).help("Warmup chains per graph.");
  auto& iterations = Argparse::add<int>("--iterations").def(3).help("Measured chains per graph.");
  auto& output = Argparse::add<std::string>("-o|--output").def("qwen3_split_e2e_chain_probe.json");
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!part1.isSet() || !part2.isSet() || !part3.isSet() || !part4.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--part1/--part2/--part3/--part4 are required");
  }
  if (graph.get() != "s1" && graph.get() != "s32" && graph.get() != "both") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--graph must be s1, s32, or both");
  }
  if (warmup.get() < 0 || iterations.get() <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "warmup must be non-negative and iterations positive");
  }

  const std::vector<std::string> contexts = {part1.get(), part2.get(), part3.get(), part4.get()};
  auto backends = std::vector<std::shared_ptr<QNNBackend>>();
  backends.reserve(contexts.size());
  for (const auto& context : contexts) {
    auto backend = std::make_shared<QNNBackend>();
    if (!backend->loadContext(context)) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kQnnError, "failed to load split context {}", context);
    }
    backends.push_back(std::move(backend));
  }

  // Tensor::empty(..., kQNN) must use the first allocator. The other three
  // allocators only register those shared buffers in their own QNN contexts.
  auto& context = mllm::Context::instance();
  context.registerBackend(backends.at(0));
  context.memoryManager()->registerAllocator(
      mllm::kQNN, backends.at(0)->allocator(),
      {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
  context.dispatcherManager()->registerDispatcher(
      mllm::qnn::createQNNDispatcher(context.dispatcherManager()->getExecutor(), {}));

  std::vector<ChainResult> results;
  if (graph.get() == "s1" || graph.get() == "both") {
    results.push_back(runChain(backends, 1, warmup.get(), iterations.get()));
  }
  if (graph.get() == "s32" || graph.get() == "both") {
    results.push_back(runChain(backends, 32, warmup.get(), iterations.get()));
  }
  writeJson(output.get(), contexts, warmup.get(), iterations.get(), results);

  for (const auto& result : results) {
    fmt::print("{}: median={:.3f} ms p95={:.3f} ms boundary=[",
               result.graph, percentile(result.samples_us, 0.5) / 1000.0,
               percentile(result.samples_us, 0.95) / 1000.0);
    for (size_t i = 0; i < result.boundary_fnv1a.size(); ++i) {
      if (i != 0) { fmt::print(", "); }
      fmt::print("{:016x}", result.boundary_fnv1a[i]);
    }
    fmt::print("]\n");
  }
  fmt::print("wrote {} (probe-only; original numerical/performance gates remain unchanged)\n", output.get());
  return 0;
});
