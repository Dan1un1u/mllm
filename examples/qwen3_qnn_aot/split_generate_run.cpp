// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>

#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::QNNAllocator;
using mllm::qnn::QNNBackend;

namespace {

constexpr int kContextLength = 1024;
constexpr int kPastLength = kContextLength - 1;
constexpr int kHiddenSize = 2048;
constexpr int kHeadDim = 128;
constexpr int kKVHeads = 8;
constexpr int kVocabSize = 151936;

uint64_t fnv1a(const void* data, size_t bytes) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= ptr[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint16_t quantizeRope(double value) {
  const auto rounded = std::nearbyint(value * 32768.0 + 32768.0);
  const auto clamped = std::max(0.0, std::min(65535.0, rounded));
  return static_cast<uint16_t>(clamped);
}

void fillRopeAndMask(Tensor& sin, Tensor& cos, Tensor& mask, int position, int past_tokens) {
  constexpr double kRopeTheta = 1000000.0;
  constexpr int kHalfDim = kHeadDim / 2;
  for (int d = 0; d < kHalfDim; ++d) {
    const double inv_freq = std::pow(kRopeTheta, -static_cast<double>(d) / kHalfDim);
    const auto sin_value = quantizeRope(std::sin(position * inv_freq));
    const auto cos_value = quantizeRope(std::cos(position * inv_freq));
    sin.ptr<uint16_t>()[d] = sin_value;
    sin.ptr<uint16_t>()[d + kHalfDim] = sin_value;
    cos.ptr<uint16_t>()[d] = cos_value;
    cos.ptr<uint16_t>()[d + kHalfDim] = cos_value;
  }

  auto* mask_ptr = mask.ptr<uint16_t>();
  std::fill(mask_ptr, mask_ptr + mask.numel(), uint16_t{0});
  std::fill(mask_ptr, mask_ptr + past_tokens, uint16_t{65535});
  mask_ptr[kContextLength - 1] = uint16_t{65535};
}

void appendKey(Tensor& cache, const Tensor& present, int past_tokens) {
  auto* dst = cache.ptr<uint8_t>();
  const auto* src = present.ptr<uint8_t>();
  for (int h = 0; h < kKVHeads; ++h) {
    for (int d = 0; d < kHeadDim; ++d) {
      dst[(h * kHeadDim + d) * kPastLength + past_tokens] = src[h * kHeadDim + d];
    }
  }
}

void appendValue(Tensor& cache, const Tensor& present, int past_tokens) {
  auto* dst = cache.ptr<uint8_t>();
  const auto* src = present.ptr<uint8_t>();
  for (int h = 0; h < kKVHeads; ++h) {
    for (int d = 0; d < kHeadDim; ++d) {
      dst[(h * kPastLength + past_tokens) * kHeadDim + d] = src[h * kHeadDim + d];
    }
  }
}

struct PartIO {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<Tensor> key_inputs;
  std::vector<Tensor> value_inputs;
};

class SplitGenerator {
 public:
  SplitGenerator(const std::vector<std::string>& contexts) {
    backends_.reserve(contexts.size());
    for (const auto& path : contexts) {
      auto backend = std::make_shared<QNNBackend>();
      if (!backend->loadContext(path)) { throw std::runtime_error("failed to load context: " + path); }
      backends_.push_back(std::move(backend));
    }

    auto& context = mllm::Context::instance();
    context.registerBackend(backends_.at(0));
    context.memoryManager()->registerAllocator(
        mllm::kQNN, backends_.at(0)->allocator(),
        {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
    context.dispatcherManager()->registerDispatcher(
        mllm::qnn::createQNNDispatcher(context.dispatcherManager()->getExecutor(), {}));
    initIO();
  }

  uint32_t runToken(uint32_t token, int position) {
    if (position != past_tokens_) { throw std::runtime_error("non-contiguous token position"); }
    if (past_tokens_ >= kPastLength) { throw std::runtime_error("context length exceeded"); }

    part1_inputs_.at(0).ptr<int32_t>()[0] = static_cast<int32_t>(token);
    fillRopeAndMask(sin_, cos_, mask_, position, past_tokens_);

    execute(0, "model.0.s1", part1_inputs_, part1_outputs_);
    execute(1, "model.0.s1", part2_inputs_.inputs, part2_inputs_.outputs);
    execute(2, "model.0.s1", part3_inputs_.inputs, part3_inputs_.outputs);
    execute(3, "model.0.s1", part4_inputs_.inputs, part4_inputs_.outputs);

    appendCaches(part2_inputs_, part2_outputs_, 10, past_tokens_);
    appendCaches(part3_inputs_, part3_outputs_, 10, past_tokens_);
    appendCaches(part4_inputs_, part4_outputs_, 8, past_tokens_);
    ++past_tokens_;

    const auto* logits = part4_outputs_.at(0).ptr<uint16_t>();
    return static_cast<uint32_t>(std::distance(logits, std::max_element(logits, logits + kVocabSize)));
  }

  uint64_t hiddenHash() const {
    return fnv1a(part4_inputs_.inputs.at(0).ptr<uint16_t>(),
                 part4_inputs_.inputs.at(0).numel() * sizeof(uint16_t));
  }

  uint64_t logitsHash() const {
    return fnv1a(part4_outputs_.at(0).ptr<uint16_t>(),
                 part4_outputs_.at(0).numel() * sizeof(uint16_t));
  }

 private:
  static void appendCaches(PartIO& io, const std::vector<Tensor>& outputs, int block_count, int past_tokens) {
    for (int i = 0; i < block_count; ++i) {
      appendKey(io.key_inputs.at(static_cast<size_t>(i)), outputs.at(static_cast<size_t>(1 + i)),
                past_tokens);
      appendValue(io.value_inputs.at(static_cast<size_t>(i)),
                  outputs.at(static_cast<size_t>(1 + block_count + i)), past_tokens);
    }
  }

  void execute(size_t index, const std::string& graph, std::vector<Tensor>& inputs,
               std::vector<Tensor>& outputs) {
    auto allocator = std::static_pointer_cast<QNNAllocator>(backends_.at(index)->allocator());
    backends_.at(index)->graphExecute(graph, inputs, outputs, allocator.get());
  }

  static Tensor makeTensor(const std::vector<int32_t>& shape, mllm::DataTypes dtype) {
    return Tensor::empty(shape, dtype, mllm::kQNN).alloc();
  }

  static void initCacheTensor(Tensor& tensor) {
    std::fill(tensor.ptr<uint8_t>(), tensor.ptr<uint8_t>() + tensor.numel(), uint8_t{0});
  }

  void initPart(PartIO& io, int block_count, Tensor& hidden) {
    io.inputs = {hidden, sin_, cos_, mask_};
    io.key_inputs.reserve(block_count);
    io.value_inputs.reserve(block_count);
    for (int i = 0; i < block_count; ++i) {
      auto key = makeTensor({1, kKVHeads, kHeadDim, kPastLength}, mllm::kUInt8);
      auto value = makeTensor({1, kKVHeads, kPastLength, kHeadDim}, mllm::kUInt8);
      initCacheTensor(key);
      initCacheTensor(value);
      io.key_inputs.push_back(key);
      io.value_inputs.push_back(value);
      io.inputs.push_back(key);
      io.inputs.push_back(value);
    }
    io.outputs.push_back(hidden);
    for (int i = 0; i < block_count; ++i) {
      io.outputs.push_back(makeTensor({1, kKVHeads, kHeadDim, 1}, mllm::kUInt8));
    }
    for (int i = 0; i < block_count; ++i) {
      io.outputs.push_back(makeTensor({1, kKVHeads, 1, kHeadDim}, mllm::kUInt8));
    }
  }

  void initIO() {
    auto sequence = makeTensor({1, 1}, mllm::kInt32);
    part1_inputs_ = {sequence};
    auto hidden1 = makeTensor({1, 1, kHiddenSize}, mllm::kUInt16);
    auto hidden2 = makeTensor({1, 1, kHiddenSize}, mllm::kUInt16);
    auto hidden3 = makeTensor({1, 1, kHiddenSize}, mllm::kUInt16);
    auto logits = makeTensor({1, 1, 1, kVocabSize}, mllm::kUInt16);
    part1_outputs_ = {hidden1};

    sin_ = makeTensor({1, 1, kHeadDim}, mllm::kUInt16);
    cos_ = makeTensor({1, 1, kHeadDim}, mllm::kUInt16);
    mask_ = makeTensor({1, 1, 1, kContextLength}, mllm::kUInt16);
    initPart(part2_inputs_, 10, hidden1);
    initPart(part3_inputs_, 10, hidden2);
    initPart(part4_inputs_, 8, hidden3);
    part2_outputs_ = part2_inputs_.outputs;
    part3_outputs_ = part3_inputs_.outputs;
    part4_inputs_.outputs.at(0) = logits;
    part4_outputs_ = part4_inputs_.outputs;
  }

  std::vector<std::shared_ptr<QNNBackend>> backends_;
  std::vector<Tensor> part1_inputs_;
  std::vector<Tensor> part1_outputs_;
  PartIO part2_inputs_;
  PartIO part3_inputs_;
  PartIO part4_inputs_;
  std::vector<Tensor> part2_outputs_;
  std::vector<Tensor> part3_outputs_;
  std::vector<Tensor> part4_outputs_;
  Tensor sin_;
  Tensor cos_;
  Tensor mask_;
  int past_tokens_ = 0;
};

void writeJson(const std::string& path, const std::vector<int64_t>& prompt_tokens,
               const std::vector<uint32_t>& generated_tokens, const std::vector<int64_t>& token_us,
               uint64_t hidden_hash, uint64_t logits_hash) {
  std::ofstream stream(path);
  if (!stream.is_open()) { throw std::runtime_error("cannot open output JSON: " + path); }
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"contract\": \"qwen3_split_e2e_generation_probe_v1\",\n"
         << "  \"gate_status\": \"probe_only_not_full_model_gate\",\n"
         << "  \"numerical_oracle\": \"not_run\",\n"
         << "  \"performance_gate\": \"not_run\",\n"
         << "  \"prompt_token_count\": " << prompt_tokens.size() << ",\n"
         << "  \"generated_tokens\": [";
  for (size_t i = 0; i < generated_tokens.size(); ++i) {
    if (i != 0) { stream << ", "; }
    stream << generated_tokens[i];
  }
  stream << "],\n  \"token_wall_us\": [";
  for (size_t i = 0; i < token_us.size(); ++i) {
    if (i != 0) { stream << ", "; }
    stream << token_us[i];
  }
  stream << "],\n"
         << "  \"last_hidden_fnv1a\": \"" << fmt::format("{:016x}", hidden_hash) << "\",\n"
         << "  \"last_logits_fnv1a\": \"" << fmt::format("{:016x}", logits_hash) << "\"\n"
         << "}\n";
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& part1 = Argparse::add<std::string>("--part1").help("Embedding context path.");
  auto& part2 = Argparse::add<std::string>("--part2").help("Layers 0-9 context path.");
  auto& part3 = Argparse::add<std::string>("--part3").help("Layers 10-19 context path.");
  auto& part4 = Argparse::add<std::string>("--part4").help("Layers 20-27 plus lm_head context path.");
  auto& tokenizer_path = Argparse::add<std::string>("--tokenizer").help("Qwen3 tokenizer JSON path.");
  auto& prompt = Argparse::add<std::string>("--prompt").def("hello").help("User prompt.");
  auto& max_new_tokens = Argparse::add<int>("--max_new_tokens").def(1);
  auto& output = Argparse::add<std::string>("-o|--output").def("qwen3_split_generation_probe.json");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!part1.isSet() || !part2.isSet() || !part3.isSet() || !part4.isSet() || !tokenizer_path.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--part1/--part2/--part3/--part4/--tokenizer are required");
  }
  if (max_new_tokens.get() <= 0) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "max_new_tokens must be positive"); }

  mllm::models::qwen3::Qwen3Tokenizer tokenizer(tokenizer_path.get());
  auto message = tokenizer.convertMessage({.prompt = prompt.get()});
  std::vector<int64_t> prompt_tokens;
  prompt_tokens.reserve(static_cast<size_t>(message.at("sequence").shape()[1]));
  for (int i = 0; i < message.at("sequence").shape()[1]; ++i) {
    prompt_tokens.push_back(message.at("sequence").ptr<int64_t>()[i]);
  }
  if (prompt_tokens.empty() || prompt_tokens.size() >= kPastLength) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "prompt token count must be in [1, 1022]");
  }

  SplitGenerator generator({part1.get(), part2.get(), part3.get(), part4.get()});
  std::vector<uint32_t> generated_tokens;
  std::vector<int64_t> token_us;
  uint32_t next_token = 0;
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    const auto begin = std::chrono::steady_clock::now();
    next_token = generator.runToken(static_cast<uint32_t>(prompt_tokens[i]), static_cast<int>(i));
    const auto end = std::chrono::steady_clock::now();
    token_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  }
  generated_tokens.push_back(next_token);
  fmt::print("{}", mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(next_token)));
  for (int i = 1; i < max_new_tokens.get(); ++i) {
    const auto begin = std::chrono::steady_clock::now();
    next_token = generator.runToken(next_token, static_cast<int>(prompt_tokens.size() + i - 1));
    const auto end = std::chrono::steady_clock::now();
    token_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    generated_tokens.push_back(next_token);
    fmt::print("{}", mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(next_token)));
  }
  fmt::print("\n");
  writeJson(output.get(), prompt_tokens, generated_tokens, token_us, generator.hiddenHash(), generator.logitsHash());
  fmt::print("wrote {} (probe-only; numerical and 5% performance gates unchanged)\n", output.get());
  return 0;
});
