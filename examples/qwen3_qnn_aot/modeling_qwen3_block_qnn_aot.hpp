#pragma once

#include <string>
#include <vector>

#include <mllm/compile/ir/Trace.hpp>
#include <mllm/models/ARGeneration.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"

namespace mllm::models::qwen3::block_aot {

class Qwen3BlockRoot final : public nn::Module {
 public:
  Qwen3BlockRoot() = default;

  Qwen3BlockRoot(const std::string& name, const Qwen3Config& cfg, int first_layer, int block_count,
                 sha::R3Mode r3_mode)
      : nn::Module(name) {
    for (int offset = 0; offset < block_count; ++offset) {
      const int layer = first_layer + offset;
      blocks_.emplace_back(
          reg<sha::Qwen3DecoderSHA>("layers." + std::to_string(layer), cfg, r3_mode, layer));
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs.at(0);
    std::vector<Tensor> outputs;
    outputs.reserve(1 + 2 * blocks_.size());
    for (size_t offset = 0; offset < blocks_.size(); ++offset) {
      const size_t cache = 4 + 2 * offset;
      auto block_outputs = blocks_[offset](hidden, inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(cache),
                                           inputs.at(cache + 1));
      hidden = block_outputs.at(0);
      outputs.push_back(block_outputs.at(1));
      outputs.push_back(block_outputs.at(2));
    }
    outputs.insert(outputs.begin(), hidden);
    return outputs;
  }

 private:
  std::vector<sha::Qwen3DecoderSHA> blocks_;
};

class Qwen3StandaloneBlock final : public ARGeneration, public nn::Module {
 public:
  Qwen3StandaloneBlock(const Qwen3Config& cfg, int first_layer, int block_count = 1,
                       sha::R3Mode r3_mode = sha::R3Mode::kNone)
      : first_layer_(first_layer), block_count_(block_count) {
    root_ = reg<Qwen3BlockRoot>("model", cfg, first_layer, block_count, r3_mode);
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::lowlevel::traceStart();
    std::vector<Tensor> root_inputs = {
        input.at("hidden_states"), input.at("sin"), input.at("cos"), input.at("causal_mask")};
    for (int offset = 0; offset < block_count_; ++offset) {
      const auto suffix = block_count_ == 1 ? std::string() : "_" + std::to_string(first_layer_ + offset);
      root_inputs.push_back(input.at("past_key" + suffix));
      root_inputs.push_back(input.at("past_value" + suffix));
    }
    auto outputs = root_(root_inputs);

    const auto carry_qdq = [](Tensor& output, Tensor boundary) {
      output.attach("scale", boundary.getExtraTensorViewInTensor("scale").impl(), true);
      if (boundary.hasAttachedView("zero_point")) {
        output.attach("zero_point", boundary.getExtraTensorViewInTensor("zero_point").impl(), true);
      }
    };
    carry_qdq(outputs.at(0), input.at("hidden_states"));
    for (int offset = 0; offset < block_count_; ++offset) {
      const auto suffix = block_count_ == 1 ? std::string() : "_" + std::to_string(first_layer_ + offset);
      carry_qdq(outputs.at(1 + 2 * offset), input.at("past_key" + suffix));
      carry_qdq(outputs.at(2 + 2 * offset), input.at("past_value" + suffix));
    }

    auto block_ir = ir::lowlevel::traceStop();
    return {{"model", block_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    return {};
  }

 private:
  Qwen3BlockRoot root_;
  int first_layer_ = 0;
  int block_count_ = 1;
};

}  // namespace mllm::models::qwen3::block_aot
