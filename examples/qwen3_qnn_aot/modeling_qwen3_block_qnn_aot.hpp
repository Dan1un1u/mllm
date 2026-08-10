#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mllm/compile/ir/Trace.hpp>
#include <mllm/models/ARGeneration.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"

namespace mllm::models::qwen3::block_aot {

class Qwen3BlockRoot final : public nn::Module {
 public:
  Qwen3BlockRoot() = default;

  Qwen3BlockRoot(const std::string& name, const Qwen3Config& cfg, int layer_idx, sha::R3Mode r3_mode)
      : nn::Module(name) {
    // Keep the production-style symbol path so existing Qwen3 profiler
    // classification continues to recognize every operation as Layer 5.
    block_ = reg<sha::Qwen3DecoderSHA>("layers." + std::to_string(layer_idx), cfg, layer_idx, r3_mode);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return block_(inputs);
  }

 private:
  sha::Qwen3DecoderSHA block_;
};

class Qwen3StandaloneBlock final : public ARGeneration, public nn::Module {
 public:
  Qwen3StandaloneBlock(const Qwen3Config& cfg, int layer_idx, sha::R3Mode r3_mode = sha::R3Mode::kNone) {
    root_ = reg<Qwen3BlockRoot>("model", cfg, layer_idx, r3_mode);
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::lowlevel::traceStart();
    auto outputs = root_(input.at("hidden_states"), input.at("sin"), input.at("cos"), input.at("causal_mask"),
                         input.at("past_key"), input.at("past_value"));

    // In the full model these boundary quantization specs are solved by the
    // next decoder layer and by the KV-cache consumers. A standalone block has
    // no such consumers, so carry the existing boundary QDQ metadata onto its
    // three graph outputs without inserting an executable identity/QDQ op.
    const auto carry_qdq = [](Tensor& output, Tensor boundary) {
      output.attach("scale", boundary.getExtraTensorViewInTensor("scale").impl(), true);
      if (boundary.hasAttachedView("zero_point")) {
        output.attach("zero_point", boundary.getExtraTensorViewInTensor("zero_point").impl(), true);
      }
    };
    carry_qdq(outputs.at(0), input.at("hidden_states"));
    carry_qdq(outputs.at(1), input.at("past_key"));
    carry_qdq(outputs.at(2), input.at("past_value"));

    auto block_ir = ir::lowlevel::traceStop();
    return {{"model", block_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    return {};
  }

 private:
  Qwen3BlockRoot root_;
};

}  // namespace mllm::models::qwen3::block_aot
