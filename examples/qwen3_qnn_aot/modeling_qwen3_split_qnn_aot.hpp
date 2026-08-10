#pragma once

#include <string>
#include <vector>

#include <mllm/compile/ir/Trace.hpp>
#include <mllm/models/ARGeneration.hpp>

#include "modeling_qwen3_block_qnn_aot.hpp"

namespace mllm::models::qwen3::split_aot {

using sha::vi32;

class Qwen3SplitRoot final : public nn::Module {
 public:
  Qwen3SplitRoot() = default;

  Qwen3SplitRoot(const std::string& name, const Qwen3Config& cfg, int part_id, int first_layer,
                 int block_count, sha::R3Mode r3_mode, sha::R1BoundaryMode r1_boundary_mode)
      : nn::Module(name), part_id_(part_id), first_layer_(first_layer), block_count_(block_count),
        hidden_size_(cfg.hidden_size), r1_boundary_mode_(r1_boundary_mode) {
    if (part_id_ == 1) {
      embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
      if (r1_boundary_mode_ == sha::R1BoundaryMode::kOnline) {
        r1_dense_ = reg<nn::Param>("r1_dense", "model.r1_dense.weight",
                                   Tensor::shape_t{cfg.hidden_size, cfg.hidden_size});
      }
      return;
    }

    blocks_.reserve(static_cast<size_t>(block_count_));
    for (int offset = 0; offset < block_count_; ++offset) {
      const int layer = first_layer_ + offset;
      blocks_.emplace_back(
          reg<sha::Qwen3DecoderSHA>("layers." + std::to_string(layer), cfg, r3_mode, layer));
    }

    if (part_id_ == 4) {
      norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
      if (r1_boundary_mode_ == sha::R1BoundaryMode::kOnline) {
        r1_dense_ = reg<nn::Param>("r1_dense", "model.r1_dense.weight",
                                   Tensor::shape_t{cfg.hidden_size, cfg.hidden_size});
      }
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs,
                              const std::vector<AnyValue>& args) override {
    if (part_id_ == 1) {
      auto hidden = embedding_(inputs.at(0));
      if (r1_boundary_mode_ == sha::R1BoundaryMode::kOnline) {
        hidden = nn::functional::matmul(hidden, r1_dense_());
      }
      return {hidden};
    }

    auto hidden = inputs.at(0);
    const auto& sin = inputs.at(1);
    const auto& cos = inputs.at(2);
    const auto& causal_mask = inputs.at(3);

    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    keys.reserve(static_cast<size_t>(block_count_));
    values.reserve(static_cast<size_t>(block_count_));
    for (int offset = 0; offset < block_count_; ++offset) {
      const size_t cache = 4 + 2 * static_cast<size_t>(offset);
      auto block_outputs = blocks_.at(static_cast<size_t>(offset))(
          hidden, sin, cos, causal_mask, inputs.at(cache), inputs.at(cache + 1));
      hidden = block_outputs.at(0);
      keys.push_back(block_outputs.at(1));
      values.push_back(block_outputs.at(2));
    }

    if (part_id_ == 4) {
      if (r1_boundary_mode_ == sha::R1BoundaryMode::kOnline) {
        hidden = nn::functional::matmul(hidden, r1_dense_());
      }
      hidden = sha::ptq::QDQ(this, hidden, "norm_input_qdq");
      hidden = norm_(hidden);
      hidden = hidden.view({1, 1, -1, hidden_size_}, true);
    }

    std::vector<Tensor> outputs;
    outputs.reserve(1 + 2 * static_cast<size_t>(block_count_));
    outputs.push_back(hidden);
    outputs.insert(outputs.end(), keys.begin(), keys.end());
    outputs.insert(outputs.end(), values.begin(), values.end());
    return outputs;
  }

 private:
  int part_id_ = 0;
  int first_layer_ = 0;
  int block_count_ = 0;
  int hidden_size_ = 2048;
  sha::R1BoundaryMode r1_boundary_mode_ = sha::R1BoundaryMode::kNone;
  std::vector<sha::Qwen3DecoderSHA> blocks_;
  nn::Embedding embedding_;
  nn::RMSNorm norm_;
  nn::Param r1_dense_;
};

class Qwen3SplitPart final : public ARGeneration, public nn::Module {
 public:
  Qwen3SplitPart(const Qwen3Config& cfg, int part_id, int first_layer, int block_count,
                 sha::R3Mode r3_mode, sha::R1BoundaryMode r1_boundary_mode)
      : cfg_(cfg), part_id_(part_id), first_layer_(first_layer), block_count_(block_count) {
    root_ = reg<Qwen3SplitRoot>("model", cfg, part_id, first_layer, block_count, r3_mode,
                                r1_boundary_mode);
    if (part_id == 4) {
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::lowlevel::traceStart();

    if (part_id_ == 1) {
      auto outputs = root_(std::vector<Tensor>{input.at("sequence")});
      attachBoundaryQDQ(outputs.at(0), "model.embed_tokens");
      auto part_ir = ir::lowlevel::traceStop();
      return {{"model", part_ir}};
    }

    std::vector<Tensor> root_inputs = {input.at("hidden_states"), input.at("sin"), input.at("cos"),
                                       input.at("causal_mask")};
    for (int offset = 0; offset < block_count_; ++offset) {
      const int layer = first_layer_ + offset;
      const std::string suffix = block_count_ == 1 ? std::string() : "_" + std::to_string(layer);
      root_inputs.push_back(input.at("past_key" + suffix));
      root_inputs.push_back(input.at("past_value" + suffix));
    }

    auto outputs = root_(root_inputs);
    if (part_id_ == 4) {
      auto final_hidden = sha::ptq::QDQ(this, outputs.at(0), "lm_head_input_qdq");
      final_hidden = lm_head_(final_hidden);
      outputs[0] = sha::ptq::QDQ(this, final_hidden, "lm_head_output_qdq");
    }
    if (part_id_ != 4 && first_layer_ + block_count_ < cfg_.num_hidden_layers) {
      attachBoundaryQDQ(outputs.at(0), "model.layers." +
                                         std::to_string(first_layer_ + block_count_) +
                                         ".input_layernorm_input_qdq");
    }
    for (int offset = 0; offset < block_count_; ++offset) {
      const int layer = first_layer_ + offset;
      attachBoundaryQDQ(outputs.at(1 + offset),
                        "model.layers." + std::to_string(layer) +
                            ".self_attn.k_cast_to_int8_qdq");
      attachBoundaryQDQ(outputs.at(1 + block_count_ + offset),
                        "model.layers." + std::to_string(layer) +
                            ".self_attn.v_cast_to_int8_qdq");
    }

    auto part_ir = ir::lowlevel::traceStop();
    return {{"model", part_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input,
                                 const ARGenerationArgs& args) override {
    return {};
  }

 private:
  void attachBoundaryQDQ(Tensor& tensor, const std::string& prefix) {
    auto params = getTopParameterFile();
    tensor.attach("scale", params->pull(prefix == "model.embed_tokens" ? prefix + ".scale" : prefix + ".fake_quant.scale").impl(), true);
    tensor.attach("zero_point", params->pull(prefix == "model.embed_tokens" ? prefix + ".zero_point" : prefix + ".fake_quant.zero_point").impl(), true);
  }

  const Qwen3Config& cfg_;
  int part_id_ = 0;
  int first_layer_ = 0;
  int block_count_ = 0;
  nn::Conv2D lm_head_;
  Qwen3SplitRoot root_;
};

}  // namespace mllm::models::qwen3::split_aot
