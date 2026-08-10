#pragma once

#include <string>
#include <unordered_map>

#include <mllm/mllm.hpp>
#include <mllm/models/ARGeneration.hpp>
#include <mllm/models/qwen3/configuration_qwen3.hpp>

namespace qwen3_qnn_aot::block {

template <typename ParamsT>
inline void attachQDQ(mllm::Tensor& tensor, const ParamsT& params, const std::string& prefix) {
  tensor.attach("scale", params->pull(prefix + ".fake_quant.scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".fake_quant.zero_point").impl(), true);
}

template <typename ParamsT>
inline mllm::models::ARGenerationOutputPast makeTraceInputs(
    int seq_len, int context_len, const mllm::models::qwen3::Qwen3Config& cfg, const ParamsT& params,
    int first_layer, int block_count = 1) {
  const std::string first_prefix = "model.layers." + std::to_string(first_layer);

  auto hidden = mllm::Tensor::zeros({1, seq_len, cfg.hidden_size}, mllm::kUInt16);
  hidden = hidden.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  attachQDQ(hidden, params, first_prefix + ".input_layernorm_input_qdq");
  hidden.setName("hidden_states");

  auto sin = mllm::Tensor::zeros({1, seq_len, cfg.head_dim}, mllm::kUInt16);
  sin = sin.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  attachQDQ(sin, params, "model.sin_embedding_input_qdq");
  sin.setName("sin");

  auto cos = mllm::Tensor::zeros({1, seq_len, cfg.head_dim}, mllm::kUInt16);
  cos = cos.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  attachQDQ(cos, params, "model.cos_embedding_input_qdq");
  cos.setName("cos");

  auto causal_mask = mllm::Tensor::zeros({1, 1, seq_len, context_len}, mllm::kUInt16);
  causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
  causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
  causal_mask.setName("causal_mask");

  mllm::models::ARGenerationOutputPast inputs = {
      {"hidden_states", hidden}, {"sin", sin}, {"cos", cos}, {"causal_mask", causal_mask}};
  for (int offset = 0; offset < block_count; ++offset) {
    const int layer = first_layer + offset;
    const std::string prefix = "model.layers." + std::to_string(layer);
    const auto suffix = block_count == 1 ? std::string() : "_" + std::to_string(layer);

    auto past_key = mllm::Tensor::zeros(
        {1, cfg.num_key_value_heads, cfg.head_dim, context_len - seq_len}, mllm::kUInt8);
    past_key = past_key.__unsafeSetDType(mllm::kUInt8PerTensorSym);
    attachQDQ(past_key, params, prefix + ".self_attn.k_cast_to_int8_qdq");
    past_key.setName("past_key" + suffix);
    inputs["past_key" + suffix] = past_key;

    auto past_value = mllm::Tensor::zeros(
        {1, cfg.num_key_value_heads, context_len - seq_len, cfg.head_dim}, mllm::kUInt8);
    past_value = past_value.__unsafeSetDType(mllm::kUInt8PerTensorSym);
    attachQDQ(past_value, params, prefix + ".self_attn.v_cast_to_int8_qdq");
    past_value.setName("past_value" + suffix);
    inputs["past_value" + suffix] = past_value;
  }
  return inputs;
}

}  // namespace qwen3_qnn_aot::block
