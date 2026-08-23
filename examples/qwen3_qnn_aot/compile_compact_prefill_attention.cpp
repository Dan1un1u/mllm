// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile the layer-14 native QNN attention core for the first s32 prefill
// chunk.  The full-width control consumes the accepted 1024-column cache
// layout; the compact candidate contains only the 32 current tokens.  QNN
// MatMul, masked Softmax, qparams, head count, and graph outputs are otherwise
// identical.

#include <array>
#include <cmath>
#include <string_view>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kSeq = 32;
constexpr int32_t kContext = 1024;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kQueryHeads = 16;
constexpr int32_t kKvHeads = 8;
constexpr int32_t kQueryHeadsPerKvHead = kQueryHeads / kKvHeads;
constexpr int32_t kLayer = 14;
constexpr std::string_view kLayerPath = "layers.14.self_attn.";

std::string sourcePrefix(std::string_view name) {
  return "model.layers." + std::to_string(kLayer) + ".self_attn." +
         std::string(name) + ".fake_quant";
}

void copyPerHeadQparams(const mllm::ParameterFile::ptr_t& params) {
  constexpr std::array<std::string_view, 9> kQdqNames = {
      "qk_matmul_output_qdq", "scaling_qdq", "mul_0_output_qdq",
      "reduce_min_output_qdq", "neg_20_qdq", "minus_0_output_qdq",
      "where_attn_qdq", "softmax_output_qdq",
      "attn_value_matmul_output_qdq",
  };
  const std::string layer_prefix =
      "model.layers." + std::to_string(kLayer) + ".self_attn.";
  for (const auto name : kQdqNames) {
    for (const auto suffix : {std::string_view{"scale"},
                              std::string_view{"zero_point"}}) {
      const auto source = layer_prefix + std::string(name) +
                          ".fake_quant." + std::string(suffix);
      auto value = params->pull(source);
      for (int32_t head = 0; head < kQueryHeads; ++head) {
        const auto destination = layer_prefix + std::string(name) + "_h" +
                                 std::to_string(head) + ".fake_quant." +
                                 std::string(suffix);
        params->push(destination,
                     value.contiguous()
                         .setMemType(mllm::kParamsNormal)
                         .setName(destination));
      }
    }
  }
}

void attachQparams(mllm::Tensor& tensor,
                   const mllm::ParameterFile::ptr_t& params,
                   const std::string& prefix, mllm::DataTypes dtype) {
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(),
                true);
}

class CompactPrefillAttention final : public mllm::nn::Module {
 public:
  explicit CompactPrefillAttention(const std::string& name)
      : mllm::nn::Module(name) {}

  std::vector<mllm::Tensor> forward(
      const std::vector<mllm::Tensor>& inputs,
      const std::vector<mllm::AnyValue>& /*args*/) override {
    constexpr size_t kExpectedInputs = kQueryHeads + 2 * kKvHeads + 1;
    MLLM_RT_ASSERT_EQ(inputs.size(), kExpectedInputs);
    auto causal_mask = inputs.back();
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kQueryHeads);

    for (int32_t head = 0; head < kQueryHeads; ++head) {
      const auto suffix = std::to_string(head);
      const int32_t kv_head = head / kQueryHeadsPerKvHead;
      const auto& query = inputs[head];
      const auto& key = inputs[kQueryHeads + kv_head];
      const auto& value = inputs[kQueryHeads + kKvHeads + kv_head];

      auto attn = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(query, key),
          std::string(kLayerPath) + "qk_matmul_output_qdq_h" + suffix);

      auto scale = mllm::Tensor::constant(
          1.0f / std::sqrt(static_cast<float>(kHeadDim)), mllm::kFloat32);
      scale = mllm::models::qwen3::sha::ptq::QDQ(
          this, scale,
          std::string(kLayerPath) + "scaling_qdq_h" + suffix);
      attn = mllm::models::qwen3::sha::ptq::QDQ(
          this, attn.mulConstant(scale),
          std::string(kLayerPath) + "mul_0_output_qdq_h" + suffix);

      auto attn_min = mllm::models::qwen3::sha::ptq::QDQ(
          this, attn.min(-1, true),
          std::string(kLayerPath) + "reduce_min_output_qdq_h" + suffix);
      auto minus_value = mllm::Tensor::constant(-20, mllm::kFloat32);
      minus_value = mllm::models::qwen3::sha::ptq::QDQ(
          this, minus_value,
          std::string(kLayerPath) + "neg_20_qdq_h" + suffix);
      auto masked_value = mllm::models::qwen3::sha::ptq::QDQ(
          this, attn_min.addConstant(minus_value),
          std::string(kLayerPath) + "minus_0_output_qdq_h" + suffix);
      // Visibility is already boolean at runtime. Feeding it directly avoids
      // the existing QDQ_CONSTANT(float 0) payload/encoding mismatch at the
      // asymmetric causal-mask boundary.
      attn = mllm::nn::functional::where(causal_mask, attn, masked_value);
      attn = mllm::models::qwen3::sha::ptq::QDQ(
          this, attn,
          std::string(kLayerPath) + "where_attn_qdq_h" + suffix);
      attn = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::softmax(attn, -1),
          std::string(kLayerPath) + "softmax_output_qdq_h" + suffix);

      auto output = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(attn, value),
          std::string(kLayerPath) +
              "attn_value_matmul_output_qdq_h" + suffix);
      outputs.push_back(output);
    }
    return outputs;
  }
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path =
      Argparse::add<std::string>("-m|--model_path")
          .help("Accepted RMSNorm-A8 source model.");
  auto& qnn_aot_cfg =
      Argparse::add<std::string>("-aot_cfg|--aot_config")
          .help("QNN AOT config.");
  auto& qnn_env =
      Argparse::add<std::string>("-qnn_env|--qnn_env_path")
          .help("QAIRT x86 library path.");
  auto& output_context =
      Argparse::add<std::string>("-o|--output_context_name")
          .help("Output context.");
  auto& width_arg =
      Argparse::add<int>("--width").help("Attention width: 32 or 1024.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet() ||
      !output_context.isSet() || !width_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(width_arg.get() == kSeq || width_arg.get() == kContext);

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  qwen3_qnn_aot::addCausalMaskParams(params);
  copyPerHeadQparams(params);
  auto model = CompactPrefillAttention("model");
  model.load(params);

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kQueryHeads + 2 * kKvHeads + 1);
  for (int32_t head = 0; head < kQueryHeads; ++head) {
    auto query = mllm::Tensor::zeros({1, 1, kSeq, kHeadDim}, mllm::kUInt8)
                     .setName("query_" + std::to_string(head));
    attachQparams(query, params, sourcePrefix("q_rope_add_0_output_qdq"),
                  mllm::kUInt8PerTensorAsy);
    inputs.push_back(query);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto key =
        mllm::Tensor::zeros({1, 1, kHeadDim, width_arg.get()}, mllm::kUInt8)
            .setName("key_" + std::to_string(head));
    attachQparams(key, params, sourcePrefix("k_cast_to_int8_qdq"),
                  mllm::kUInt8PerTensorSym);
    inputs.push_back(key);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto value =
        mllm::Tensor::zeros({1, 1, width_arg.get(), kHeadDim}, mllm::kUInt8)
            .setName("value_" + std::to_string(head));
    attachQparams(value, params, sourcePrefix("v_cast_to_int8_qdq"),
                  mllm::kUInt8PerTensorSym);
    inputs.push_back(value);
  }
  auto causal_mask =
      mllm::Tensor::zeros({1, 1, kSeq, width_arg.get()}, mllm::kBool)
          .setName("causal_mask");
  inputs.push_back(causal_mask);

  mllm::ir::lowlevel::traceStart();
  auto outputs = model(inputs);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(),
      mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(
      &qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "compact_prefill_attention_w" +
                    std::to_string(width_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("Compact-prefill attention compilation completed: " +
              output_context.get());
  return 0;
});
