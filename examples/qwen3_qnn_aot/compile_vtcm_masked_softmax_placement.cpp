// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile either an interior QK -> custom Softmax -> PV placement graph or a
// graph-boundary numerical fixture, using accepted layer-14 A8 encodings.

#include <cstdlib>
#include <string_view>

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

constexpr int32_t kHeads = 16;
constexpr int32_t kKvHeads = 4;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kContext = 1024;
constexpr int32_t kKvGroups = kHeads / kKvHeads;
constexpr std::string_view kLayerPath = "model.layers.14.self_attn.";

std::string qparamPrefix(std::string_view base, int32_t head) {
  return std::string(kLayerPath) + std::string(base) + "_h" + std::to_string(head) + ".fake_quant";
}

void duplicateQparams(const mllm::ParameterFile::ptr_t& params) {
  const auto copy = [&](std::string_view base, int32_t count) {
    for (const auto suffix : {std::string_view{"scale"}, std::string_view{"zero_point"}}) {
      const auto source = std::string(kLayerPath) + std::string(base) + ".fake_quant." + std::string(suffix);
      auto value = params->pull(source);
      for (int32_t head = 0; head < count; ++head) {
        const auto destination = qparamPrefix(base, head) + "." + std::string(suffix);
        params->push(destination, value.contiguous().setMemType(mllm::kParamsNormal).setName(destination));
      }
    }
  };

  copy("q_rope_add_0_output_qdq", kHeads);
  copy("k_cast_to_int8_qdq", kKvHeads);
  copy("v_cast_to_int8_qdq", kKvHeads);
  copy("qk_matmul_output_qdq", kHeads);
  copy("scaling_qdq", kHeads);
  copy("mul_0_output_qdq", kHeads);
  copy("softmax_output_qdq", kHeads);
  copy("attn_value_matmul_output_qdq", kHeads);
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix,
                   mllm::DataTypes dtype) {
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class PlacementGraph final : public mllm::nn::Module {
 public:
  explicit PlacementGraph(const std::string& name) : mllm::nn::Module(name) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(kHeads + 2 * kKvHeads + 1));
    const auto& causal_mask = inputs.back();
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kHeads);
    for (int32_t head = 0; head < kHeads; ++head) {
      const int32_t kv_head = head / kKvGroups;
      auto scores = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(inputs[head], inputs[kHeads + kv_head]),
          "layers.14.self_attn.qk_matmul_output_qdq_h" + std::to_string(head));
      auto scale = mllm::Tensor::constant(1.0f / 11.313708498984761f, mllm::kFloat32);
      scale = mllm::models::qwen3::sha::ptq::QDQ(
          this, scale, "layers.14.self_attn.scaling_qdq_h" + std::to_string(head));
      scores = mllm::models::qwen3::sha::ptq::QDQ(
          this, scores.mulConstant(scale), "layers.14.self_attn.mul_0_output_qdq_h" + std::to_string(head));
      auto probabilities = mllm::models::qwen3::sha::ptq::QDQ(
          this, scores + causal_mask, "layers.14.self_attn.softmax_output_qdq_h" + std::to_string(head));
      outputs.push_back(mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(probabilities, inputs[kHeads + kKvHeads + kv_head]),
          "layers.14.self_attn.attn_value_matmul_output_qdq_h" + std::to_string(head)));
    }
    return outputs;
  }
};

class NumericalGraph final : public mllm::nn::Module {
 public:
  explicit NumericalGraph(const std::string& name) : mllm::nn::Module(name) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), 2);
    auto scores = inputs[0];
    return {mllm::models::qwen3::sha::ptq::QDQ(
        this, scores + inputs[1], "layers.14.self_attn.softmax_output_qdq_h0")};
  }
};

void compileAndSave(const mllm::ir::IRContext::ptr_t& ir, const mllm::ParameterFile::ptr_t& params,
                    const std::string& qnn_env_path, const std::string& qnn_aot_cfg, const std::string& output_context) {
  auto qnn_aot_env =
      mllm::qnn::aot::QnnAOTEnv(qnn_env_path, mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg, params));
  pm.run();
  qnn_aot_env.saveContext("context.0", output_context);
  qnn_aot_env.destroyContext("context.0");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Accepted RMSNorm-A8 model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");
  auto& numerical_fixture =
      Argparse::add<bool>("--numerical_fixture").help("Expose one custom output.").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet()
      || !seq_arg.isSet() || (seq_arg.get() != 1 && seq_arg.get() != 32)) {
    Argparse::printHelp();
    return 2;
  }

  setenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX", "1", 1);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  qwen3_qnn_aot::addCausalMaskParams(params);
  duplicateQparams(params);

  if (numerical_fixture.get()) {
    NumericalGraph model("model");
    model.load(params);
    auto scores = mllm::Tensor::zeros({1, 1, seq_arg.get(), kContext}, mllm::kUInt8).setName("scores");
    attachQparams(scores, params, qparamPrefix("mul_0_output_qdq", 0), mllm::kUInt8PerTensorAsy);
    auto mask = mllm::Tensor::zeros({1, 1, seq_arg.get(), kContext}, qwen3_qnn_aot::kCausalMaskStorageType)
                    .setName("causal_mask");
    attachQparams(mask, params, "causal_mask", qwen3_qnn_aot::kCausalMaskQuantType);
    mllm::ir::lowlevel::traceStart();
    auto outputs = model(std::vector<mllm::Tensor>{scores, mask});
    auto ir = mllm::ir::lowlevel::traceStop();
    (void)outputs;
    compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
    mllm::print("Compiled numerical custom Softmax: " + output_context.get());
    return 0;
  }

  PlacementGraph model("model");
  model.load(params);
  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 2 * kKvHeads + 1);
  for (int32_t head = 0; head < kHeads; ++head) {
    auto q = mllm::Tensor::zeros({1, 1, seq_arg.get(), kHeadDim}, mllm::kUInt8).setName("q_h" + std::to_string(head));
    attachQparams(q, params, qparamPrefix("q_rope_add_0_output_qdq", head), mllm::kUInt8PerTensorAsy);
    inputs.push_back(q);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto k = mllm::Tensor::zeros({1, 1, kHeadDim, kContext}, mllm::kUInt8).setName("k_h" + std::to_string(head));
    attachQparams(k, params, qparamPrefix("k_cast_to_int8_qdq", head), mllm::kUInt8PerTensorSym);
    inputs.push_back(k);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto v = mllm::Tensor::zeros({1, 1, kContext, kHeadDim}, mllm::kUInt8).setName("v_h" + std::to_string(head));
    attachQparams(v, params, qparamPrefix("v_cast_to_int8_qdq", head), mllm::kUInt8PerTensorSym);
    inputs.push_back(v);
  }
  auto mask = mllm::Tensor::zeros({1, 1, seq_arg.get(), kContext}, qwen3_qnn_aot::kCausalMaskStorageType)
                  .setName("causal_mask");
  attachQparams(mask, params, "causal_mask", qwen3_qnn_aot::kCausalMaskQuantType);
  inputs.push_back(mask);
  mllm::ir::lowlevel::traceStart();
  auto outputs = model(inputs);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;
  compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
  mllm::print("Compiled interior custom Softmax placement graph: " + output_context.get());
  return 0;
});
