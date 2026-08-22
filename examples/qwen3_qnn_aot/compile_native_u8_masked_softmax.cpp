// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile the exact layer-14 native-U8 masked-softmax pattern used by the
// accepted Qwen3 SHA graph. The experiment changes only the QAIRT release and
// graph-finalize P point.

#include <array>
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
constexpr int32_t kContext = 1024;
constexpr std::string_view kLayerPath = "layers.14.self_attn.";

std::string qparamPrefix(int32_t head) {
  return "model.layers.14.self_attn.mul_0_output_qdq_h" + std::to_string(head) + ".fake_quant";
}

void addPerHeadQparams(const mllm::ParameterFile::ptr_t& params) {
  constexpr std::array<std::string_view, 6> kQdqNames = {
      "mul_0_output_qdq", "reduce_min_output_qdq", "neg_20_qdq", "minus_0_output_qdq", "where_attn_qdq", "softmax_output_qdq",
  };
  const std::string layer_prefix = "model.layers.14.self_attn.";
  for (const auto name : kQdqNames) {
    for (const auto suffix : {std::string_view{"scale"}, std::string_view{"zero_point"}}) {
      const auto source = layer_prefix + std::string(name) + ".fake_quant." + std::string(suffix);
      auto value = params->pull(source);
      for (int32_t head = 0; head < kHeads; ++head) {
        const auto destination =
            layer_prefix + std::string(name) + "_h" + std::to_string(head) + ".fake_quant." + std::string(suffix);
        params->push(destination, value.contiguous().setMemType(mllm::kParamsNormal).setName(destination));
      }
    }
  }
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class NativeU8MaskedSoftmax final : public mllm::nn::Module {
 public:
  explicit NativeU8MaskedSoftmax(const std::string& name) : mllm::nn::Module(name) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(kHeads + 1));
    auto causal_mask = inputs.back();
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kHeads);
    for (int32_t head = 0; head < kHeads; ++head) {
      const auto suffix = std::to_string(head);
      auto attn = inputs[head];
      auto attn_min = mllm::models::qwen3::sha::ptq::QDQ(this, attn.min(-1, true),
                                                         std::string(kLayerPath) + "reduce_min_output_qdq_h" + suffix);
      auto minus_value = mllm::Tensor::constant(-20, mllm::kFloat32);
      minus_value = mllm::models::qwen3::sha::ptq::QDQ(this, minus_value, std::string(kLayerPath) + "neg_20_qdq_h" + suffix);
      auto masked_value = mllm::models::qwen3::sha::ptq::QDQ(this, attn_min.addConstant(minus_value),
                                                             std::string(kLayerPath) + "minus_0_output_qdq_h" + suffix);
      auto zero = mllm::Tensor::constant(0.f, mllm::kFloat32);
      zero = mllm::models::qwen3::sha::ptq::QDQ_CONSTANT(this, zero, "constant_zero");
      attn = mllm::nn::functional::where(causal_mask.equalConstant(zero), attn, masked_value);
      attn = mllm::models::qwen3::sha::ptq::QDQ(this, attn, std::string(kLayerPath) + "where_attn_qdq_h" + suffix);
      attn = mllm::models::qwen3::sha::ptq::QDQ(this, mllm::nn::functional::softmax(attn, -1),
                                                std::string(kLayerPath) + "softmax_output_qdq_h" + suffix);
      outputs.push_back(attn);
    }
    return outputs;
  }
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Accepted RMSNorm-A8 model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet() || !seq_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(seq_arg.get() == 1 || seq_arg.get() == 32);

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  qwen3_qnn_aot::addCausalMaskParams(params);
  addPerHeadQparams(params);
  auto model = NativeU8MaskedSoftmax("model");
  model.load(params);

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 1);
  for (int32_t head = 0; head < kHeads; ++head) {
    auto attn = mllm::Tensor::zeros({1, 1, seq_arg.get(), kContext}, mllm::kUInt8).setName("attn_" + std::to_string(head));
    attachQparams(attn, params, qparamPrefix(head));
    inputs.push_back(attn);
  }
  auto causal_mask = mllm::Tensor::zeros({1, 1, seq_arg.get(), kContext}, qwen3_qnn_aot::kCausalMaskStorageType);
  causal_mask = causal_mask.__unsafeSetDType(qwen3_qnn_aot::kCausalMaskQuantType).setName("causal_mask");
  causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
  causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
  inputs.push_back(causal_mask);

  mllm::ir::lowlevel::traceStart();
  auto outputs = model(inputs);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;

  auto qnn_aot_env =
      mllm::qnn::aot::QnnAOTEnv(qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "native_u8_masked_softmax_s" + std::to_string(seq_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("Native-U8 masked-softmax compilation completed: " + output_context.get());
  return 0;
});
