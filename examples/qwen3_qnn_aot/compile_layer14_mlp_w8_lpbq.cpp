// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile the complete middle-layer Qwen3 MLP as one standalone QNN graph.
// Both variants use the accepted W4A8 activation qparams and identical
// NHWC/HWIO Conv2D expression.  Only static-weight encoding differs:
// W4G32 LPBQ versus per-tensor symmetric W8.

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/core/aops/Conv2DOp.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>
#include <mllm/nn/layers/Conv2D.hpp>

using mllm::Argparse;

namespace {

constexpr int32_t kHidden = 2048;
constexpr int32_t kIntermediate = 6144;

mllm::aops::Conv2DOpImplType parseVariant(const std::string& value) {
  if (value == "lpbq") return mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32;
  if (value == "w8a8") return mllm::aops::Conv2DOpImplType::kQNN_W8A8;
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--variant must be lpbq or w8a8; got '{}'", value);
}

void attachA8Qparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                     const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class Layer14MLP final : public mllm::nn::Module {
 public:
  Layer14MLP(const std::string& name, mllm::aops::Conv2DOpImplType impl) : mllm::nn::Module(name) {
    const auto property = std::tuple{
        std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
        std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}};
    gate_ = reg<mllm::nn::Conv2D>(
        "layers.14.mlp.gate_proj", kHidden, kIntermediate,
        std::get<0>(property), std::get<1>(property), std::get<2>(property), std::get<3>(property), false, impl);
    up_ = reg<mllm::nn::Conv2D>(
        "layers.14.mlp.up_proj", kHidden, kIntermediate,
        std::get<0>(property), std::get<1>(property), std::get<2>(property), std::get<3>(property), false, impl);
    down_ = reg<mllm::nn::Conv2D>(
        "layers.14.mlp.down_proj", kIntermediate, kHidden,
        std::get<0>(property), std::get<1>(property), std::get<2>(property), std::get<3>(property), false, impl);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = inputs.front();
    x = x.view({1, 1, -1, kHidden}, true);

    auto up = up_(x);
    attachA8Qparams(up, getTopParameterFile(),
                    "model.layers.14.mlp.up_proj_output_qdq.fake_quant");
    up = up.view({1, -1, kIntermediate}, true);

    auto gate = gate_(x);
    attachA8Qparams(gate, getTopParameterFile(),
                    "model.layers.14.mlp.gate_proj_output_qdq.fake_quant");
    gate = gate.view({1, -1, kIntermediate}, true);

    auto sigmoid = mllm::nn::functional::sigmoid(gate);
    attachA8Qparams(sigmoid, getTopParameterFile(),
                    "model.layers.14.mlp.sigmoid_output_qdq.fake_quant");
    auto activated = gate * sigmoid;
    attachA8Qparams(activated, getTopParameterFile(),
                    "model.layers.14.mlp.act_output_qdq.fake_quant");

    auto down_input = activated * up;
    attachA8Qparams(down_input, getTopParameterFile(),
                    "model.layers.14.mlp.down_proj_input_qdq.fake_quant");
    down_input = down_input.view({1, 1, -1, kIntermediate}, true);

    auto output = down_(down_input).view({1, -1, kHidden}, true);
    attachA8Qparams(output, getTopParameterFile(),
                    "model.layers.14.add_1_lhs_input_qdq.fake_quant");
    return {output};
  }

 private:
  mllm::nn::Conv2D gate_;
  mllm::nn::Conv2D up_;
  mllm::nn::Conv2D down_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact layer-14 model file.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config file.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86_64 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context binary.");
  auto& variant = Argparse::add<std::string>("--variant").help("lpbq or w8a8.");
  auto& seq = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !variant.isSet() || !seq.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(seq.get() == 1 || seq.get() == 32);

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = Layer14MLP("model", parseVariant(variant.get()));
  model.load(params);

  auto input = mllm::Tensor::zeros({1, seq.get(), kHidden}, mllm::kUInt8).setName("input");
  attachA8Qparams(input, params,
                  "model.layers.14.mlp.up_proj_input_qdq.fake_quant");

  mllm::ir::lowlevel::traceStart();
  auto output = model(input).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  mllm::redirect("layer14_mlp_" + variant.get() + "_s" + std::to_string(seq.get()) + ".mir",
                 [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("Layer-14 MLP compilation completed: " + output_context.get());
  return 0;
});
