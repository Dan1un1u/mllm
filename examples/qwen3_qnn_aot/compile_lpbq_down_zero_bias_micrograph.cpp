// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile the real Qwen3 layer-14 down projection at S=32 as a standalone
// QNN graph. The only candidate variable is whether Conv2d receives an
// explicit, static, all-zero U8 bias or omits the logical bias input.

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>

#include "mllm/core/aops/Conv2DOp.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/layers/Conv2D.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kSeq = 32;
constexpr int32_t kInChannels = 6144;
constexpr int32_t kOutChannels = 2048;

void attachA8Qparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                     const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class DownProjectionMicrograph final : public mllm::nn::Module {
 public:
  explicit DownProjectionMicrograph(bool explicit_zero_bias)
      : mllm::nn::Module("model") {
    mllm::aops::Conv2DOpOptions options;
    options.in_channels = kInChannels;
    options.out_channels = kOutChannels;
    options.kernel_size = {1, 1};
    options.stride = {1, 1};
    options.padding = {0, 0};
    options.dilation = {1, 1};
    options.bias = false;
    options.qnn_explicit_zero_bias = explicit_zero_bias;
    options.impl_type = mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32;
    down_proj_ = reg<mllm::nn::Conv2D>("layers.14.mlp.down_proj", options);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto input = inputs.front();
    auto output = down_proj_(input.view({1, 1, kSeq, kInChannels}, true))
                      .view({1, kSeq, kOutChannels}, true);
    attachA8Qparams(output, getTopParameterFile(),
                    "model.layers.14.add_1_lhs_input_qdq.fake_quant");
    return {output};
  }

 private:
  mllm::nn::Conv2D down_proj_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Accepted RMSNorm-U8 model file.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config file.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86_64 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context binary.");
  auto& bias_mode = Argparse::add<std::string>("--bias_mode").help("omitted or explicit_u8_zero.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !bias_mode.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if (bias_mode.get() != "omitted" && bias_mode.get() != "explicit_u8_zero") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--bias_mode must be omitted or explicit_u8_zero; got '{}'", bias_mode.get());
  }

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = DownProjectionMicrograph(bias_mode.get() == "explicit_u8_zero");
  model.load(params);

  auto input = mllm::Tensor::zeros({1, kSeq, kInChannels}, mllm::kUInt8).setName("input");
  attachA8Qparams(input, params,
                  "model.layers.14.mlp.down_proj_input_qdq.fake_quant");

  mllm::ir::lowlevel::traceStart();
  auto output = model(input).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto mir_name = "lpbq_down_s32_" + bias_mode.get() + ".mir";
  mllm::redirect(mir_name, [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("LPBQ down-projection zero-bias micrograph compilation completed: "
              + output_context.get());
  return 0;
});
