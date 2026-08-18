// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compare the accepted layer-14 S=1 LPBQ gate/up Conv2D with an exact
// output-channel split. The candidate keeps W4G32 and the original A8 qparams,
// but exposes two 2048x3072 projections so each lowered weight working set is
// half the reference 2048x6144 projection.

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>

#include "mllm/core/aops/Conv2DOp.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/layers/Conv2D.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kSeq = 1;
constexpr int32_t kInChannels = 2048;
constexpr int32_t kOutChannels = 6144;
constexpr int32_t kSplitOutChannels = 3072;

void attachA8Qparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                     const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

mllm::aops::Conv2DOpOptions projectionOptions(int32_t out_channels) {
  mllm::aops::Conv2DOpOptions options;
  options.in_channels = kInChannels;
  options.out_channels = out_channels;
  options.kernel_size = {1, 1};
  options.stride = {1, 1};
  options.padding = {0, 0};
  options.dilation = {1, 1};
  options.bias = false;
  options.impl_type = mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32;
  return options;
}

class GateUpProjectionMicrograph final : public mllm::nn::Module {
 public:
  GateUpProjectionMicrograph(const std::string& projection, bool split)
      : mllm::nn::Module("model"), projection_(projection), split_(split) {
    const auto base_name = "layers.14.mlp." + projection;
    if (split_) {
      projection_0_ = reg<mllm::nn::Conv2D>(base_name + ".oc0", projectionOptions(kSplitOutChannels));
      projection_1_ = reg<mllm::nn::Conv2D>(base_name + ".oc1", projectionOptions(kSplitOutChannels));
    } else {
      projection_full_ = reg<mllm::nn::Conv2D>(base_name, projectionOptions(kOutChannels));
    }
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto input = inputs.front();
    input = input.view({1, 1, kSeq, kInChannels}, true);
    const auto output_qparam = "model.layers.14.mlp." + projection_ + "_output_qdq.fake_quant";
    if (!split_) {
      auto output = projection_full_(input).view({1, kSeq, kOutChannels}, true);
      attachA8Qparams(output, getTopParameterFile(), output_qparam);
      return {output};
    }

    auto output_0 = projection_0_(input).view({1, kSeq, kSplitOutChannels}, true);
    auto output_1 = projection_1_(input).view({1, kSeq, kSplitOutChannels}, true);
    attachA8Qparams(output_0, getTopParameterFile(), output_qparam);
    attachA8Qparams(output_1, getTopParameterFile(), output_qparam);
    auto output = mllm::nn::functional::concat({output_0, output_1}, -1);
    attachA8Qparams(output, getTopParameterFile(), output_qparam);
    return {output};
  }

 private:
  std::string projection_;
  bool split_;
  mllm::nn::Conv2D projection_full_;
  mllm::nn::Conv2D projection_0_;
  mllm::nn::Conv2D projection_1_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact gate/up micrograph model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config file.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86_64 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context binary.");
  auto& layout = Argparse::add<std::string>("--layout").help("full or split2.");
  auto& projection = Argparse::add<std::string>("--projection").help("gate_proj or up_proj.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !layout.isSet() || !projection.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if (layout.get() != "full" && layout.get() != "split2") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--layout must be full or split2; got '{}'", layout.get());
  }
  if (projection.get() != "gate_proj" && projection.get() != "up_proj") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--projection must be gate_proj or up_proj; got '{}'", projection.get());
  }

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = GateUpProjectionMicrograph(projection.get(), layout.get() == "split2");
  model.load(params);

  auto input = mllm::Tensor::zeros({1, kSeq, kInChannels}, mllm::kUInt8).setName("input");
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

  const auto mir_name = "lpbq_" + projection.get() + "_s1_" + layout.get() + ".mir";
  mllm::redirect(mir_name, [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("LPBQ gate/up output-channel split micrograph compilation completed: "
              + output_context.get());
  return 0;
});
