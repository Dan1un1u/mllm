// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile one real Qwen3 layer-14 MLP projection as a standalone QNN graph.
// The graph keeps the accepted W4G32 codes, LPBQ scales, and A8 qparams; only
// the QNN projection expression and its physical static-weight layout vary.

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>

#include "mllm/core/aops/Conv2DOp.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/layers/Conv2D.hpp"
#include "mllm/nn/layers/Linear.hpp"
#include "mllm/nn/layers/Param.hpp"

using mllm::Argparse;

namespace {

enum class ProjectionLayout { kConv, kFullyConnected, kMatMul };

ProjectionLayout parseLayout(const std::string& value) {
  if (value == "conv") return ProjectionLayout::kConv;
  if (value == "fc") return ProjectionLayout::kFullyConnected;
  if (value == "matmul") return ProjectionLayout::kMatMul;
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--layout must be conv, fc, or matmul; got '{}'", value);
}

std::pair<int32_t, int32_t> projectionShape(const std::string& value) {
  if (value == "gate_proj" || value == "up_proj") return {2048, 6144};
  if (value == "down_proj") return {6144, 2048};
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                  "--projection must be gate_proj, up_proj, or down_proj; got '{}'", value);
}

std::string inputQparamPrefix(const std::string& projection) {
  if (projection == "gate_proj" || projection == "up_proj") {
    return "model.layers.14.mlp.up_proj_input_qdq.fake_quant";
  }
  return "model.layers.14.mlp.down_proj_input_qdq.fake_quant";
}

std::string outputQparamPrefix(const std::string& projection) {
  if (projection == "gate_proj" || projection == "up_proj") {
    return "model.layers.14.mlp." + projection + "_output_qdq.fake_quant";
  }
  return "model.layers.14.add_1_lhs_input_qdq.fake_quant";
}

void attachA8Qparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                     const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class ProjectionMicrograph final : public mllm::nn::Module {
 public:
  ProjectionMicrograph(const std::string& name, ProjectionLayout layout, const std::string& projection,
                       int32_t in_channels, int32_t out_channels)
      : mllm::nn::Module(name),
        layout_(layout),
        projection_(projection),
        in_channels_(in_channels),
        out_channels_(out_channels) {
    const auto op_name = "layers.14.mlp." + projection;
    switch (layout_) {
      case ProjectionLayout::kConv:
        conv_ = reg<mllm::nn::Conv2D>(op_name, in_channels_, out_channels_, std::vector<int32_t>{1, 1},
                                     std::vector<int32_t>{1, 1}, std::vector<int32_t>{0, 0},
                                     std::vector<int32_t>{1, 1}, false,
                                     mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32);
        break;
      case ProjectionLayout::kFullyConnected:
        fc_ = reg<mllm::nn::Linear>(op_name, in_channels_, out_channels_, false,
                                    mllm::aops::LinearImplTypes::kQNN_LPBQ_w4a8o8_G32);
        break;
      case ProjectionLayout::kMatMul:
        weight_ = reg<mllm::nn::Param>(op_name + ".weight", name + "." + op_name + ".weight");
        break;
    }
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = inputs.front();
    mllm::Tensor output;
    switch (layout_) {
      case ProjectionLayout::kConv:
        output = conv_(x.view({1, 1, -1, in_channels_}, true)).view({1, -1, out_channels_}, true);
        break;
      case ProjectionLayout::kFullyConnected:
        output = fc_(x).view({1, -1, out_channels_}, true);
        break;
      case ProjectionLayout::kMatMul:
        output = mllm::nn::functional::matmul(x, weight_.weight(), false, false,
                                              mllm::aops::MatMulOpType::kQNN_LPBQ_w4a8o8_G32);
        break;
    }
    attachA8Qparams(output, getTopParameterFile(), outputQparamPrefix(projection_));
    return {output};
  }

 private:
  ProjectionLayout layout_;
  std::string projection_;
  int32_t in_channels_;
  int32_t out_channels_;
  mllm::nn::Conv2D conv_;
  mllm::nn::Linear fc_;
  mllm::nn::Param weight_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Relayout-specific model file.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config file.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86_64 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context binary.");
  auto& layout_arg = Argparse::add<std::string>("--layout").help("conv, fc, or matmul.");
  auto& projection_arg =
      Argparse::add<std::string>("--projection").help("gate_proj, up_proj, or down_proj.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet()
      || !layout_arg.isSet() || !projection_arg.isSet() || !seq_arg.isSet()) {
    Argparse::printHelp();
    return -1;
  }
  MLLM_RT_ASSERT(seq_arg.get() == 1 || seq_arg.get() == 32);

  const auto layout = parseLayout(layout_arg.get());
  const auto [in_channels, out_channels] = projectionShape(projection_arg.get());
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = ProjectionMicrograph("model", layout, projection_arg.get(), in_channels, out_channels);
  model.load(params);

  auto input = mllm::Tensor::zeros({1, seq_arg.get(), in_channels}, mllm::kUInt8).setName("input");
  attachA8Qparams(input, params, inputQparamPrefix(projection_arg.get()));

  mllm::ir::lowlevel::traceStart();
  auto output = model(input).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto mir_name = "lpbq_mlp_" + layout_arg.get() + "_" + projection_arg.get() + "_s"
                        + std::to_string(seq_arg.get()) + ".mir";
  mllm::redirect(mir_name, [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("LPBQ MLP micrograph compilation completed: " + output_context.get());
  return 0;
});
