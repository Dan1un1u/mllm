// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile one real Qwen3 LPBQ projection as a standalone QNN graph.  A16 and
// A8 use the same Conv2D expression and byte-identical W4G32 static tensors;
// only the asymmetric activation dtype/qparams and matching recipe differ.

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/core/aops/Conv2DOp.hpp>
#include <mllm/nn/Module.hpp>
#include <mllm/nn/layers/Conv2D.hpp>

using mllm::Argparse;

namespace {

struct Shape {
  int32_t input;
  int32_t output;
};

Shape projectionShape(const std::string& projection) {
  if (projection == "gate_proj" || projection == "up_proj") return {2048, 6144};
  if (projection == "down_proj") return {6144, 2048};
  if (projection == "lm_head") return {2048, 151936};
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                  "--projection must be gate_proj, up_proj, down_proj, or lm_head; got '{}'",
                  projection);
}

mllm::DataTypes activationType(const std::string& activation) {
  if (activation == "a16") return mllm::kUInt16PerTensorAsy;
  if (activation == "a8") return mllm::kUInt8PerTensorAsy;
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                  "--activation must be a16 or a8; got '{}'", activation);
}

mllm::DataTypes storageType(const std::string& activation) {
  return activation == "a16" ? mllm::kUInt16 : mllm::kUInt8;
}

mllm::aops::Conv2DOpImplType implementation(const std::string& activation) {
  return activation == "a16"
             ? mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G32
             : mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32;
}

std::string opName(const std::string& projection) {
  return projection == "lm_head" ? "lm_head" : "layers.14.mlp." + projection;
}

std::string qparamPrefix(const std::string& activation, const std::string& projection,
                         const std::string& side) {
  return "diagnostic." + activation + "." + projection + "." + side;
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                   const std::string& prefix, mllm::DataTypes dtype) {
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class Projection final : public mllm::nn::Module {
 public:
  Projection(const std::string& name, const std::string& projection, const std::string& activation,
             Shape shape)
      : mllm::nn::Module(name), projection_(projection), activation_(activation), shape_(shape) {
    op_ = reg<mllm::nn::Conv2D>(
        opName(projection_), shape_.input, shape_.output, std::vector<int32_t>{1, 1},
        std::vector<int32_t>{1, 1}, std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1},
        false, implementation(activation_));
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = inputs.front();
    x = x.view({1, 1, -1, shape_.input}, true);
    auto output = op_(x).view({1, -1, shape_.output}, true);
    attachQparams(output, getTopParameterFile(),
                  qparamPrefix(activation_, projection_, "output"),
                  activationType(activation_));
    return {output};
  }

 private:
  std::string projection_;
  std::string activation_;
  Shape shape_;
  mllm::nn::Conv2D op_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact diagnostic model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& activation_arg = Argparse::add<std::string>("--activation").help("a16 or a8.");
  auto& projection_arg = Argparse::add<std::string>("--projection").help("Projection name.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !activation_arg.isSet() || !projection_arg.isSet()
      || !seq_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(seq_arg.get() == 1 || seq_arg.get() == 32);
  (void)activationType(activation_arg.get());
  const auto shape = projectionShape(projection_arg.get());

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = Projection("model", projection_arg.get(), activation_arg.get(), shape);
  model.load(params);
  auto input = mllm::Tensor::zeros({1, seq_arg.get(), shape.input},
                                   storageType(activation_arg.get())).setName("input");
  attachQparams(input, params,
                qparamPrefix(activation_arg.get(), projection_arg.get(), "input"),
                activationType(activation_arg.get()));

  mllm::ir::lowlevel::traceStart();
  auto output = model(input).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "lpbq_" + activation_arg.get() + "_" + projection_arg.get()
                    + "_s" + std::to_string(seq_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("LPBQ activation-width projection compilation completed: " + output_context.get());
  return 0;
});
