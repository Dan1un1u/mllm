// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0014: compile a single batched U8xS8 MatMul whose physical tile is
// exactly one V79 HMX activation crouton and one 32x32 signed weight tile.

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>

using mllm::Argparse;

namespace {

void attachAsymmetric(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix,
                      int32_t zero_point) {
  params->push(prefix + ".scale", mllm::Tensor::constant(1.0F, mllm::kFloat32));
  params->push(prefix + ".zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32));
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

void attachSignedSymmetric(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix) {
  params->push(prefix + ".scale", mllm::Tensor::constant(1.0F, mllm::kFloat32));
  tensor = tensor.__unsafeSetDType(mllm::kInt8PerTensorSym);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
}

class HmxProbe final : public mllm::nn::Module {
 public:
  explicit HmxProbe(const std::string& name) : mllm::nn::Module(name) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto output = mllm::nn::functional::matmul(inputs.at(0), inputs.at(1));
    output.setName("output");
    attachAsymmetric(output, getTopParameterFile(), "probe.output", 0);
    return {output};
  }
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& mode = Argparse::add<std::string>("--mode").help("single, mixed-resource, or sequential-hmx.").def("single");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet()
      || (mode.get() != "single" && mode.get() != "mixed-resource" && mode.get() != "sequential-hmx")) {
    Argparse::printHelp();
    return 2;
  }

  auto params = mllm::ParameterFile::create();
  auto model = HmxProbe("model");
  model.load(params);
  auto activation = mllm::Tensor::zeros({1, 8, 8, 32}, mllm::kUInt8).setName("activation");
  auto weight = mllm::Tensor::zeros({1, 1, 32, 32}, mllm::kInt8).setName("weight_packed");
  attachAsymmetric(activation, params, "probe.activation", 128);
  attachSignedSymmetric(weight, params, "probe.weight");

  mllm::ir::lowlevel::traceStart();
  auto output = model(activation, weight).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  if (mode.get() == "sequential-hmx") {
    setenv("MLLM_QNN_QHPI_SEQUENTIAL_HMX_PROBE", "1", 1);
  } else if (mode.get() == "mixed-resource") {
    setenv("MLLM_QNN_QHPI_MIXED_RESOURCE_PROBE", "1", 1);
  } else {
    setenv("MLLM_QNN_QHPI_U8S8_HMX_PROBE", "1", 1);
  }
  auto qnn_aot_env =
      mllm::qnn::aot::QnnAOTEnv(qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const char* mir_name = mode.get() == "sequential-hmx" ? "qhpi_sequential_hmx_probe.mir"
                         : mode.get() == "mixed-resource" ? "qhpi_mixed_resource_probe.mir"
                                                          : "qhpi_u8_hmx_probe.mir";
  mllm::redirect(mir_name, [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("QHPI " + mode.get() + " probe compilation completed: " + output_context.get());
  return 0;
});
