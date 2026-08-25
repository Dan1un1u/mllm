// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0016 Stage B compiler for the fixed layer-14 s32 first-GQA-group
// QK -> Softmax -> AV micrograph.

#include <memory>

#include <mllm/mllm.hpp>
#include <mllm/backends/base/PluginInterface.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/ir/linalg/Op.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/nn/Module.hpp>

using mllm::Argparse;

namespace {

void attachAsymmetric(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix,
                      float scale, int32_t zero_point) {
  params->push(prefix + ".scale", mllm::Tensor::constant(scale, mllm::kFloat32));
  params->push(prefix + ".zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32));
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class FusedGqaOp final : public mllm::plugin::interface::CustomizedOp {
 public:
  FusedGqaOp() : CustomizedOp("FusedGqaHmxSoftmaxAv") {
    setName("model.layers.14.self_attn.exp0016_gqa_group0");
    setDeviceType(mllm::kQNN);
  }

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs,
             std::vector<mllm::Tensor>& outputs) override {
    auto* ir_context = static_cast<mllm::ir::IRContext*>(trace_context);
    const auto input_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_context, inputs);
    const auto output_irs = mllm::ir::tensor::wrapTensors2TensorIR(ir_context, outputs);
    ir_context->create<mllm::ir::linalg::CustomizedOp>(shared_from_this(), input_irs, output_irs);
  }
};

class FusedGqaModel final : public mllm::nn::Module {
 public:
  explicit FusedGqaModel(const std::string& name) : mllm::nn::Module(name), op_(std::make_shared<FusedGqaOp>()) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    if (inputs.size() != 4) throw std::runtime_error("FusedGqaModel expects Q, K, V, and mask");
    // Compile-time tensors remain CPU-owned, matching the existing AOT
    // compilers. The CustomizedOp itself carries the QNN device assignment.
    std::vector<mllm::Tensor> outputs{mllm::Tensor::empty({1, 2, 32, 128}, mllm::kUInt8).setName("output")};
    attachAsymmetric(outputs[0], getTopParameterFile(), "gqa.output", 0.5333649516105652F, 229);
    auto trace_context = mllm::Context::instance().thisThread()->ir_context;
    op_->trace(trace_context.get(), inputs, outputs);
    return outputs;
  }

 private:
  std::shared_ptr<FusedGqaOp> op_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet()) {
    Argparse::printHelp();
    return 2;
  }

  auto params = mllm::ParameterFile::create();
  auto model = FusedGqaModel("model");
  model.load(params);
  auto query = mllm::Tensor::zeros({1, 2, 32, 128}, mllm::kUInt8).setName("query");
  auto key = mllm::Tensor::zeros({1, 1, 128, 1024}, mllm::kUInt8).setName("key_transposed");
  auto value = mllm::Tensor::zeros({1, 1, 1024, 128}, mllm::kUInt8).setName("value");
  auto mask = mllm::Tensor::zeros({1, 1, 32, 1024}, mllm::kUInt8).setName("causal_mask");
  attachAsymmetric(query, params, "gqa.query", 0.32302287220954895F, 122);
  attachAsymmetric(key, params, "gqa.key", 0.33071592450141907F, 128);
  attachAsymmetric(value, params, "gqa.value", 1.4026927947998047F, 128);
  attachAsymmetric(mask, params, "gqa.mask", 3.9215688047988815e-6F, 255);

  mllm::ir::lowlevel::traceStart();
  const auto output = model(query, key, value, mask).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;
  mllm::redirect("qhpi_fused_gqa_pre.mir", [&]() { mllm::print(ir); });

  auto qnn_aot_env =
      mllm::qnn::aot::QnnAOTEnv(qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();
  mllm::redirect("qhpi_fused_gqa.mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("EXP-0016 fused GQA compilation completed: " + output_context.get());
  return 0;
});
