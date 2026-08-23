// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile one real layer-14 K/V head projection as a standalone W4G32/A8
// graph.  The graph is intentionally limited to one 2048 -> 128 Conv2D so
// QAIRT finalize scheduling can be varied without changing model math.

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

constexpr int32_t kInputChannels = 2048;
constexpr int32_t kOutputChannels = 128;

void validateProjection(const std::string& projection) {
  if (projection != "k_proj" && projection != "v_proj") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--projection must be k_proj or v_proj; got '{}'", projection);
  }
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                   const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class KVHeadProjection final : public mllm::nn::Module {
 public:
  explicit KVHeadProjection(const std::string& projection)
      : mllm::nn::Module("model"), projection_(projection) {
    projection_op_ = reg<mllm::nn::Conv2D>(
        "layers.14.self_attn." + projection_ + ".0", kInputChannels, kOutputChannels,
        std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
        std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}, false,
        mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto input = inputs.front();
    input = input.view({1, 1, -1, kInputChannels}, true);
    auto output = projection_op_(input).view({1, -1, kOutputChannels}, true);
    attachQparams(output, getTopParameterFile(),
                  "diagnostic." + projection_ + ".output");
    return {output};
  }

 private:
  std::string projection_;
  mllm::nn::Conv2D projection_op_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact K/V model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& projection_arg = Argparse::add<std::string>("--projection").help("k_proj or v_proj.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1, 32, or 64.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !projection_arg.isSet() || !seq_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  validateProjection(projection_arg.get());
  MLLM_RT_ASSERT(seq_arg.get() == 1 || seq_arg.get() == 32 || seq_arg.get() == 64);

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = KVHeadProjection(projection_arg.get());
  model.load(params);
  auto input = mllm::Tensor::zeros({1, seq_arg.get(), kInputChannels}, mllm::kUInt8)
                   .setName("input");
  attachQparams(input, params, "diagnostic." + projection_arg.get() + ".input");

  mllm::ir::lowlevel::traceStart();
  auto output = model(input).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "kv_head_" + projection_arg.get() + "_s" + std::to_string(seq_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("K/V head projection compilation completed: " + output_context.get());
  return 0;
});
