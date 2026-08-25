// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0016 Stage-B native comparator. The graph uses the accepted layer-14
// W4A8 RMSNorm-A8 qparams and the same packed graph I/O as the candidate, but
// retains the baseline's two independent Q-head QK/Softmax/AV paths.

#include <cmath>
#include <memory>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>

using mllm::Argparse;

namespace {

void attachAsymmetric(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                      const std::string& prefix, mllm::DataTypes dtype, float scale, int32_t zero_point) {
  params->push(prefix + ".scale", mllm::Tensor::constant(scale, mllm::kFloat32));
  params->push(prefix + ".zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32));
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

void attachConstantQuantization(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                                const std::string& prefix, float scale, int32_t zero_point) {
  params->push(prefix + ".scale", mllm::Tensor::constant(scale, mllm::kFloat32));
  params->push(prefix + ".zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32));
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class NativeGqaModel final : public mllm::nn::Module {
 public:
  NativeGqaModel(const std::string& name, bool w4a16) : mllm::nn::Module(name), w4a16_(w4a16) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    if (inputs.size() != 4) throw std::runtime_error("NativeGqaModel expects packed Q, K, V, and mask");
    const auto& query = inputs[0];
    const auto& key = inputs[1];
    const auto& value = inputs[2];
    auto mask = inputs[3];
    const auto activation_dtype = w4a16_ ? mllm::kUInt16PerTensorAsy : mllm::kUInt8PerTensorAsy;
    std::vector<mllm::Tensor> head_outputs;
    head_outputs.reserve(2);

    for (int head = 0; head < 2; ++head) {
      const std::string prefix = "native.head" + std::to_string(head);
      auto q = query.slice({mllm::kAll, {head, head + 1}, mllm::kAll, mllm::kAll}, true);
      attachAsymmetric(q, getTopParameterFile(), prefix + ".query", activation_dtype,
                       w4a16_ ? 0.0011854973854497075F : 0.32302287220954895F,
                       w4a16_ ? 32133 : 122);

      auto score = mllm::nn::functional::matmul(q, key);
      attachAsymmetric(score, getTopParameterFile(), prefix + ".qk", activation_dtype,
                       w4a16_ ? 0.007394818589091301F : 2.030411958694458F,
                       w4a16_ ? 26930 : 113);

      auto scale = mllm::Tensor::constant(1.0F / std::sqrt(128.0F), mllm::kFloat32);
      attachConstantQuantization(scale, getTopParameterFile(), prefix + ".scale",
                                 w4a16_ ? 1.3487197065842338e-6F : 0.0003466209745965898F, 0);
      score = score.mulConstant(scale);
      attachAsymmetric(score, getTopParameterFile(), prefix + ".scaled", activation_dtype,
                       w4a16_ ? 0.0006536157452501357F : 0.17946475744247437F,
                       w4a16_ ? 26931 : 113);

      auto row_min = score.min(-1, true);
      attachAsymmetric(row_min, getTopParameterFile(), prefix + ".row_min", activation_dtype,
                       w4a16_ ? 0.0003613728331401944F : 0.1029057577252388F,
                       w4a16_ ? 48709 : 197);
      auto minus_twenty = mllm::Tensor::constant(-20.0F, mllm::kFloat32);
      attachConstantQuantization(minus_twenty, getTopParameterFile(), prefix + ".minus_twenty",
                                 w4a16_ ? 0.0003051804378628731F : 0.0784313753247261F,
                                 w4a16_ ? 65535 : 255);
      auto masked_value = row_min.addConstant(minus_twenty);
      attachAsymmetric(masked_value, getTopParameterFile(), prefix + ".masked_value", activation_dtype,
                       w4a16_ ? 0.0005737727624364197F : 0.15794645249843597F,
                       w4a16_ ? 65535 : 255);

      auto zero = mllm::Tensor::constant(0.0F, mllm::kFloat32);
      attachConstantQuantization(zero, getTopParameterFile(), prefix + ".zero",
                                 w4a16_ ? 1.5259022489999552e-8F : 3.9215688047988815e-6F,
                                 w4a16_ ? 65535 : 255);
      auto selected = mllm::nn::functional::where(mask.equalConstant(zero), score, masked_value);
      attachAsymmetric(selected, getTopParameterFile(), prefix + ".selected", activation_dtype,
                       w4a16_ ? 0.0009587961831130087F : 0.2578960955142975F,
                       w4a16_ ? 39218 : 156);

      auto probability = mllm::nn::functional::softmax(selected, -1);
      attachAsymmetric(probability, getTopParameterFile(), prefix + ".probability", activation_dtype,
                       w4a16_ ? 1.5259021893143654e-5F : 1.0F / 255.0F, 0);
      auto output = mllm::nn::functional::matmul(probability, value);
      attachAsymmetric(output, getTopParameterFile(), prefix + ".output", activation_dtype,
                       w4a16_ ? 0.0021023168228566647F : 0.5333649516105652F,
                       w4a16_ ? 58560 : 229);
      head_outputs.push_back(output);
    }

    auto packed = mllm::nn::functional::concat(head_outputs, 1);
    attachAsymmetric(packed, getTopParameterFile(), "native.output", activation_dtype,
                     w4a16_ ? 0.0021023168228566647F : 0.5333649516105652F,
                     w4a16_ ? 58560 : 229);
    return {packed};
  }

 private:
  bool w4a16_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& activation_variant =
      Argparse::add<std::string>("--activation_variant").help("w4a8 or w4a16 comparator.").def("w4a8");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet()
      || (activation_variant.get() != "w4a8" && activation_variant.get() != "w4a16")) {
    Argparse::printHelp();
    return 2;
  }

  const bool w4a16 = activation_variant.get() == "w4a16";
  const auto activation_storage_dtype = w4a16 ? mllm::kUInt16 : mllm::kUInt8;
  const auto activation_dtype = w4a16 ? mllm::kUInt16PerTensorAsy : mllm::kUInt8PerTensorAsy;
  auto params = mllm::ParameterFile::create();
  auto model = NativeGqaModel("model", w4a16);
  model.load(params);
  auto query = mllm::Tensor::zeros({1, 2, 32, 128}, activation_storage_dtype).setName("query");
  auto key = mllm::Tensor::zeros({1, 1, 128, 1024}, mllm::kUInt8).setName("key_transposed");
  auto value = mllm::Tensor::zeros({1, 1, 1024, 128}, mllm::kUInt8).setName("value");
  auto mask = mllm::Tensor::zeros({1, 1, 32, 1024}, activation_storage_dtype).setName("causal_mask");
  attachAsymmetric(query, params, "native.query", activation_dtype,
                   w4a16 ? 0.0011854973854497075F : 0.32302287220954895F,
                   w4a16 ? 32133 : 122);
  attachAsymmetric(key, params, "native.key", mllm::kUInt8PerTensorAsy,
                   w4a16 ? 0.32952550053596497F : 0.33071592450141907F, 128);
  attachAsymmetric(value, params, "native.value", mllm::kUInt8PerTensorAsy,
                   w4a16 ? 1.407699704170227F : 1.4026927947998047F, 128);
  attachAsymmetric(mask, params, "native.mask", activation_dtype,
                   w4a16 ? 1.5259022489999552e-8F : 3.9215688047988815e-6F,
                   w4a16 ? 65535 : 255);

  mllm::ir::lowlevel::traceStart();
  const auto output = model(query, key, value, mask).front();
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)output;
  mllm::redirect("native_gqa_pre.mir", [&]() { mllm::print(ir); });

  auto qnn_aot_env =
      mllm::qnn::aot::QnnAOTEnv(qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();
  mllm::redirect("native_gqa.mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("EXP-0016 native GQA " + activation_variant.get()
              + " compilation completed: " + output_context.get());
  return 0;
});
