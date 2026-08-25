// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0017 compiler. It compares the accepted split-head attention core with
// QAIRT 2.49 native U8 GroupQueryAttention over an integration-realistic
// layer-14 S=32, total-context=1024 boundary.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/backends/base/PluginInterface.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/ir/linalg/Op.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>

using mllm::Argparse;

namespace {

constexpr int32_t kQueryHeads = 16;
constexpr int32_t kKvHeads = 8;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kSequence = 32;
constexpr int32_t kPast = 992;
constexpr int32_t kContext = kPast + kSequence;

struct QParams {
  float scale;
  int32_t zero_point;
};

struct QuantContract {
  mllm::DataTypes activation_dtype;
  mllm::DataTypes activation_storage;
  QParams query;
  QParams key;
  QParams value;
  QParams qk;
  QParams scaled;
  QParams row_min;
  QParams minus_twenty;
  QParams masked_value;
  QParams mask;
  QParams selected;
  QParams probability;
  QParams output;
};

QuantContract contract(bool a16) {
  if (a16) {
    return {
        mllm::kUInt16PerTensorAsy,
        mllm::kUInt16,
        {0.0011854973854497075F, 32133},
        {0.32952550053596497F, 128},
        {1.407699704170227F, 128},
        {0.007394818589091301F, 26930},
        {0.0006536157452501357F, 26931},
        {0.0003613728331401944F, 48709},
        {0.0003051804378628731F, 65535},
        {0.0005737727624364197F, 65535},
        {1.5259022489999552e-8F, 65535},
        {0.0009587961831130087F, 39218},
        {1.5259021893143654e-5F, 0},
        {0.0021023168228566647F, 58560},
    };
  }
  return {
      mllm::kUInt8PerTensorAsy,
      mllm::kUInt8,
      {0.32302287220954895F, 122},
      {0.33071592450141907F, 128},
      {1.4026927947998047F, 128},
      {2.030411958694458F, 113},
      {0.17946475744247437F, 113},
      {0.1029057577252388F, 197},
      {0.0784313753247261F, 255},
      {0.15794645249843597F, 255},
      {3.9215688047988815e-6F, 255},
      {0.2578960955142975F, 156},
      {1.0F / 255.0F, 0},
      {0.5333649516105652F, 229},
  };
}

void attachAsymmetric(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                      const std::string& prefix, mllm::DataTypes dtype, QParams qparams) {
  params->push(prefix + ".scale", mllm::Tensor::constant(qparams.scale, mllm::kFloat32));
  params->push(prefix + ".zero_point",
               mllm::Tensor::constant(static_cast<float>(qparams.zero_point), mllm::kInt32));
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

void attachConstantQuantization(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                                const std::string& prefix, QParams qparams) {
  params->push(prefix + ".scale", mllm::Tensor::constant(qparams.scale, mllm::kFloat32));
  params->push(prefix + ".zero_point",
               mllm::Tensor::constant(static_cast<float>(qparams.zero_point), mllm::kInt32));
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class NativeGqaMarker final : public mllm::plugin::interface::CustomizedOp {
 public:
  NativeGqaMarker() : CustomizedOp("EXP0017NativeGroupQueryAttention") {
    setName("model.layers.14.self_attn.exp0017_native_group_query_attention");
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

class AttentionCore final : public mllm::nn::Module {
 public:
  AttentionCore(const std::string& name, bool native_gqa, bool native_rotary, bool a16)
      : mllm::nn::Module(name), native_gqa_(native_gqa), native_rotary_(native_rotary), a16_(a16),
        native_op_(std::make_shared<NativeGqaMarker>()) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    const size_t expected = native_rotary_ ? 39 : (native_gqa_ ? 36 : 35);
    if (inputs.size() != expected) {
      throw std::runtime_error("EXP-0017 attention input count mismatch");
    }
    std::vector<mllm::Tensor> query_heads(inputs.begin(), inputs.begin() + kQueryHeads);
    std::vector<mllm::Tensor> key_heads(inputs.begin() + kQueryHeads,
                                        inputs.begin() + kQueryHeads + kKvHeads);
    std::vector<mllm::Tensor> value_heads(inputs.begin() + kQueryHeads + kKvHeads,
                                          inputs.begin() + kQueryHeads + 2 * kKvHeads);
    const auto& past_key = inputs[32];
    const auto& past_value = inputs[33];
    return native_gqa_ ? runNative(query_heads, key_heads, value_heads, past_key, past_value,
                                   inputs[34], inputs[35],
                                   native_rotary_ ? std::vector<mllm::Tensor>{inputs[36], inputs[37], inputs[38]}
                                                  : std::vector<mllm::Tensor>{})
                       : runDecomposed(query_heads, key_heads, value_heads, past_key, past_value,
                                       inputs[34]);
  }

 private:
  std::vector<mllm::Tensor> runNative(std::vector<mllm::Tensor>& query_heads,
                                      std::vector<mllm::Tensor>& key_heads,
                                      std::vector<mllm::Tensor>& value_heads,
                                      const mllm::Tensor& past_key, const mllm::Tensor& past_value,
                                      const mllm::Tensor& seqlens, const mllm::Tensor& total_sequence,
                                      const std::vector<mllm::Tensor>& rotary_inputs) {
    const auto q = contract(false);
    auto packed_query = mllm::nn::functional::concat(query_heads, 1);
    attachAsymmetric(packed_query, getTopParameterFile(), "native.pack_query_bnsh",
                     q.activation_dtype, q.query);
    packed_query = packed_query.transpose(1, 2).view({1, kSequence, kQueryHeads * kHeadDim}, true);
    attachAsymmetric(packed_query, getTopParameterFile(), "native.pack_query_bsh",
                     q.activation_dtype, q.query);

    auto packed_key = mllm::nn::functional::concat(key_heads, 1);
    attachAsymmetric(packed_key, getTopParameterFile(), "native.pack_key_bnsh",
                     q.activation_dtype, q.key);
    packed_key = packed_key.transpose(1, 2).view({1, kSequence, kKvHeads * kHeadDim}, true);
    attachAsymmetric(packed_key, getTopParameterFile(), "native.pack_key_bsh",
                     q.activation_dtype, q.key);

    auto packed_value = mllm::nn::functional::concat(value_heads, 1);
    attachAsymmetric(packed_value, getTopParameterFile(), "native.pack_value_bnsh",
                     q.activation_dtype, q.value);
    packed_value = packed_value.transpose(1, 2).view({1, kSequence, kKvHeads * kHeadDim}, true);
    attachAsymmetric(packed_value, getTopParameterFile(), "native.pack_value_bsh",
                     q.activation_dtype, q.value);

    auto past_key_bnsh = past_key;
    past_key_bnsh = past_key_bnsh.transpose(2, 3);
    attachAsymmetric(past_key_bnsh, getTopParameterFile(), "native.past_key_bnsh",
                     q.activation_dtype, q.key);

    std::vector<mllm::Tensor> outputs{
        mllm::Tensor::empty({1, kSequence, kQueryHeads * kHeadDim}, mllm::kUInt8).setName("attention_output"),
        mllm::Tensor::empty({1, kKvHeads, kContext, kHeadDim}, mllm::kUInt8).setName("present_key"),
        mllm::Tensor::empty({1, kKvHeads, kContext, kHeadDim}, mllm::kUInt8).setName("present_value"),
    };
    attachAsymmetric(outputs[0], getTopParameterFile(), "native.output", q.activation_dtype, q.output);
    attachAsymmetric(outputs[1], getTopParameterFile(), "native.present_key", q.activation_dtype, q.key);
    attachAsymmetric(outputs[2], getTopParameterFile(), "native.present_value", q.activation_dtype, q.value);
    auto trace_context = mllm::Context::instance().thisThread()->ir_context;
    std::vector<mllm::Tensor> native_inputs{
        packed_query, seqlens, total_sequence, packed_key, packed_value, past_key_bnsh, past_value};
    native_inputs.insert(native_inputs.end(), rotary_inputs.begin(), rotary_inputs.end());
    native_op_->trace(trace_context.get(), native_inputs, outputs);

    auto new_key = outputs[1].slice({mllm::kAll, mllm::kAll, {kPast, kContext}, mllm::kAll}, true)
                       .transpose(2, 3);
    attachAsymmetric(new_key, getTopParameterFile(), "native.new_key", q.activation_dtype, q.key);
    auto new_value = outputs[2].slice({mllm::kAll, mllm::kAll, {kPast, kContext}, mllm::kAll}, true);
    attachAsymmetric(new_value, getTopParameterFile(), "native.new_value", q.activation_dtype, q.value);
    return {outputs[0], new_key, new_value};
  }

  std::vector<mllm::Tensor> runDecomposed(std::vector<mllm::Tensor>& query_heads,
                                          std::vector<mllm::Tensor>& key_heads,
                                          std::vector<mllm::Tensor>& value_heads,
                                          const mllm::Tensor& past_key,
                                          const mllm::Tensor& past_value,
                                          mllm::Tensor mask) {
    const auto q = contract(a16_);
    std::vector<mllm::Tensor> full_keys;
    std::vector<mllm::Tensor> full_values;
    std::vector<mllm::Tensor> new_keys_transposed;
    full_keys.reserve(kKvHeads);
    full_values.reserve(kKvHeads);
    new_keys_transposed.reserve(kKvHeads);
    for (int kv = 0; kv < kKvHeads; ++kv) {
      const auto prefix = "decomposed.kv" + std::to_string(kv);
      auto past_k = past_key.slice({mllm::kAll, {kv, kv + 1}, mllm::kAll, mllm::kAll}, true);
      attachAsymmetric(past_k, getTopParameterFile(), prefix + ".past_key", mllm::kUInt8PerTensorAsy, q.key);
      auto new_k = key_heads[kv].transpose(2, 3);
      attachAsymmetric(new_k, getTopParameterFile(), prefix + ".new_key_transposed",
                       mllm::kUInt8PerTensorAsy, q.key);
      new_keys_transposed.push_back(new_k);
      auto full_k = mllm::nn::functional::concat({past_k, new_k}, 3);
      attachAsymmetric(full_k, getTopParameterFile(), prefix + ".full_key",
                       mllm::kUInt8PerTensorAsy, q.key);
      full_keys.push_back(full_k);

      auto past_v = past_value.slice({mllm::kAll, {kv, kv + 1}, mllm::kAll, mllm::kAll}, true);
      attachAsymmetric(past_v, getTopParameterFile(), prefix + ".past_value",
                       mllm::kUInt8PerTensorAsy, q.value);
      auto full_v = mllm::nn::functional::concat({past_v, value_heads[kv]}, 2);
      attachAsymmetric(full_v, getTopParameterFile(), prefix + ".full_value",
                       mllm::kUInt8PerTensorAsy, q.value);
      full_values.push_back(full_v);
    }

    std::vector<mllm::Tensor> head_outputs;
    head_outputs.reserve(kQueryHeads);
    for (int head = 0; head < kQueryHeads; ++head) {
      const auto prefix = "decomposed.head" + std::to_string(head);
      auto score = mllm::nn::functional::matmul(query_heads[head], full_keys[head / 2]);
      attachAsymmetric(score, getTopParameterFile(), prefix + ".qk", q.activation_dtype, q.qk);
      auto scale = mllm::Tensor::constant(1.0F / std::sqrt(static_cast<float>(kHeadDim)), mllm::kFloat32);
      attachConstantQuantization(scale, getTopParameterFile(), prefix + ".scale",
                                 a16_ ? QParams{1.3487197065842338e-6F, 0}
                                      : QParams{0.0003466209745965898F, 0});
      score = score.mulConstant(scale);
      attachAsymmetric(score, getTopParameterFile(), prefix + ".scaled", q.activation_dtype, q.scaled);
      auto row_min = score.min(-1, true);
      attachAsymmetric(row_min, getTopParameterFile(), prefix + ".row_min", q.activation_dtype, q.row_min);
      auto minus_twenty = mllm::Tensor::constant(-20.0F, mllm::kFloat32);
      attachConstantQuantization(minus_twenty, getTopParameterFile(), prefix + ".minus_twenty", q.minus_twenty);
      auto masked_value = row_min.addConstant(minus_twenty);
      attachAsymmetric(masked_value, getTopParameterFile(), prefix + ".masked_value",
                       q.activation_dtype, q.masked_value);
      auto zero = mllm::Tensor::constant(0.0F, mllm::kFloat32);
      attachConstantQuantization(zero, getTopParameterFile(), prefix + ".zero", q.mask);
      auto selected = mllm::nn::functional::where(mask.equalConstant(zero), score, masked_value);
      attachAsymmetric(selected, getTopParameterFile(), prefix + ".selected",
                       q.activation_dtype, q.selected);
      auto probability = mllm::nn::functional::softmax(selected, -1);
      attachAsymmetric(probability, getTopParameterFile(), prefix + ".probability",
                       q.activation_dtype, q.probability);
      auto output = mllm::nn::functional::matmul(probability, full_values[head / 2]);
      attachAsymmetric(output, getTopParameterFile(), prefix + ".output",
                       q.activation_dtype, q.output);
      head_outputs.push_back(output);
    }
    auto packed = mllm::nn::functional::concat(head_outputs, 1).transpose(1, 2)
                      .view({1, kSequence, kQueryHeads * kHeadDim}, true);
    attachAsymmetric(packed, getTopParameterFile(), "decomposed.output", q.activation_dtype, q.output);
    auto new_key = mllm::nn::functional::concat(new_keys_transposed, 1);
    attachAsymmetric(new_key, getTopParameterFile(), "decomposed.new_key",
                     mllm::kUInt8PerTensorAsy, q.key);
    auto new_value = mllm::nn::functional::concat(value_heads, 1);
    attachAsymmetric(new_value, getTopParameterFile(), "decomposed.new_value",
                     mllm::kUInt8PerTensorAsy, q.value);
    return {packed, new_key, new_value};
  }

  bool native_gqa_;
  bool native_rotary_;
  bool a16_;
  std::shared_ptr<NativeGqaMarker> native_op_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& variant = Argparse::add<std::string>("--variant").help(
      "native_gqa, native_gqa_rotary, or decomposed.");
  auto& activation = Argparse::add<std::string>("--activation").help("a8 or a16.").def("a8");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet() || !variant.isSet()
      || (variant.get() != "native_gqa" && variant.get() != "native_gqa_rotary"
          && variant.get() != "decomposed")
      || (activation.get() != "a8" && activation.get() != "a16")
      || (variant.get() != "decomposed" && activation.get() != "a8")) {
    Argparse::printHelp();
    return 2;
  }

  const bool native_gqa = variant.get() != "decomposed";
  const bool native_rotary = variant.get() == "native_gqa_rotary";
  const bool a16 = activation.get() == "a16";
  const auto q = contract(a16);
  auto params = mllm::ParameterFile::create();
  auto model = AttentionCore("model", native_gqa, native_rotary, a16);
  model.load(params);

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(native_rotary ? 39 : (native_gqa ? 36 : 35));
  for (int head = 0; head < kQueryHeads; ++head) {
    auto tensor = mllm::Tensor::zeros({1, 1, kSequence, kHeadDim}, q.activation_storage)
                      .setName("query_head_" + std::to_string(head));
    attachAsymmetric(tensor, params, "input.query." + std::to_string(head), q.activation_dtype, q.query);
    inputs.push_back(tensor);
  }
  for (int head = 0; head < kKvHeads; ++head) {
    auto tensor = mllm::Tensor::zeros({1, 1, kSequence, kHeadDim}, mllm::kUInt8)
                      .setName("key_head_" + std::to_string(head));
    attachAsymmetric(tensor, params, "input.key." + std::to_string(head),
                     mllm::kUInt8PerTensorAsy, q.key);
    inputs.push_back(tensor);
  }
  for (int head = 0; head < kKvHeads; ++head) {
    auto tensor = mllm::Tensor::zeros({1, 1, kSequence, kHeadDim}, mllm::kUInt8)
                      .setName("value_head_" + std::to_string(head));
    attachAsymmetric(tensor, params, "input.value." + std::to_string(head),
                     mllm::kUInt8PerTensorAsy, q.value);
    inputs.push_back(tensor);
  }
  const int32_t past_length = native_rotary ? kContext : kPast;
  auto past_key = mllm::Tensor::zeros({1, kKvHeads, kHeadDim, past_length}, mllm::kUInt8)
                      .setName("past_key");
  attachAsymmetric(past_key, params, "input.past_key", mllm::kUInt8PerTensorAsy, q.key);
  inputs.push_back(past_key);
  auto past_value = mllm::Tensor::zeros({1, kKvHeads, past_length, kHeadDim}, mllm::kUInt8)
                        .setName("past_value");
  attachAsymmetric(past_value, params, "input.past_value", mllm::kUInt8PerTensorAsy, q.value);
  inputs.push_back(past_value);
  if (native_gqa) {
    inputs.push_back(mllm::Tensor::zeros({1}, mllm::kInt32).setName("seqlens_k_minus_one"));
    inputs.push_back(mllm::Tensor::zeros({}, mllm::kInt32).setName("total_sequence_length"));
    if (native_rotary) {
      auto cos_cache = mllm::Tensor::zeros({kContext, kHeadDim / 2}, mllm::kUInt8)
                           .setName("cos_cache");
      attachAsymmetric(cos_cache, params, "input.cos_cache", mllm::kUInt8PerTensorAsy,
                       {2.0F / 255.0F, 128});
      inputs.push_back(cos_cache);
      auto sin_cache = mllm::Tensor::zeros({kContext, kHeadDim / 2}, mllm::kUInt8)
                           .setName("sin_cache");
      attachAsymmetric(sin_cache, params, "input.sin_cache", mllm::kUInt8PerTensorAsy,
                       {2.0F / 255.0F, 128});
      inputs.push_back(sin_cache);
      inputs.push_back(
          mllm::Tensor::zeros({1, kSequence}, mllm::kInt64).setName("position_ids"));
    }
  } else {
    auto mask = mllm::Tensor::zeros({1, 1, kSequence, kContext}, q.activation_storage).setName("causal_mask");
    attachAsymmetric(mask, params, "input.mask", q.activation_dtype, q.mask);
    inputs.push_back(mask);
  }

  mllm::ir::lowlevel::traceStart();
  const auto outputs = model(inputs);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;
  const auto stem = "exp0017_" + variant.get() + "_" + activation.get();
  mllm::redirect(stem + "_pre.mir", [&]() { mllm::print(ir); });

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("EXP-0017 compilation completed: " + variant.get() + " " + activation.get());
  return 0;
});
