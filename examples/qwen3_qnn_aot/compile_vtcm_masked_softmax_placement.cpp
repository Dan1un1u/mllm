// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile either an interior QK -> custom Softmax -> PV placement graph or a
// graph-boundary numerical fixture, using accepted layer-14 A8 encodings.

#include <cstdlib>
#include <string_view>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/nn/Module.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kHeads = 16;
constexpr int32_t kKvHeads = 4;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kMaximumContext = 1024;
constexpr int32_t kKvGroups = kHeads / kKvHeads;
constexpr int32_t kPairHeads = 2;
constexpr int32_t kPairs = kHeads / kPairHeads;
constexpr std::string_view kLayerPath = "model.layers.14.self_attn.";

std::string qparamPrefix(std::string_view base, int32_t head) {
  return std::string(kLayerPath) + std::string(base) + "_h" + std::to_string(head) + ".fake_quant";
}

bool useVtcmMaskedSoftmax() {
  const char* enabled = std::getenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX");
  return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
}

void duplicateQparams(const mllm::ParameterFile::ptr_t& params) {
  const auto copy = [&](std::string_view base, int32_t count) {
    for (const auto suffix : {std::string_view{"scale"}, std::string_view{"zero_point"}}) {
      const auto source = std::string(kLayerPath) + std::string(base) + ".fake_quant." + std::string(suffix);
      auto value = params->pull(source);
      for (int32_t head = 0; head < count; ++head) {
        const auto destination = qparamPrefix(base, head) + "." + std::string(suffix);
        params->push(destination, value.contiguous().setMemType(mllm::kParamsNormal).setName(destination));
      }
    }
  };

  copy("q_rope_add_0_output_qdq", kHeads);
  copy("k_cast_to_int8_qdq", kKvHeads);
  copy("v_cast_to_int8_qdq", kKvHeads);
  copy("qk_matmul_output_qdq", kHeads);
  copy("scaling_qdq", kHeads);
  copy("mul_0_output_qdq", kHeads);
  copy("softmax_output_qdq", kHeads);
  copy("attn_value_matmul_output_qdq", kHeads);
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params, const std::string& prefix,
                   mllm::DataTypes dtype) {
  tensor = tensor.__unsafeSetDType(dtype);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class PlacementGraph final : public mllm::nn::Module {
 public:
  PlacementGraph(const std::string& name, int32_t context, bool joined_output)
      : mllm::nn::Module(name), context_(context), joined_output_(joined_output) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(kHeads + 2 * kKvHeads + 1));
    auto position_ids = inputs.back();
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kHeads);
    for (int32_t head = 0; head < kHeads; ++head) {
      const int32_t kv_head = head / kKvGroups;
      auto scores =
          mllm::models::qwen3::sha::ptq::QDQ(this, mllm::nn::functional::matmul(inputs[head], inputs[kHeads + kv_head]),
                                             "layers.14.self_attn.qk_matmul_output_qdq_h" + std::to_string(head));
      if (!useVtcmMaskedSoftmax()) {
        auto scale = mllm::Tensor::constant(1.0f / 11.313708498984761f, mllm::kFloat32);
        scale = mllm::models::qwen3::sha::ptq::QDQ(this, scale, "layers.14.self_attn.scaling_qdq_h" + std::to_string(head));
        scores = mllm::models::qwen3::sha::ptq::QDQ(this, scores.mulConstant(scale),
                                                    "layers.14.self_attn.mul_0_output_qdq_h" + std::to_string(head));
      }
      auto causal_positions = position_ids.view({1, 1, position_ids.size(0), 1}, true);
      if (position_ids.size(0) == 32) {
        scores = scores.view({1, 8, 4, context_}, true);
        causal_positions = causal_positions.view({1, 8, 4, 1}, true);
      }
      auto probabilities = mllm::models::qwen3::sha::ptq::QDQ(
          this, scores + causal_positions, "layers.14.self_attn.softmax_output_qdq_h" + std::to_string(head));
      auto output = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(probabilities, inputs[kHeads + kKvHeads + kv_head]),
          "layers.14.self_attn.attn_value_matmul_output_qdq_h" + std::to_string(head));
      if (output.size(1) == 8 && output.size(2) == 4) { output = output.view({1, 1, 32, kHeadDim}, true); }
      outputs.push_back(output);
    }
    if (joined_output_) { return {mllm::nn::functional::concat(outputs, 1)}; }
    return outputs;
  }

 private:
  int32_t context_;
  bool joined_output_;
};

// Keep all attention heads in one logical QNN op chain.  The split graph
// above mirrors the physical per-head lowering seen in the full model; this
// variant tests whether one batched custom-op boundary can amortize the large
// fixed invocation cost.  For decode, 16 one-row heads are viewed as four
// groups of four rows so every 128-byte HVX vector contains useful data.
class PackedPlacementGraph final : public mllm::nn::Module {
 public:
  PackedPlacementGraph(const std::string& name, int32_t context) : mllm::nn::Module(name), context_(context) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(4));
    auto keys = inputs[1];
    auto values = inputs[2];
    keys = keys.repeat(kKvGroups, 1);
    values = values.repeat(kKvGroups, 1);
    auto scores = mllm::models::qwen3::sha::ptq::QDQ(this, mllm::nn::functional::matmul(inputs[0], keys),
                                                     "layers.14.self_attn.qk_matmul_output_qdq_h0");
    auto position_ids = inputs[3];
    auto causal_positions = position_ids.view({1, 1, position_ids.size(0), 1}, true);
    if (inputs[3].size(0) == 1) { scores = scores.view({1, 4, 4, context_}, true); }
    auto probabilities =
        mllm::models::qwen3::sha::ptq::QDQ(this, scores + causal_positions, "layers.14.self_attn.softmax_output_qdq_h0");
    if (inputs[3].size(0) == 1) { probabilities = probabilities.view({1, kHeads, 1, context_}, true); }
    auto output = mllm::models::qwen3::sha::ptq::QDQ(this, mllm::nn::functional::matmul(probabilities, values),
                                                     "layers.14.self_attn.attn_value_matmul_output_qdq_h0");
    return {output};
  }

 private:
  int32_t context_;
};

// Natural Qwen3 GQA grouping: each invocation covers the four query heads
// that share one K/V head.  This preserves four-way inter-group parallelism
// while amortizing the fixed custom-op cost across four rows.
class GroupedPlacementGraph final : public mllm::nn::Module {
 public:
  GroupedPlacementGraph(const std::string& name, int32_t context) : mllm::nn::Module(name), context_(context) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(3 * kKvHeads + 1));
    auto position_ids = inputs.back();
    auto causal_positions = position_ids.view({1, 1, position_ids.size(0), 1}, true);
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kKvHeads);
    for (int32_t group = 0; group < kKvHeads; ++group) {
      auto keys = inputs[kKvHeads + group];
      auto values = inputs[2 * kKvHeads + group];
      keys = keys.repeat(kKvGroups, 1);
      values = values.repeat(kKvGroups, 1);
      const int32_t representative_head = group * kKvGroups;
      auto scores = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(inputs[group], keys),
          "layers.14.self_attn.qk_matmul_output_qdq_h" + std::to_string(representative_head));
      if (position_ids.size(0) == 1) { scores = scores.view({1, 1, kKvGroups, context_}, true); }
      auto probabilities = mllm::models::qwen3::sha::ptq::QDQ(
          this, scores + causal_positions, "layers.14.self_attn.softmax_output_qdq_h" + std::to_string(representative_head));
      if (position_ids.size(0) == 1) { probabilities = probabilities.view({1, kKvGroups, 1, context_}, true); }
      outputs.push_back(mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(probabilities, values),
          "layers.14.self_attn.attn_value_matmul_output_qdq_h" + std::to_string(representative_head)));
    }
    return outputs;
  }

 private:
  int32_t context_;
};

// Intermediate packing point between the original 16 single-head ops and the
// four natural GQA-group ops.  Each pair stays within one GQA group, so both
// heads share K/V while eight independent chains remain available to HTP.
class PairedPlacementGraph final : public mllm::nn::Module {
 public:
  PairedPlacementGraph(const std::string& name, int32_t context) : mllm::nn::Module(name), context_(context) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), static_cast<size_t>(kPairs + 2 * kKvHeads + 1));
    auto position_ids = inputs.back();
    auto causal_positions = position_ids.view({1, 1, position_ids.size(0), 1}, true);
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kPairs);
    for (int32_t pair = 0; pair < kPairs; ++pair) {
      const int32_t kv_head = pair / (kKvGroups / kPairHeads);
      auto keys = inputs[kPairs + kv_head];
      auto values = inputs[kPairs + kKvHeads + kv_head];
      keys = keys.repeat(kPairHeads, 1);
      values = values.repeat(kPairHeads, 1);
      const int32_t representative_head = pair * kPairHeads;
      auto scores = mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(inputs[pair], keys),
          "layers.14.self_attn.qk_matmul_output_qdq_h" + std::to_string(representative_head));
      if (position_ids.size(0) == 1) { scores = scores.view({1, 1, kPairHeads, context_}, true); }
      auto probabilities = mllm::models::qwen3::sha::ptq::QDQ(
          this, scores + causal_positions, "layers.14.self_attn.softmax_output_qdq_h" + std::to_string(representative_head));
      if (position_ids.size(0) == 1) { probabilities = probabilities.view({1, kPairHeads, 1, context_}, true); }
      outputs.push_back(mllm::models::qwen3::sha::ptq::QDQ(
          this, mllm::nn::functional::matmul(probabilities, values),
          "layers.14.self_attn.attn_value_matmul_output_qdq_h" + std::to_string(representative_head)));
    }
    return outputs;
  }

 private:
  int32_t context_;
};

class NumericalGraph final : public mllm::nn::Module {
 public:
  NumericalGraph(const std::string& name, int32_t context, int32_t heads)
      : mllm::nn::Module(name), context_(context), heads_(heads) {}

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    MLLM_RT_ASSERT_EQ(inputs.size(), 2);
    auto scores = inputs[0];
    auto positions = inputs[1];
    if (heads_ > 1 && scores.size(2) == 1) {
      if (heads_ >= kKvGroups) {
        scores = scores.view({1, heads_ / kKvGroups, kKvGroups, context_}, true);
      } else {
        scores = scores.view({1, 1, heads_, context_}, true);
      }
    } else if (heads_ == 1 && scores.size(2) == 32) {
      scores = scores.view({1, 8, 4, context_}, true);
      positions = positions.view({1, 8, 4, 1}, true);
    }
    return {mllm::models::qwen3::sha::ptq::QDQ(this, scores + positions, "layers.14.self_attn.softmax_output_qdq_h0")};
  }

 private:
  int32_t context_;
  int32_t heads_;
};

void compileAndSave(const mllm::ir::IRContext::ptr_t& ir, const mllm::ParameterFile::ptr_t& params,
                    const std::string& qnn_env_path, const std::string& qnn_aot_cfg, const std::string& output_context) {
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path, mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg, params));
  pm.run();
  qnn_aot_env.saveContext("context.0", output_context);
  qnn_aot_env.destroyContext("context.0");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Accepted RMSNorm-A8 model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");
  auto& context_arg =
      Argparse::add<int>("--context").help("KV context width, a multiple of 32 up to 1024.").def(kMaximumContext);
  auto& numerical_fixture = Argparse::add<bool>("--numerical_fixture").help("Expose one custom output.").def(false);
  auto& packed_heads = Argparse::add<bool>("--packed_heads").help("Use one batched 16-head attention chain.").def(false);
  auto& grouped_heads = Argparse::add<bool>("--grouped_heads").help("Use four natural 4-query-head GQA chains.").def(false);
  auto& paired_heads =
      Argparse::add<bool>("--paired_heads").help("Use eight 2-query-head chains within GQA groups.").def(false);
  auto& multithreaded_kernel =
      Argparse::add<bool>("--multithreaded_kernel").help("Use the QHPI self-sliced HVX kernel.").def(false);
  auto& joined_output =
      Argparse::add<bool>("--joined_output").help("Join the 16 split-head outputs at the graph boundary.").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet() || !output_context.isSet() || !seq_arg.isSet()
      || (seq_arg.get() != 1 && seq_arg.get() != 32)) {
    Argparse::printHelp();
    return 2;
  }
  if (context_arg.get() < 32 || context_arg.get() > kMaximumContext || context_arg.get() % 32 != 0
      || context_arg.get() < seq_arg.get()) {
    Argparse::printHelp();
    return 2;
  }
  if (static_cast<int32_t>(packed_heads.get()) + static_cast<int32_t>(grouped_heads.get())
          + static_cast<int32_t>(paired_heads.get())
      > 1) {
    Argparse::printHelp();
    return 2;
  }
  if (joined_output.get() && (packed_heads.get() || grouped_heads.get() || paired_heads.get())) {
    Argparse::printHelp();
    return 2;
  }

  setenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX", "1", 1);
  if (multithreaded_kernel.get()) { setenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX_MT", "1", 1); }
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  qwen3_qnn_aot::addCausalMaskParams(params);
  duplicateQparams(params);

  if (numerical_fixture.get()) {
    const int32_t heads =
        packed_heads.get() ? kHeads : (grouped_heads.get() ? kKvGroups : (paired_heads.get() ? kPairHeads : 1));
    NumericalGraph model("model", context_arg.get(), heads);
    model.load(params);
    auto scores = mllm::Tensor::zeros({1, heads, seq_arg.get(), context_arg.get()}, mllm::kUInt8).setName("scores");
    attachQparams(scores, params, qparamPrefix("qk_matmul_output_qdq", 0), mllm::kUInt8PerTensorAsy);
    auto positions = mllm::Tensor::zeros({1, 1, seq_arg.get(), 1}, mllm::kInt32).setName("position_ids");
    mllm::ir::lowlevel::traceStart();
    auto outputs = model(std::vector<mllm::Tensor>{scores, positions});
    auto ir = mllm::ir::lowlevel::traceStop();
    (void)outputs;
    compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
    mllm::print("Compiled numerical custom Softmax: " + output_context.get());
    return 0;
  }

  if (paired_heads.get()) {
    PairedPlacementGraph model("model", context_arg.get());
    model.load(params);
    std::vector<mllm::Tensor> inputs;
    inputs.reserve(kPairs + 2 * kKvHeads + 1);
    for (int32_t pair = 0; pair < kPairs; ++pair) {
      const int32_t representative_head = pair * kPairHeads;
      auto q =
          mllm::Tensor::zeros({1, kPairHeads, seq_arg.get(), kHeadDim}, mllm::kUInt8).setName("q_p" + std::to_string(pair));
      attachQparams(q, params, qparamPrefix("q_rope_add_0_output_qdq", representative_head), mllm::kUInt8PerTensorAsy);
      inputs.push_back(q);
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      auto k = mllm::Tensor::zeros({1, 1, kHeadDim, context_arg.get()}, mllm::kUInt8).setName("k_g" + std::to_string(group));
      attachQparams(k, params, qparamPrefix("k_cast_to_int8_qdq", group), mllm::kUInt8PerTensorSym);
      inputs.push_back(k);
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      auto v = mllm::Tensor::zeros({1, 1, context_arg.get(), kHeadDim}, mllm::kUInt8).setName("v_g" + std::to_string(group));
      attachQparams(v, params, qparamPrefix("v_cast_to_int8_qdq", group), mllm::kUInt8PerTensorSym);
      inputs.push_back(v);
    }
    inputs.push_back(mllm::Tensor::zeros({seq_arg.get()}, mllm::kInt32).setName("position_ids"));
    mllm::ir::lowlevel::traceStart();
    auto outputs = model(inputs);
    auto ir = mllm::ir::lowlevel::traceStop();
    (void)outputs;
    compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
    mllm::print("Compiled paired-head custom Softmax placement graph: " + output_context.get());
    return 0;
  }

  if (grouped_heads.get()) {
    GroupedPlacementGraph model("model", context_arg.get());
    model.load(params);
    std::vector<mllm::Tensor> inputs;
    inputs.reserve(3 * kKvHeads + 1);
    for (int32_t group = 0; group < kKvHeads; ++group) {
      const int32_t representative_head = group * kKvGroups;
      auto q =
          mllm::Tensor::zeros({1, kKvGroups, seq_arg.get(), kHeadDim}, mllm::kUInt8).setName("q_g" + std::to_string(group));
      attachQparams(q, params, qparamPrefix("q_rope_add_0_output_qdq", representative_head), mllm::kUInt8PerTensorAsy);
      inputs.push_back(q);
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      auto k = mllm::Tensor::zeros({1, 1, kHeadDim, context_arg.get()}, mllm::kUInt8).setName("k_g" + std::to_string(group));
      attachQparams(k, params, qparamPrefix("k_cast_to_int8_qdq", group), mllm::kUInt8PerTensorSym);
      inputs.push_back(k);
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      auto v = mllm::Tensor::zeros({1, 1, context_arg.get(), kHeadDim}, mllm::kUInt8).setName("v_g" + std::to_string(group));
      attachQparams(v, params, qparamPrefix("v_cast_to_int8_qdq", group), mllm::kUInt8PerTensorSym);
      inputs.push_back(v);
    }
    inputs.push_back(mllm::Tensor::zeros({seq_arg.get()}, mllm::kInt32).setName("position_ids"));
    mllm::ir::lowlevel::traceStart();
    auto outputs = model(inputs);
    auto ir = mllm::ir::lowlevel::traceStop();
    (void)outputs;
    compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
    mllm::print("Compiled grouped-head custom Softmax placement graph: " + output_context.get());
    return 0;
  }

  if (packed_heads.get()) {
    PackedPlacementGraph model("model", context_arg.get());
    model.load(params);
    std::vector<mllm::Tensor> inputs;
    auto q = mllm::Tensor::zeros({1, kHeads, seq_arg.get(), kHeadDim}, mllm::kUInt8).setName("q");
    attachQparams(q, params, qparamPrefix("q_rope_add_0_output_qdq", 0), mllm::kUInt8PerTensorAsy);
    inputs.push_back(q);
    auto k = mllm::Tensor::zeros({1, kKvHeads, kHeadDim, context_arg.get()}, mllm::kUInt8).setName("k");
    attachQparams(k, params, qparamPrefix("k_cast_to_int8_qdq", 0), mllm::kUInt8PerTensorSym);
    inputs.push_back(k);
    auto v = mllm::Tensor::zeros({1, kKvHeads, context_arg.get(), kHeadDim}, mllm::kUInt8).setName("v");
    attachQparams(v, params, qparamPrefix("v_cast_to_int8_qdq", 0), mllm::kUInt8PerTensorSym);
    inputs.push_back(v);
    inputs.push_back(mllm::Tensor::zeros({seq_arg.get()}, mllm::kInt32).setName("position_ids"));
    mllm::ir::lowlevel::traceStart();
    auto outputs = model(inputs);
    auto ir = mllm::ir::lowlevel::traceStop();
    (void)outputs;
    compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
    mllm::print("Compiled packed-head custom Softmax placement graph: " + output_context.get());
    return 0;
  }

  PlacementGraph model("model", context_arg.get(), joined_output.get());
  model.load(params);
  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 2 * kKvHeads + 1);
  for (int32_t head = 0; head < kHeads; ++head) {
    auto q = mllm::Tensor::zeros({1, 1, seq_arg.get(), kHeadDim}, mllm::kUInt8).setName("q_h" + std::to_string(head));
    attachQparams(q, params, qparamPrefix("q_rope_add_0_output_qdq", head), mllm::kUInt8PerTensorAsy);
    inputs.push_back(q);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto k = mllm::Tensor::zeros({1, 1, kHeadDim, context_arg.get()}, mllm::kUInt8).setName("k_h" + std::to_string(head));
    attachQparams(k, params, qparamPrefix("k_cast_to_int8_qdq", head), mllm::kUInt8PerTensorSym);
    inputs.push_back(k);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto v = mllm::Tensor::zeros({1, 1, context_arg.get(), kHeadDim}, mllm::kUInt8).setName("v_h" + std::to_string(head));
    attachQparams(v, params, qparamPrefix("v_cast_to_int8_qdq", head), mllm::kUInt8PerTensorSym);
    inputs.push_back(v);
  }
  inputs.push_back(mllm::Tensor::zeros({seq_arg.get()}, mllm::kInt32).setName("position_ids"));
  mllm::ir::lowlevel::traceStart();
  auto outputs = model(inputs);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;
  compileAndSave(ir, params, qnn_env.get(), qnn_aot_cfg.get(), output_context.get());
  mllm::print(std::string(joined_output.get() ? "Compiled joined-output" : "Compiled interior")
              + " custom Softmax placement graph: " + output_context.get());
  return 0;
});
