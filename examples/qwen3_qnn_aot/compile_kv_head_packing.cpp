// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compare the accepted per-head K/V LPBQ expression with one packed LPBQ
// projection.  Both graphs expose eight [1,S,128] UInt8 outputs so the only
// semantic variable is projection packing plus the candidate's output slices.

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
constexpr int32_t kHeadChannels = 128;
constexpr int32_t kHeads = 8;
constexpr int32_t kPackedChannels = kHeadChannels * kHeads;

void validateChoice(const std::string& projection, const std::string& mode) {
  if (projection != "k_proj" && projection != "v_proj") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--projection must be k_proj or v_proj; got '{}'", projection);
  }
  if (mode != "split" && mode != "packed") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--mode must be split or packed; got '{}'", mode);
  }
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                   const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

class KVHeadPacking final : public mllm::nn::Module {
 public:
  KVHeadPacking(const std::string& projection, const std::string& mode)
      : mllm::nn::Module("model"), projection_(projection), mode_(mode) {
    const auto prefix = "layers.14.self_attn." + projection_;
    if (mode_ == "split") {
      split_projections_.reserve(kHeads);
      for (int32_t head = 0; head < kHeads; ++head) {
        split_projections_.emplace_back(reg<mllm::nn::Conv2D>(
            prefix + "." + std::to_string(head), kInputChannels, kHeadChannels,
            std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
            std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}, false,
            mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32));
      }
    } else {
      packed_projection_ = reg<mllm::nn::Conv2D>(
          prefix + ".packed", kInputChannels, kPackedChannels,
          std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
          std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}, false,
          mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32);
    }
  }

  std::vector<mllm::Tensor> forward(
      const std::vector<mllm::Tensor>& inputs,
      const std::vector<mllm::AnyValue>& /*args*/) override {
    auto input = inputs.front();
    input = input.view({1, 1, -1, kInputChannels}, true);
    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kHeads);
    const auto qparam_prefix = "diagnostic." + projection_ + ".output";

    if (mode_ == "split") {
      for (int32_t head = 0; head < kHeads; ++head) {
        auto output = split_projections_[head](input);
        attachQparams(output, getTopParameterFile(), qparam_prefix);
        output = output.view({1, -1, kHeadChannels}, true);
        outputs.push_back(output);
      }
      return outputs;
    }

    auto packed = packed_projection_(input);
    attachQparams(packed, getTopParameterFile(), qparam_prefix);
    for (int32_t head = 0; head < kHeads; ++head) {
      auto output = packed.slice(
          {mllm::kAll, mllm::kAll, mllm::kAll,
           {head * kHeadChannels, (head + 1) * kHeadChannels}},
          true);
      attachQparams(output, getTopParameterFile(), qparam_prefix);
      output = output.view({1, -1, kHeadChannels}, true);
      outputs.push_back(output);
    }
    return outputs;
  }

 private:
  std::string projection_;
  std::string mode_;
  std::vector<mllm::nn::Conv2D> split_projections_;
  mllm::nn::Conv2D packed_projection_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact K/V model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& projection_arg = Argparse::add<std::string>("--projection").help("k_proj or v_proj.");
  auto& mode_arg = Argparse::add<std::string>("--mode").help("split or packed.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 32 or 64.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !projection_arg.isSet() || !mode_arg.isSet()
      || !seq_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  validateChoice(projection_arg.get(), mode_arg.get());
  MLLM_RT_ASSERT(seq_arg.get() == 32 || seq_arg.get() == 64);

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = KVHeadPacking(projection_arg.get(), mode_arg.get());
  model.load(params);
  auto input = mllm::Tensor::zeros({1, seq_arg.get(), kInputChannels}, mllm::kUInt8)
                   .setName("input");
  attachQparams(input, params, "diagnostic." + projection_arg.get() + ".input");

  mllm::ir::lowlevel::traceStart();
  auto outputs = model(input);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(),
      mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(
      &qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "kv_head_packing_" + mode_arg.get() + "_" +
                    projection_arg.get() + "_s" + std::to_string(seq_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("K/V head packing compilation completed: " + output_context.get());
  return 0;
});
