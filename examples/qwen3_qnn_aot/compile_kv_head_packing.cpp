// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile one layer-14 K/V projection either as the accepted SHA per-head
// expression or as a single packed projection. Both variants use the same
// asymmetric U8 qparams and byte-equivalent W4G32 LPBQ payload.

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
constexpr int32_t kOutputChannels = kHeadChannels * kHeads;

void validateProjection(const std::string& projection) {
  if (projection != "k_proj" && projection != "v_proj") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--projection must be k_proj or v_proj; got '{}'", projection);
  }
}

void validateVariant(const std::string& variant) {
  if (variant != "per_head" && variant != "packed") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--variant must be per_head or packed; got '{}'", variant);
  }
}

std::string opPrefix(const std::string& projection) {
  return "layers.14.self_attn." + projection;
}

std::string qparamPrefix(const std::string& projection, const std::string& side) {
  return "diagnostic.a8." + projection + "." + side;
}

void attachQparams(mllm::Tensor& tensor, const mllm::ParameterFile::ptr_t& params,
                   const std::string& prefix) {
  tensor = tensor.__unsafeSetDType(mllm::kUInt8PerTensorAsy);
  tensor.attach("scale", params->pull(prefix + ".scale").impl(), true);
  tensor.attach("zero_point", params->pull(prefix + ".zero_point").impl(), true);
}

constexpr auto kImpl = mllm::aops::Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32;

class KVProjection final : public mllm::nn::Module {
 public:
  KVProjection(const std::string& name, const std::string& projection,
               const std::string& variant)
      : mllm::nn::Module(name), projection_(projection), variant_(variant) {
    if (variant_ == "packed") {
      packed_ = reg<mllm::nn::Conv2D>(
          opPrefix(projection_), kInputChannels, kOutputChannels,
          std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
          std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}, false, kImpl);
    } else {
      for (int32_t head = 0; head < kHeads; ++head) {
        heads_.emplace_back(reg<mllm::nn::Conv2D>(
            opPrefix(projection_) + "." + std::to_string(head), kInputChannels,
            kHeadChannels, std::vector<int32_t>{1, 1}, std::vector<int32_t>{1, 1},
            std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 1}, false, kImpl));
      }
    }
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = inputs.front();
    x = x.view({1, 1, -1, kInputChannels}, true);
    if (variant_ == "packed") {
      auto output = packed_(x).view({1, -1, kOutputChannels}, true);
      attachQparams(output, getTopParameterFile(), qparamPrefix(projection_, "output"));
      // Preserve the accepted SHA downstream interface. The packed
      // projection must hand eight [B,S,128] tensors to the existing
      // per-head RMSNorm/RoPE/cache path, so expose the same eight graph
      // outputs as the per-head control instead of benchmarking a cheaper
      // one-output graph boundary.
      std::vector<mllm::Tensor> outputs;
      outputs.reserve(kHeads);
      for (int32_t head = 0; head < kHeads; ++head) {
        const int32_t start = head * kHeadChannels;
        auto sliced = output.slice({mllm::kAll, mllm::kAll, {start, start + kHeadChannels}}, true);
        attachQparams(sliced, getTopParameterFile(), qparamPrefix(projection_, "output"));
        outputs.push_back(sliced);
      }
      return outputs;
    }

    std::vector<mllm::Tensor> outputs;
    outputs.reserve(kHeads);
    for (auto& head : heads_) {
      auto output = head(x).view({1, -1, kHeadChannels}, true);
      attachQparams(output, getTopParameterFile(), qparamPrefix(projection_, "output"));
      outputs.push_back(output);
    }
    return outputs;
  }

 private:
  std::string projection_;
  std::string variant_;
  mllm::nn::Conv2D packed_;
  std::vector<mllm::nn::Conv2D> heads_;
};

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Compact diagnostic model.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path").help("QAIRT x86 library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name").help("Output context.");
  auto& projection_arg = Argparse::add<std::string>("--projection").help("k_proj or v_proj.");
  auto& variant_arg = Argparse::add<std::string>("--variant").help("per_head or packed.");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !qnn_aot_cfg.isSet() || !qnn_env.isSet()
      || !output_context.isSet() || !projection_arg.isSet() || !variant_arg.isSet()
      || !seq_arg.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(seq_arg.get() == 1 || seq_arg.get() == 32);
  validateProjection(projection_arg.get());
  validateVariant(variant_arg.get());

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  auto model = KVProjection("model", projection_arg.get(), variant_arg.get());
  model.load(params);
  auto input = mllm::Tensor::zeros({1, seq_arg.get(), kInputChannels}, mllm::kUInt8)
                   .setName("input");
  attachQparams(input, params, qparamPrefix(projection_arg.get(), "input"));

  mllm::ir::lowlevel::traceStart();
  auto outputs = model(input);
  auto ir = mllm::ir::lowlevel::traceStop();
  (void)outputs;

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg.get(), params));
  pm.run();

  const auto stem = "kv_head_packing_split8_" + projection_arg.get() + "_" + variant_arg.get()
                    + "_s" + std::to_string(seq_arg.get());
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("K/V head-packing compilation completed: " + output_context.get());
  return 0;
});
