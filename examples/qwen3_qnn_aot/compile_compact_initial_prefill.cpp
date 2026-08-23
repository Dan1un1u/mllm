// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Compile an end-to-end Qwen3 s32 first-prefill graph.  The full control uses
// the established 1024-column attention width and 992-token placeholder cache;
// the compact candidate omits the semantically empty cache and uses width 32.
// Both variants use the same BOOL8 causal visibility contract so attention
// width is the only graph-level variable in the A/B experiment.

#include <string>

#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/mllm.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kSequence = 32;
constexpr int32_t kContext = 1024;

std::unordered_map<std::string, mllm::Tensor> makeInputs(
    int32_t width, const mllm::models::qwen3::Qwen3Config& config,
    const mllm::ParameterFile::ptr_t& params) {
  auto inputs = qwen3_qnn_aot::makeTraceInputs(
      kSequence, width == kContext ? kContext : kSequence, config, params);
  inputs["causal_mask"] =
      mllm::Tensor::zeros({1, 1, kSequence, width}, mllm::kBool)
          .setName("causal_mask");

  if (width == kSequence) {
    // A first chunk has no semantic past.  Erasing the zero-width placeholders
    // keeps them out of the graph signature and, more importantly, out of all
    // QK/Softmax/AV work.
    for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      inputs.erase("past_key_" + std::to_string(layer));
      inputs.erase("past_value_" + std::to_string(layer));
    }
  }
  return inputs;
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path")
                         .help("Accepted RMSNorm-A8 source model.");
  auto& model_config =
      Argparse::add<std::string>("-c|--config").help("Qwen3 model config.");
  auto& aot_config = Argparse::add<std::string>("-aot_cfg|--aot_config")
                         .help("QNN AOT config.");
  auto& qnn_env = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                      .help("QAIRT x86 library path.");
  auto& output_context =
      Argparse::add<std::string>("-o|--output_context_name")
          .help("Output context.");
  auto& width =
      Argparse::add<int>("--width").help("Attention width: 32 or 1024.");
  auto& logits = Argparse::add<std::string>("--logits")
                     .help("Emit all logits, last-token logits, or no logits.")
                     .def("all");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !model_config.isSet() || !aot_config.isSet() ||
      !qnn_env.isSet() || !output_context.isSet() || !width.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  MLLM_RT_ASSERT(width.get() == kSequence || width.get() == kContext);
  MLLM_RT_ASSERT(logits.get() == "all" || logits.get() == "last" || logits.get() == "none");

  auto config = mllm::models::qwen3::Qwen3Config(model_config.get());
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::models::qwen3::sha::prepareParametersForSHA(params, config);
  auto model = mllm::models::qwen3::sha::Qwen3ForCausalLM_SHA(
      config, logits.get() == "last", logits.get() == "none");
  qwen3_qnn_aot::addCausalMaskParams(params);
  model.load(params);

  auto inputs = makeInputs(width.get(), config, params);
  auto ir = model.trace(inputs, {});
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(
                         aot_config.get()));
  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(
      &qnn_aot_env, aot_config.get(), params));
  pm.run();

  const auto stem = "compact_initial_prefill_w" +
                    std::to_string(width.get()) + "_logits_" + logits.get();
  mllm::redirect(stem + ".mir", [&]() { mllm::print(ir["model"]); });
  qnn_aot_env.saveContext("context.0", output_context.get());
  mllm::print("Compact initial-prefill compilation completed: " +
              output_context.get());
  return 0;
});
