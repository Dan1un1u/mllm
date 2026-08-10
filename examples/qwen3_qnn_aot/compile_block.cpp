// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <filesystem>
#include <string>

#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/mllm.hpp>

#include "compile_block_common.hpp"
#include "compile_common.hpp"
#include "modeling_qwen3_block_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Standalone block model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Qwen3 model config file path.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.47.0.260601/lib/x86_64-linux-clang/")
                           .help("QNN AOT environment library path.");
  auto& output_context =
      Argparse::add<std::string>("-o|--output_context_name").help("Output QNN context path.");
  auto& layer = Argparse::add<int>("--layer").def(5).help("Transformer layer index; prototype contract uses Layer 5.");
  auto& block_count = Argparse::add<int>("--block_count").def(1).help("Number of consecutive decoder blocks.");
  auto& r3 = Argparse::add<std::string>("--r3")
                 .def("none")
                 .help("Post-RoPE R3 realization: none, dense, or fwht-graph.");
  auto& trace_seq = Argparse::add<int>("--trace_seq")
                        .def(0)
                        .help("Trace one graph (1 or 32); 0 traces both into one context.");
  auto& mir_dir = Argparse::add<std::string>("--mir_dir").def(".").help("Directory for MIR dumps.");

  Argparse::parse(argc, argv);
  constexpr int kContextLength = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !model_cfg_path.isSet() || !qnn_aot_cfg.isSet() || !output_context.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "model, config, AOT config, and output context are required");
  }
  if (layer.get() < 0 || block_count.get() <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "layer must be non-negative and block_count must be positive");
  }
  if (trace_seq.get() != 0 && trace_seq.get() != 1 && trace_seq.get() != 32) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--trace_seq must be 0, 1, or 32");
  }

  auto r3_mode = mllm::models::qwen3::sha::R3Mode::kNone;
  if (r3.get() == "dense") {
    r3_mode = mllm::models::qwen3::sha::R3Mode::kDense;
  } else if (r3.get() == "fwht-graph") {
    r3_mode = mllm::models::qwen3::sha::R3Mode::kFWHTGraph;
  } else if (r3.get() != "none") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--r3 must be none, dense, or fwht-graph");
  }

  std::filesystem::create_directories(mir_dir.get());
  auto cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  if (layer.get() + block_count.get() > cfg.num_hidden_layers) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "consecutive block range exceeds model depth");
  }
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::models::qwen3::sha::prepareParametersForSHA(params, cfg);
  qwen3_qnn_aot::addCausalMaskParams(params);

  auto model = mllm::models::qwen3::block_aot::Qwen3StandaloneBlock(cfg, layer.get(), block_count.get(), r3_mode);
  model.load(params);

  auto qnn_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));

  auto trace = [&](int seq_len) {
    auto inputs = qwen3_qnn_aot::block::makeTraceInputs(
        seq_len, kContextLength, cfg, params, layer.get(), block_count.get());
    auto ir = model.trace(inputs, {});
    mllm::ir::PassManager pm(ir.at("model"));
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_env, qnn_aot_cfg.get(), params));
    pm.run();
    const auto path = std::filesystem::path(mir_dir.get()) /
                      ("qwen3_layer5_block_s" + std::to_string(seq_len) + ".mir");
    mllm::redirect(path.string(), [&]() { mllm::print(ir.at("model")); });
  };

  if (trace_seq.get() == 0 || trace_seq.get() == 32) { trace(32); }
  if (trace_seq.get() == 0 || trace_seq.get() == 1) { trace(1); }
  qnn_env.saveContext("context.0", output_context.get());
  mllm::print("Standalone Layer 5 SHA context written to " + output_context.get() + " (R3=" + r3.get() + ")");
});
