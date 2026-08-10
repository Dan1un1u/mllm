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
#include "modeling_qwen3_split_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Qwen3 model config file path.");
  auto& qnn_aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.47.0.260601/lib/x86_64-linux-clang/")
                           .help("QNN AOT environment library path.");
  auto& output_context = Argparse::add<std::string>("-o|--output_context_name")
                             .help("Output QNN context path.");
  auto& part = Argparse::add<int>("--part").def(2)
                   .help("1=embedding, 2=layers 0-9, 3=layers 10-19, 4=layers 20-27+lm_head.");
  auto& r3 = Argparse::add<std::string>("--r3").def("none")
                 .help("Post-RoPE R3 realization: none or dense.");
  auto& r1_boundary = Argparse::add<std::string>("--r1_boundary").def("folded")
                          .help("Global R1 boundary: folded/none or online.");
  auto& trace_seq = Argparse::add<int>("--trace_seq").def(0)
                        .help("Trace one graph (1 or 32); 0 traces both graphs.");
  auto& mir_dir = Argparse::add<std::string>("--mir_dir").def(".")
                      .help("Directory for lowered MIR graph dumps.");

  Argparse::parse(argc, argv);
  constexpr int kContextLength = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !model_cfg_path.isSet() || !qnn_aot_cfg.isSet() ||
      !output_context.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "model, config, AOT config, and output context are required");
  }
  if (part.get() < 1 || part.get() > 4) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--part must be 1, 2, 3, or 4");
  }
  if (trace_seq.get() != 0 && trace_seq.get() != 1 && trace_seq.get() != 32) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--trace_seq must be 0, 1, or 32");
  }
  if (r3.get() != "none" && r3.get() != "dense") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "split compiler supports --r3 none or dense");
  }
  if (r1_boundary.get() != "none" && r1_boundary.get() != "folded" &&
      r1_boundary.get() != "online") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--r1_boundary must be none, folded, or online");
  }

  int first_layer = 0;
  int block_count = 0;
  switch (part.get()) {
    case 1: first_layer = 0; block_count = 0; break;
    case 2: first_layer = 0; block_count = 10; break;
    case 3: first_layer = 10; block_count = 10; break;
    case 4: first_layer = 20; block_count = 8; break;
    default: MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "invalid split part");
  }

  auto cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  if (cfg.num_hidden_layers != 28) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "upstream split layout currently requires the 28-layer Qwen3-1.7B config");
  }

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::models::qwen3::sha::prepareParametersForSHA(params, cfg);
  qwen3_qnn_aot::addCausalMaskParams(params);

  auto r3_mode = mllm::models::qwen3::sha::R3Mode::kNone;
  if (r3.get() == "dense") { r3_mode = mllm::models::qwen3::sha::R3Mode::kDense; }
  auto r1_mode = mllm::models::qwen3::sha::R1BoundaryMode::kNone;
  if (r1_boundary.get() == "online") {
    r1_mode = mllm::models::qwen3::sha::R1BoundaryMode::kOnline;
  }

  auto model = mllm::models::qwen3::split_aot::Qwen3SplitPart(
      cfg, part.get(), first_layer, block_count, r3_mode, r1_mode);
  model.load(params);

  auto qnn_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(),
      mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg.get()));
  std::filesystem::create_directories(mir_dir.get());

  auto trace = [&](int seq_len) {
    mllm::models::ARGenerationOutputPast inputs;
    if (part.get() == 1) {
      auto sequence = mllm::Tensor::zeros({1, seq_len}, mllm::kInt32);
      sequence.setName("sequence");
      inputs["sequence"] = sequence;
    } else {
      inputs = qwen3_qnn_aot::block::makeTraceInputs(
          seq_len, kContextLength, cfg, params, first_layer, block_count);
    }
    auto ir = model.trace(inputs, {});
    mllm::ir::PassManager pm(ir.at("model"));
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_env, qnn_aot_cfg.get(), params));
    pm.run();
    const auto path = std::filesystem::path(mir_dir.get()) /
                      ("qwen3_split_part" + std::to_string(part.get()) + "_s" +
                       std::to_string(seq_len) + ".mir");
    mllm::redirect(path.string(), [&]() { mllm::print(ir.at("model")); });
  };

  if (trace_seq.get() == 0 || trace_seq.get() == 32) { trace(32); }
  if (trace_seq.get() == 0 || trace_seq.get() == 1) { trace(1); }
  qnn_env.saveContext("context.0", output_context.get());
  mllm::print("Qwen3 split part " + std::to_string(part.get()) +
              " context written (r3=" + r3.get() + ", r1_boundary=" +
              r1_boundary.get() + ")");
});
