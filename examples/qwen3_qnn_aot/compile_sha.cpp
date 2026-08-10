// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Benefits:
// 1. Reduces QNN AOT compilation time
// 2. Improves HTP runtime performance
// 3. Enables better memory locality per head
//
// Usage:
//   ./compile_sha -m /path/to/model.mllm -c /path/to/config.json -aot_cfg /path/to/qnn_aot_cfg.json

#include <filesystem>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/")
                           .help("QNN AOT Environment path.");
  auto& output_context_path = Argparse::add<std::string>("-o|--output_context_name").help("Output QNN context path.");
  auto& schematic_only = Argparse::add<bool>("--schematic_only")
                             .help("Finalize graph(s) and emit Optrace schematics without saving a context binary.");
  auto& trace_seq = Argparse::add<int>("--trace_seq")
                        .def(0)
                        .help("Trace only one graph (1 or 32); 0 traces both graphs.");
  auto& r3 = Argparse::add<std::string>("--r3")
                 .def("none")
                 .help("Post-RoPE R3 realization: none, dense, or fwht-graph.");
  auto& r1_boundary = Argparse::add<std::string>("--r1_boundary")
                          .def("none")
                          .help("Global R1 boundary: none/folded or online reference.");
  auto& mir_dir = Argparse::add<std::string>("--mir_dir")
                      .def(".")
                      .help("Directory for lowered MIR graph dumps.");

  Argparse::parse(argc, argv);

  constexpr int kContextLength = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }
  if (!output_context_path.isSet() && !schematic_only.get()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No output context path provided");
    Argparse::printHelp();
    return -1;
  }
  if (trace_seq.get() != 0 && trace_seq.get() != 1 && trace_seq.get() != 32) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--trace_seq must be 0, 1, or 32");
  }
  if (r3.get() != "none" && r3.get() != "dense" && r3.get() != "fwht-graph") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--r3 must be none, dense, or fwht-graph");
  }
  if (r1_boundary.get() != "none" && r1_boundary.get() != "folded" &&
      r1_boundary.get() != "online") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--r1_boundary must be none, folded, or online");
  }

  auto r3_mode = mllm::models::qwen3::R3Mode::kNone;
  if (r3.get() == "dense") { r3_mode = mllm::models::qwen3::R3Mode::kDense; }
  if (r3.get() == "fwht-graph") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "fused full-model compiler only supports --r3 dense");
  }
  auto r1_mode = mllm::models::qwen3::R1BoundaryMode::kNone;
  if (r1_boundary.get() == "online") { r1_mode = mllm::models::qwen3::R1BoundaryMode::kOnline; }

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  // Load original parameters
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);

  // Create the fused full-model graph. It keeps the original Q/K/V projections
  // and applies the same post-RoPE dense R3 carrier per decoder layer. The
  // folded production checkpoint has no runtime R1 op; online is a reference.
  auto model = mllm::models::qwen3::Qwen3ForCausalLM(model_cfg, r3_mode, r1_mode);

  qwen3_qnn_aot::addCausalMaskParams(params);
  model.load(params);

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  std::filesystem::create_directories(mir_dir.get());
  auto trace_and_dump = [&](int seq_len) {
    const auto mir_path = mir_dir.get() + "/qwen3_qnn_aot_sha_" + std::to_string(seq_len) + ".mir";
    auto trace_inputs = qwen3_qnn_aot::makeTraceInputs(seq_len, kContextLength, model_cfg, params);
    mllm::print("Tracing fused Qwen3 model (seq=" + std::to_string(seq_len) + ")...");
    auto ir = model.trace(trace_inputs, {});
    mllm::print("Fused Qwen3 model traced successfully.");
    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();
    mllm::redirect(mir_path, [&]() { mllm::print(ir["model"]); });
  };

  if (trace_seq.get() == 0 || trace_seq.get() == 32) { trace_and_dump(32); }
  if (trace_seq.get() == 0 || trace_seq.get() == 1) { trace_and_dump(1); }

  if (!schematic_only.get()) {
    qnn_aot_env.saveContext("context.0", output_context_path.get());
  }

  mllm::print("Fused full-model compilation completed successfully (r3=" + r3.get() +
             ", r1_boundary=" + r1_boundary.get() + ")!");
  mllm::print("Output files:");
  if (trace_seq.get() == 0 || trace_seq.get() == 32) {
    mllm::print("  - qwen3_qnn_aot_sha_32.mir (IR dump for seq=32)");
  }
  if (trace_seq.get() == 0 || trace_seq.get() == 1) {
    mllm::print("  - qwen3_qnn_aot_sha_1.mir (IR dump for seq=1)");
  }
  if (!schematic_only.get()) {
    mllm::print("  - " + output_context_path.get() + " (QNN context)");
  }
});
