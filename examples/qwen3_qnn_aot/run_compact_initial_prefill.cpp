// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Execute an end-to-end Qwen3 s32 first-prefill A/B graph directly through
// QNN.  It deliberately excludes tokenizer and cache-manager overhead so the
// experiment measures the graph change before runtime integration.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kSequence = 32;
constexpr int32_t kContext = 1024;
constexpr int32_t kLayers = 28;
constexpr int32_t kKvHeads = 8;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kHidden = 2048;
constexpr int32_t kVocabulary = 151936;

std::vector<uint64_t> readMacroDurations(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("profiling-off timer missing: " + path.string());
  }
  std::vector<uint64_t> durations;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.starts_with("graph,")) continue;
    const auto separator = line.rfind(',');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid macro timer row: " + line);
    }
    durations.push_back(std::stoull(line.substr(separator + 1)));
  }
  return durations;
}

double median(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  if (values.empty()) return 0.0;
  const auto middle = values.size() / 2;
  return values.size() % 2
             ? static_cast<double>(values[middle])
             : (static_cast<double>(values[middle - 1]) + values[middle]) /
                   2.0;
}

void appendBytes(std::ofstream& stream, const mllm::Tensor& tensor,
                 size_t bytes) {
  stream.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>()),
               static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to write canonical output");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path =
      Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name =
      Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& output_path = Argparse::add<std::string>("--output")
                          .help("Canonical logits + KV output.");
  auto& timing_path = Argparse::add<std::string>("--timing_csv")
                          .help("Classified timings.");
  auto& width =
      Argparse::add<int>("--width").help("Attention width: 32 or 1024.");
  auto& logits_mode = Argparse::add<std::string>("--logits")
                          .help("Graph output logits: all, last, or none.")
                          .def("all");
  auto& warmup =
      Argparse::add<int>("--warmup").help("Warmup executions.").def(0);
  auto& iterations = Argparse::add<int>("--iterations")
                         .help("Measured executions.")
                         .def(1);
  auto& profile_level = Argparse::add<std::string>("--profile_level")
                            .help("off or optrace.")
                            .def("off");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !output_path.isSet() ||
      !timing_path.isSet() || !width.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if ((width.get() != kSequence && width.get() != kContext) ||
      warmup.get() < 0 || iterations.get() <= 0 ||
      (logits_mode.get() != "all" && logits_mode.get() != "last" &&
       logits_mode.get() != "none") ||
      (profile_level.get() != "off" && profile_level.get() != "optrace")) {
    std::cerr << "invalid width, iteration count, or profile level\n";
    return 2;
  }

  const auto timing_file = std::filesystem::absolute(timing_path.get());
  std::filesystem::create_directories(timing_file.parent_path());
  const auto macro_file = timing_file.parent_path() / "qnn_macro_profile.csv";
  setenv("MLLM_QNN_PROFILE_LEVEL", profile_level.get().c_str(), 1);
  setenv("MLLM_QNN_PROFILE_DIR", timing_file.parent_path().c_str(), 1);
  std::ofstream(macro_file, std::ios::trunc)
      << "graph,execution,profiled,captured,graph_execute_us\n";

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(
      mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) throw std::runtime_error("QNN backend is unavailable");

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(width.get() == kContext ? 3 + 2 * kLayers : 3);

  auto input_ids =
      mllm::Tensor::empty({1, kSequence}, mllm::kInt32, mllm::kQNN).alloc();
  input_ids.setName("input_ids");
  for (int32_t token = 0; token < kSequence; ++token) {
    input_ids.ptr<int32_t>()[token] = 1000 + (token * 7919) % 140000;
  }
  inputs.push_back(input_ids);

  auto position_ids =
      mllm::Tensor::empty({kSequence}, mllm::kInt32, mllm::kQNN).alloc();
  position_ids.setName("position_ids");
  for (int32_t token = 0; token < kSequence; ++token) {
    position_ids.ptr<int32_t>()[token] = token;
  }
  inputs.push_back(position_ids);

  auto mask = mllm::Tensor::empty({1, 1, kSequence, width.get()},
                                  mllm::kBool, mllm::kQNN)
                  .alloc();
  mask.setName("attention_mask");
  std::fill_n(mask.ptr<uint8_t>(),
              static_cast<size_t>(kSequence) * width.get(), uint8_t{0});
  const int32_t current_begin = width.get() - kSequence;
  for (int32_t row = 0; row < kSequence; ++row) {
    for (int32_t column = 0; column <= row; ++column) {
      mask.ptr<uint8_t>()[static_cast<size_t>(row) * width.get() +
                          current_begin + column] = 1;
    }
  }
  inputs.push_back(mask);

  if (width.get() == kContext) {
    const int32_t past = kContext - kSequence;
    const size_t cache_bytes =
        static_cast<size_t>(kKvHeads) * kHeadDim * past;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
      auto key = mllm::Tensor::empty({1, kKvHeads, kHeadDim, past},
                                     mllm::kUInt8, mllm::kQNN)
                     .alloc();
      key.setName("past_key_" + std::to_string(layer));
      std::fill_n(key.ptr<uint8_t>(), cache_bytes, uint8_t{128});
      inputs.push_back(key);
    }
    for (int32_t layer = 0; layer < kLayers; ++layer) {
      auto value = mllm::Tensor::empty({1, kKvHeads, past, kHeadDim},
                                       mllm::kUInt8, mllm::kQNN)
                       .alloc();
      value.setName("past_value_" + std::to_string(layer));
      std::fill_n(value.ptr<uint8_t>(), cache_bytes, uint8_t{128});
      inputs.push_back(value);
    }
  }

  std::vector<mllm::Tensor> outputs;
  outputs.reserve(1 + 2 * kLayers);
  const int32_t logits_rows = logits_mode.get() == "last" ? 1 : kSequence;
  const bool emit_logits = logits_mode.get() != "none";
  if (emit_logits) {
    auto logits =
        mllm::Tensor::empty({1, 1, logits_rows, kVocabulary}, mllm::kUInt8,
                            mllm::kQNN)
            .alloc();
    logits.setName("logits");
    outputs.push_back(logits);
  } else {
    auto hidden =
        mllm::Tensor::empty({1, 1, kSequence, kHidden}, mllm::kUInt8,
                            mllm::kQNN)
            .alloc();
    hidden.setName("final_hidden");
    outputs.push_back(hidden);
  }
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    auto key = mllm::Tensor::empty({1, kKvHeads, kHeadDim, kSequence},
                                   mllm::kUInt8, mllm::kQNN)
                   .alloc();
    key.setName("present_key_" + std::to_string(layer));
    outputs.push_back(key);
  }
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    auto value = mllm::Tensor::empty({1, kKvHeads, kSequence, kHeadDim},
                                     mllm::kUInt8, mllm::kQNN)
                     .alloc();
    value.setName("present_value_" + std::to_string(layer));
    outputs.push_back(value);
  }

  const int32_t total = warmup.get() + iterations.get();
  for (int32_t invocation = 0; invocation < total; ++invocation) {
    backend->graphExecute(graph_name.get(), inputs, outputs);
  }

  const auto output_file = std::filesystem::absolute(output_path.get());
  std::filesystem::create_directories(output_file.parent_path());
  std::ofstream canonical(output_file, std::ios::binary | std::ios::trunc);
  if (!canonical.is_open()) {
    throw std::runtime_error("cannot open canonical output: " +
                             output_file.string());
  }
  if (emit_logits) {
    appendBytes(canonical, outputs[0],
                static_cast<size_t>(logits_rows) * kVocabulary);
  } else {
    appendBytes(canonical, outputs[0],
                static_cast<size_t>(kSequence) * kHidden);
  }
  const size_t present_bytes =
      static_cast<size_t>(kKvHeads) * kHeadDim * kSequence;
  for (size_t index = 1; index < outputs.size(); ++index) {
    appendBytes(canonical, outputs[index], present_bytes);
  }

  const auto durations = readMacroDurations(macro_file);
  if (durations.size() != static_cast<size_t>(total)) {
    throw std::runtime_error("unexpected macro timing count");
  }
  std::ofstream timing(timing_file, std::ios::trunc);
  timing << "phase,iteration,graph_execute_us\n";
  for (int32_t invocation = 0; invocation < total; ++invocation) {
    timing << (invocation < warmup.get() ? "warmup" : "measured") << ','
           << invocation << ',' << durations[invocation] << '\n';
  }
  std::vector<uint64_t> measured(durations.begin() + warmup.get(),
                                 durations.end());
  std::cout << "width=" << width.get() << " measured_median_us="
            << median(measured) << " logits=" << logits_mode.get()
            << " output_bytes="
            << std::filesystem::file_size(output_file) << '\n';
  return 0;
});
