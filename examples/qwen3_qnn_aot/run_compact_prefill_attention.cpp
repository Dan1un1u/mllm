// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Execute the 16-query-head / 8-KV-head s32 attention-core micrograph.

#include <algorithm>
#include <cstdint>
#include <cstring>
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

constexpr int32_t kSeq = 32;
constexpr int32_t kContext = 1024;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kQueryHeads = 16;
constexpr int32_t kKvHeads = 8;

std::vector<uint8_t> readExact(const std::string& path, size_t bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.is_open()) throw std::runtime_error("cannot open input: " + path);
  const auto actual = static_cast<size_t>(stream.tellg());
  if (actual != bytes) {
    throw std::runtime_error("input size mismatch for " + path + ": expected " +
                             std::to_string(bytes) + ", got " +
                             std::to_string(actual));
  }
  stream.seekg(0);
  std::vector<uint8_t> data(bytes);
  stream.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to read input: " + path);
  return data;
}

void writeExact(const std::string& path, const uint8_t* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) throw std::runtime_error("cannot open output: " + path);
  stream.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to write output: " + path);
}

std::vector<uint64_t> readMacroDurations(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("profiling-off timer missing: " + path.string());
  }
  std::vector<uint64_t> durations;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.starts_with("graph,")) continue;
    const auto pos = line.rfind(',');
    if (pos == std::string::npos) {
      throw std::runtime_error("invalid macro timer row: " + line);
    }
    durations.push_back(std::stoull(line.substr(pos + 1)));
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

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path =
      Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name =
      Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& query_path =
      Argparse::add<std::string>("--query").help("16 head-major U8 queries.");
  auto& key_path =
      Argparse::add<std::string>("--key").help("8 head-major U8 keys.");
  auto& value_path =
      Argparse::add<std::string>("--value").help("8 head-major U8 values.");
  auto& mask_path =
      Argparse::add<std::string>("--mask").help("U8 causal mask.");
  auto& output_path =
      Argparse::add<std::string>("--output").help("16 head-major U8 output.");
  auto& timing_path =
      Argparse::add<std::string>("--timing_csv").help("Classified timings.");
  auto& width =
      Argparse::add<int>("--width").help("Attention width: 32 or 1024.");
  auto& warmup =
      Argparse::add<int>("--warmup").help("Warmup executions.").def(0);
  auto& iterations =
      Argparse::add<int>("--iterations").help("Measured executions.").def(1);
  auto& profile_level =
      Argparse::add<std::string>("--profile_level")
          .help("off or optrace.")
          .def("off");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !query_path.isSet() ||
      !key_path.isSet() || !value_path.isSet() || !mask_path.isSet() ||
      !output_path.isSet() || !timing_path.isSet() || !width.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if ((width.get() != kSeq && width.get() != kContext) || warmup.get() < 0 ||
      iterations.get() <= 0 ||
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

  const size_t query_head_bytes = static_cast<size_t>(kSeq) * kHeadDim;
  const size_t key_head_bytes = static_cast<size_t>(kHeadDim) * width.get();
  const size_t value_head_bytes = static_cast<size_t>(width.get()) * kHeadDim;
  const size_t mask_bytes = static_cast<size_t>(kSeq) * width.get();
  const auto query_data =
      readExact(query_path.get(), query_head_bytes * kQueryHeads);
  const auto key_data = readExact(key_path.get(), key_head_bytes * kKvHeads);
  const auto value_data =
      readExact(value_path.get(), value_head_bytes * kKvHeads);
  const auto mask_data = readExact(mask_path.get(), mask_bytes);

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kQueryHeads + 2 * kKvHeads + 1);
  for (int32_t head = 0; head < kQueryHeads; ++head) {
    auto query = mllm::Tensor::empty({1, 1, kSeq, kHeadDim}, mllm::kUInt8,
                                     mllm::kQNN)
                     .alloc();
    query.setName("query_" + std::to_string(head));
    std::memcpy(query.ptr<uint8_t>(),
                query_data.data() + query_head_bytes * head,
                query_head_bytes);
    inputs.push_back(query);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto key = mllm::Tensor::empty({1, 1, kHeadDim, width.get()},
                                   mllm::kUInt8, mllm::kQNN)
                   .alloc();
    key.setName("key_" + std::to_string(head));
    std::memcpy(key.ptr<uint8_t>(), key_data.data() + key_head_bytes * head,
                key_head_bytes);
    inputs.push_back(key);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto value = mllm::Tensor::empty({1, 1, width.get(), kHeadDim},
                                     mllm::kUInt8, mllm::kQNN)
                     .alloc();
    value.setName("value_" + std::to_string(head));
    std::memcpy(value.ptr<uint8_t>(),
                value_data.data() + value_head_bytes * head,
                value_head_bytes);
    inputs.push_back(value);
  }
  auto mask = mllm::Tensor::empty({1, 1, kSeq, width.get()}, mllm::kBool,
                                  mllm::kQNN)
                  .alloc();
  mask.setName("causal_mask");
  std::memcpy(mask.ptr<uint8_t>(), mask_data.data(), mask_bytes);
  inputs.push_back(mask);

  std::vector<mllm::Tensor> outputs;
  outputs.reserve(kQueryHeads);
  for (int32_t head = 0; head < kQueryHeads; ++head) {
    auto output = mllm::Tensor::empty({1, 1, kSeq, kHeadDim}, mllm::kUInt8,
                                      mllm::kQNN)
                      .alloc();
    output.setName("output_" + std::to_string(head));
    outputs.push_back(output);
  }

  const int total = warmup.get() + iterations.get();
  for (int invocation = 0; invocation < total; ++invocation) {
    backend->graphExecute(graph_name.get(), inputs, outputs);
  }

  std::vector<uint8_t> canonical(query_head_bytes * kQueryHeads);
  for (int32_t head = 0; head < kQueryHeads; ++head) {
    std::memcpy(canonical.data() + query_head_bytes * head,
                outputs[head].ptr<uint8_t>(), query_head_bytes);
  }
  writeExact(output_path.get(), canonical.data(), canonical.size());

  const auto durations = readMacroDurations(macro_file);
  if (durations.size() != static_cast<size_t>(total)) {
    throw std::runtime_error(
        "macro timer row count mismatch: expected " + std::to_string(total) +
        ", got " + std::to_string(durations.size()));
  }
  std::ofstream timing(timing_file, std::ios::trunc);
  timing << "phase,iteration,graph_execute_us\n";
  std::vector<uint64_t> measured;
  for (size_t index = 0; index < durations.size(); ++index) {
    const bool is_warmup = index < static_cast<size_t>(warmup.get());
    timing << (is_warmup ? "warmup" : "measured") << ','
           << (is_warmup ? index : index - warmup.get()) << ','
           << durations[index] << '\n';
    if (!is_warmup) measured.push_back(durations[index]);
  }
  std::cout << "measured_median_us=" << median(measured)
            << " samples=" << measured.size() << '\n';
  return 0;
});
