// Copyright (c) MLLM Team.
// Licensed under the MIT License.

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

constexpr int32_t kHeads = 16;
constexpr int32_t kContext = 1024;

std::vector<uint8_t> readExact(const std::string& path, size_t bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.is_open()) throw std::runtime_error("cannot open input: " + path);
  const auto actual = static_cast<size_t>(stream.tellg());
  if (actual != bytes) {
    throw std::runtime_error("input size mismatch: expected " + std::to_string(bytes) + ", got " + std::to_string(actual));
  }
  stream.seekg(0);
  std::vector<uint8_t> data(bytes);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to read input: " + path);
  return data;
}

void writeExact(const std::string& path, const uint8_t* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) throw std::runtime_error("cannot open output: " + path);
  stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to write output: " + path);
}

std::vector<uint64_t> readMacroDurations(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) throw std::runtime_error("profiling-off timer missing: " + path.string());
  std::vector<uint64_t> durations;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.starts_with("graph,")) continue;
    const auto pos = line.rfind(',');
    if (pos == std::string::npos) throw std::runtime_error("invalid macro timer row: " + line);
    durations.push_back(std::stoull(line.substr(pos + 1)));
  }
  return durations;
}

double median(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  if (values.empty()) return 0.0;
  const auto middle = values.size() / 2;
  return values.size() % 2 ? static_cast<double>(values[middle])
                           : (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name = Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& attn_path = Argparse::add<std::string>("--attn").help("Sixteen head-major U8 logits.");
  auto& mask_path = Argparse::add<std::string>("--mask").help("U8 causal mask.");
  auto& output_path = Argparse::add<std::string>("--output").help("Head-major U8 output.");
  auto& timing_path = Argparse::add<std::string>("--timing_csv").help("Classified timings.");
  auto& seq = Argparse::add<int>("--seq").help("Sequence length.");
  auto& warmup = Argparse::add<int>("--warmup").help("Warmup executions.").def(0);
  auto& iterations = Argparse::add<int>("--iterations").help("Measured executions.").def(1);
  auto& profile_level = Argparse::add<std::string>("--profile_level").help("off or optrace.").def("off");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !attn_path.isSet() || !mask_path.isSet() || !output_path.isSet()
      || !timing_path.isSet() || !seq.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if ((seq.get() != 1 && seq.get() != 32) || warmup.get() < 0 || iterations.get() <= 0
      || (profile_level.get() != "off" && profile_level.get() != "optrace")) {
    std::cerr << "invalid sequence, iteration count, or profile level\n";
    return 2;
  }

  const auto timing_file = std::filesystem::absolute(timing_path.get());
  std::filesystem::create_directories(timing_file.parent_path());
  const auto macro_file = timing_file.parent_path() / "qnn_macro_profile.csv";
  setenv("MLLM_QNN_PROFILE_LEVEL", profile_level.get().c_str(), 1);
  setenv("MLLM_QNN_PROFILE_DIR", timing_file.parent_path().c_str(), 1);
  std::ofstream(macro_file, std::ios::trunc) << "graph,execution,profiled,captured,graph_execute_us\n";

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) throw std::runtime_error("QNN backend is unavailable");

  const size_t head_bytes = static_cast<size_t>(seq.get()) * kContext;
  const auto attn_data = readExact(attn_path.get(), head_bytes * kHeads);
  const auto mask_data = readExact(mask_path.get(), head_bytes);
  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 1);
  for (int32_t head = 0; head < kHeads; ++head) {
    auto input = mllm::Tensor::empty({1, 1, seq.get(), kContext}, mllm::kUInt8, mllm::kQNN).alloc();
    input.setName("attn_" + std::to_string(head));
    std::memcpy(input.ptr<uint8_t>(), attn_data.data() + head_bytes * head, head_bytes);
    inputs.push_back(input);
  }
  auto mask = mllm::Tensor::empty({1, 1, seq.get(), kContext}, mllm::kUInt8, mllm::kQNN).alloc();
  mask.setName("causal_mask");
  std::memcpy(mask.ptr<uint8_t>(), mask_data.data(), head_bytes);
  inputs.push_back(mask);

  std::vector<mllm::Tensor> outputs;
  outputs.reserve(kHeads);
  for (int32_t head = 0; head < kHeads; ++head) {
    auto output = mllm::Tensor::empty({1, 1, seq.get(), kContext}, mllm::kUInt8, mllm::kQNN).alloc();
    output.setName("output_" + std::to_string(head));
    outputs.push_back(output);
  }

  const int total = warmup.get() + iterations.get();
  for (int invocation = 0; invocation < total; ++invocation) { backend->graphExecute(graph_name.get(), inputs, outputs); }

  std::vector<uint8_t> canonical(head_bytes * kHeads);
  for (int32_t head = 0; head < kHeads; ++head) {
    std::memcpy(canonical.data() + head_bytes * head, outputs[head].ptr<uint8_t>(), head_bytes);
  }
  writeExact(output_path.get(), canonical.data(), canonical.size());

  const auto durations = readMacroDurations(macro_file);
  if (durations.size() != static_cast<size_t>(total)) {
    throw std::runtime_error("macro timer row count mismatch: expected " + std::to_string(total) + ", got "
                             + std::to_string(durations.size()));
  }
  std::ofstream timing(timing_file, std::ios::trunc);
  timing << "phase,iteration,graph_execute_us\n";
  std::vector<uint64_t> measured;
  for (size_t index = 0; index < durations.size(); ++index) {
    const bool is_warmup = index < static_cast<size_t>(warmup.get());
    timing << (is_warmup ? "warmup" : "measured") << ',' << (is_warmup ? index : index - warmup.get()) << ','
           << durations[index] << '\n';
    if (!is_warmup) measured.push_back(durations[index]);
  }
  std::cout << "measured_median_us=" << median(measured) << " samples=" << measured.size() << '\n';
  return 0;
});
