// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Argparse;

namespace {

constexpr int kHidden = 2048;

std::vector<uint8_t> readExact(const std::string& path, size_t size) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.is_open()) throw std::runtime_error("cannot open input: " + path);
  const auto actual = static_cast<size_t>(stream.tellg());
  if (actual != size) {
    throw std::runtime_error("input size mismatch: expected " + std::to_string(size)
                             + ", got " + std::to_string(actual));
  }
  stream.seekg(0);
  std::vector<uint8_t> data(size);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  if (!stream) throw std::runtime_error("failed to read input: " + path);
  return data;
}

void writeExact(const std::string& path, const uint8_t* data, size_t size) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) throw std::runtime_error("cannot open output: " + path);
  stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!stream) throw std::runtime_error("failed to write output: " + path);
}

std::vector<uint64_t> readMacroDurations(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) throw std::runtime_error("profiling-off macro timer is missing: " + path.string());
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
  auto& input_path = Argparse::add<std::string>("--input").help("Native UInt8 input file.");
  auto& output_path = Argparse::add<std::string>("--output").help("Native UInt8 output file.");
  auto& timing_path = Argparse::add<std::string>("--timing_csv").help("Classified profiling-off timings.");
  auto& seq = Argparse::add<int>("--seq").help("Sequence length.");
  auto& warmup = Argparse::add<int>("--warmup").help("Warmup executions.").def(0);
  auto& iterations = Argparse::add<int>("--iterations").help("Measured executions.").def(1);
  auto& profile_level = Argparse::add<std::string>("--profile_level").help("off or optrace.").def("off");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !input_path.isSet()
      || !output_path.isSet() || !timing_path.isSet() || !seq.isSet()) {
    Argparse::printHelp();
    return 2;
  }
  if (seq.get() <= 0 || warmup.get() < 0 || iterations.get() <= 0
      || (profile_level.get() != "off" && profile_level.get() != "optrace")) {
    std::cerr << "invalid sequence, iteration, or profile level\n";
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

  const size_t elements = static_cast<size_t>(seq.get()) * kHidden;
  const auto input_data = readExact(input_path.get(), elements);
  auto input = mllm::Tensor::empty({1, seq.get(), kHidden}, mllm::kUInt8, mllm::kQNN).alloc();
  auto output = mllm::Tensor::empty({1, seq.get(), kHidden}, mllm::kUInt8, mllm::kQNN).alloc();
  input.setName("input");
  output.setName("output");
  std::copy(input_data.begin(), input_data.end(), input.ptr<uint8_t>());
  std::vector<mllm::Tensor> inputs{input};
  std::vector<mllm::Tensor> outputs{output};

  const int total = warmup.get() + iterations.get();
  for (int invocation = 0; invocation < total; ++invocation) {
    backend->graphExecute(graph_name.get(), inputs, outputs);
  }
  writeExact(output_path.get(), output.ptr<uint8_t>(), elements);

  const auto durations = readMacroDurations(macro_file);
  if (durations.size() != static_cast<size_t>(total)) {
    throw std::runtime_error("macro timer row count mismatch: expected " + std::to_string(total)
                             + ", got " + std::to_string(durations.size()));
  }
  std::ofstream timing(timing_file, std::ios::trunc);
  timing << "phase,iteration,graph_execute_us\n";
  std::vector<uint64_t> measured;
  for (size_t i = 0; i < durations.size(); ++i) {
    const bool is_warmup = i < static_cast<size_t>(warmup.get());
    timing << (is_warmup ? "warmup" : "measured") << ','
           << (is_warmup ? i : i - warmup.get()) << ',' << durations[i] << '\n';
    if (!is_warmup) measured.push_back(durations[i]);
  }
  std::cout << "measured_median_us=" << median(measured)
            << " samples=" << measured.size() << '\n';
  return 0;
});
