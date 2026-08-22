// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdlib>
#include <cstring>
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

constexpr int32_t kHeads = 16;
constexpr int32_t kKvHeads = 4;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kContext = 1024;

mllm::Tensor makeZeroU8Tensor(const std::vector<int32_t>& shape) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt8, mllm::kQNN).alloc();
  std::memset(tensor.ptr<uint8_t>(), 0, tensor.numel());
  return tensor;
}

mllm::Tensor loadU8Tensor(const std::vector<int32_t>& shape, const std::string& path) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt8, mllm::kQNN).alloc();
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(tensor.ptr<uint8_t>()), tensor.numel()) || input.peek() != EOF) {
    throw std::runtime_error("Expected exactly " + std::to_string(tensor.numel()) + " bytes in " + path);
  }
  return tensor;
}

void saveU8Tensor(const mllm::Tensor& tensor, const std::string& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>()), tensor.numel());
  if (!output.good()) { throw std::runtime_error("Failed to write " + path); }
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& seq = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");
  auto& profile_dir = Argparse::add<std::string>("--profile_dir").help("Optrace output directory.");
  auto& iterations = Argparse::add<int>("--iterations").help("Execution count.").def(1);
  auto& numerical_fixture = Argparse::add<bool>("--numerical_fixture").help("Run the post-gate numerical fixture.").def(false);
  auto& scores_file = Argparse::add<std::string>("--scores_file").help("Raw U8 score fixture.");
  auto& mask_file = Argparse::add<std::string>("--mask_file").help("Raw U8 mask fixture.");
  auto& output_file = Argparse::add<std::string>("--output_file").help("Raw U8 probability output.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !seq.isSet() || !profile_dir.isSet() || (seq.get() != 1 && seq.get() != 32)
      || iterations.get() <= 0) {
    Argparse::printHelp();
    return 2;
  }
  if (numerical_fixture.get() && (!scores_file.isSet() || !mask_file.isSet() || !output_file.isSet())) {
    Argparse::printHelp();
    return 2;
  }

  std::filesystem::create_directories(profile_dir.get());
  setenv("MLLM_QNN_PROFILE_LEVEL", "optrace", 1);
  setenv("MLLM_QNN_PROFILE_DIR", profile_dir.get().c_str(), 1);

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) {
    std::cerr << "QNN backend is unavailable\n";
    return 3;
  }

  if (numerical_fixture.get()) {
    std::vector<mllm::Tensor> inputs;
    inputs.push_back(loadU8Tensor({1, 1, seq.get(), kContext}, scores_file.get()));
    inputs.push_back(loadU8Tensor({1, 1, seq.get(), kContext}, mask_file.get()));
    std::vector<mllm::Tensor> outputs;
    outputs.push_back(mllm::Tensor::empty({1, 1, seq.get(), kContext}, mllm::kUInt8, mllm::kQNN).alloc());
    backend->graphExecute("model.0.s1", inputs, outputs);
    saveU8Tensor(outputs.front(), output_file.get());
    std::cout << "completed numerical fixture seq=" << seq.get() << " bytes=" << outputs.front().numel() << '\n';
    return 0;
  }

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 2 * kKvHeads + 1);
  for (int32_t head = 0; head < kHeads; ++head) { inputs.push_back(makeZeroU8Tensor({1, 1, seq.get(), kHeadDim})); }
  for (int32_t head = 0; head < kKvHeads; ++head) { inputs.push_back(makeZeroU8Tensor({1, 1, kHeadDim, kContext})); }
  for (int32_t head = 0; head < kKvHeads; ++head) { inputs.push_back(makeZeroU8Tensor({1, 1, kContext, kHeadDim})); }
  inputs.push_back(makeZeroU8Tensor({1, 1, seq.get(), kContext}));

  std::vector<mllm::Tensor> outputs;
  outputs.reserve(kHeads);
  for (int32_t head = 0; head < kHeads; ++head) {
    outputs.push_back(mllm::Tensor::empty({1, 1, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
  }

  for (int32_t iteration = 0; iteration < iterations.get(); ++iteration) {
    backend->graphExecute("model.0.s1", inputs, outputs);
  }
  std::cout << "completed executions=" << iterations.get() << " seq=" << seq.get() << '\n';
  return 0;
});
