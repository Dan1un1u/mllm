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
constexpr int32_t kMaximumContext = 1024;
constexpr int32_t kPairHeads = 2;
constexpr int32_t kPairs = kHeads / kPairHeads;

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

mllm::Tensor loadInt32Tensor(const std::vector<int32_t>& shape, const std::string& path) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kInt32, mllm::kQNN).alloc();
  const auto expected_bytes = tensor.numel() * sizeof(int32_t);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(tensor.ptr<int32_t>()), expected_bytes) || input.peek() != EOF) {
    throw std::runtime_error("Expected exactly " + std::to_string(expected_bytes) + " bytes in " + path);
  }
  return tensor;
}

mllm::Tensor makePositionTensor(int32_t seq) {
  auto tensor = mllm::Tensor::empty({seq}, mllm::kInt32, mllm::kQNN).alloc();
  for (int32_t row = 0; row < seq; ++row) { tensor.ptr<int32_t>()[row] = row; }
  return tensor;
}

void saveU8Tensor(const mllm::Tensor& tensor, const std::string& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>()), tensor.numel());
  if (!output.good()) { throw std::runtime_error("Failed to write " + path); }
}
}  // namespace

MLLM_MAIN({
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& seq = Argparse::add<int>("--seq").help("Sequence length: 1 or 32.");
  auto& kv_context =
      Argparse::add<int>("--kv_context").help("KV context width, a multiple of 32 up to 1024.").def(kMaximumContext);
  auto& profile_dir = Argparse::add<std::string>("--profile_dir").help("Optrace directory.");
  auto& iterations = Argparse::add<int>("--iterations").help("Execution count.").def(1);
  auto& numerical_fixture = Argparse::add<bool>("--numerical_fixture").help("Run numerical fixture.").def(false);
  auto& packed_heads = Argparse::add<bool>("--packed_heads").help("Run the batched 16-head graph.").def(false);
  auto& grouped_heads = Argparse::add<bool>("--grouped_heads").help("Run four natural 4-query-head GQA chains.").def(false);
  auto& paired_heads =
      Argparse::add<bool>("--paired_heads").help("Run eight 2-query-head chains within GQA groups.").def(false);
  auto& joined_output =
      Argparse::add<bool>("--joined_output").help("Run the split-head graph with one joined output.").def(false);
  auto& scores_file = Argparse::add<std::string>("--scores_file").help("Raw U8 scores.");
  auto& positions_file = Argparse::add<std::string>("--positions_file").help("Raw Int32 position IDs.");
  auto& output_file = Argparse::add<std::string>("--output_file").help("Raw U8 output.");
  Argparse::parse(argc, argv);
  if (!context_path.isSet() || !seq.isSet() || !profile_dir.isSet() || (seq.get() != 1 && seq.get() != 32)
      || iterations.get() <= 0) {
    Argparse::printHelp();
    return 2;
  }
  if (kv_context.get() < 32 || kv_context.get() > kMaximumContext || kv_context.get() % 32 != 0
      || kv_context.get() < seq.get()) {
    Argparse::printHelp();
    return 2;
  }
  if (static_cast<int32_t>(packed_heads.get()) + static_cast<int32_t>(grouped_heads.get())
          + static_cast<int32_t>(paired_heads.get())
      > 1) {
    Argparse::printHelp();
    return 2;
  }
  if (joined_output.get() && (packed_heads.get() || grouped_heads.get() || paired_heads.get())) {
    Argparse::printHelp();
    return 2;
  }
  if (numerical_fixture.get() && (!scores_file.isSet() || !positions_file.isSet() || !output_file.isSet())) {
    Argparse::printHelp();
    return 2;
  }

  std::filesystem::create_directories(profile_dir.get());
  setenv("MLLM_QNN_PROFILE_LEVEL", "optrace", 1);
  setenv("MLLM_QNN_PROFILE_DIR", profile_dir.get().c_str(), 1);
  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) { return 3; }

  if (numerical_fixture.get()) {
    const int32_t heads =
        packed_heads.get() ? kHeads : (grouped_heads.get() ? kHeads / kKvHeads : (paired_heads.get() ? kPairHeads : 1));
    std::vector<mllm::Tensor> inputs{
        loadU8Tensor({1, heads, seq.get(), kv_context.get()}, scores_file.get()),
        loadInt32Tensor({1, 1, seq.get(), 1}, positions_file.get()),
    };
    std::vector<int32_t> output_shape{1, heads, seq.get(), kv_context.get()};
    if (packed_heads.get() && seq.get() == 1) {
      output_shape = {1, heads / (kHeads / kKvHeads), kHeads / kKvHeads, kv_context.get()};
    } else if ((grouped_heads.get() || paired_heads.get()) && seq.get() == 1) {
      output_shape = {1, 1, heads, kv_context.get()};
    } else if (!packed_heads.get() && !grouped_heads.get() && !paired_heads.get() && seq.get() == 32) {
      output_shape = {1, 8, 4, kv_context.get()};
    }
    std::vector<mllm::Tensor> outputs{
        mllm::Tensor::empty(output_shape, mllm::kUInt8, mllm::kQNN).alloc(),
    };
    backend->graphExecute(packed_heads.get()
                              ? "model.0.s16"
                              : (grouped_heads.get() ? "model.0.s4" : (paired_heads.get() ? "model.0.s2" : "model.0.s1")),
                          inputs, outputs);
    saveU8Tensor(outputs.front(), output_file.get());
    std::cout << "completed numerical fixture seq=" << seq.get() << '\n';
    return 0;
  }

  if (paired_heads.get()) {
    std::vector<mllm::Tensor> inputs;
    inputs.reserve(kPairs + 2 * kKvHeads + 1);
    for (int32_t pair = 0; pair < kPairs; ++pair) { inputs.push_back(makeZeroU8Tensor({1, kPairHeads, seq.get(), kHeadDim})); }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      inputs.push_back(makeZeroU8Tensor({1, 1, kHeadDim, kv_context.get()}));
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      inputs.push_back(makeZeroU8Tensor({1, 1, kv_context.get(), kHeadDim}));
    }
    inputs.push_back(positions_file.isSet() ? loadInt32Tensor({seq.get()}, positions_file.get())
                                            : makePositionTensor(seq.get()));
    std::vector<mllm::Tensor> outputs;
    for (int32_t pair = 0; pair < kPairs; ++pair) {
      outputs.push_back(mllm::Tensor::empty({1, kPairHeads, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
    }
    for (int32_t i = 0; i < iterations.get(); ++i) { backend->graphExecute("model.0.s2", inputs, outputs); }
    std::cout << "completed paired-head executions=" << iterations.get() << " seq=" << seq.get() << '\n';
    return 0;
  }

  if (grouped_heads.get()) {
    std::vector<mllm::Tensor> inputs;
    inputs.reserve(3 * kKvHeads + 1);
    for (int32_t group = 0; group < kKvHeads; ++group) {
      inputs.push_back(makeZeroU8Tensor({1, kHeads / kKvHeads, seq.get(), kHeadDim}));
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      inputs.push_back(makeZeroU8Tensor({1, 1, kHeadDim, kv_context.get()}));
    }
    for (int32_t group = 0; group < kKvHeads; ++group) {
      inputs.push_back(makeZeroU8Tensor({1, 1, kv_context.get(), kHeadDim}));
    }
    inputs.push_back(positions_file.isSet() ? loadInt32Tensor({seq.get()}, positions_file.get())
                                            : makePositionTensor(seq.get()));
    std::vector<mllm::Tensor> outputs;
    for (int32_t group = 0; group < kKvHeads; ++group) {
      outputs.push_back(mllm::Tensor::empty({1, kHeads / kKvHeads, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
    }
    for (int32_t i = 0; i < iterations.get(); ++i) { backend->graphExecute("model.0.s4", inputs, outputs); }
    std::cout << "completed grouped-head executions=" << iterations.get() << " seq=" << seq.get() << '\n';
    return 0;
  }

  if (packed_heads.get()) {
    std::vector<mllm::Tensor> inputs{
        makeZeroU8Tensor({1, kHeads, seq.get(), kHeadDim}),
        makeZeroU8Tensor({1, kKvHeads, kHeadDim, kv_context.get()}),
        makeZeroU8Tensor({1, kKvHeads, kv_context.get(), kHeadDim}),
        positions_file.isSet() ? loadInt32Tensor({seq.get()}, positions_file.get()) : makePositionTensor(seq.get()),
    };
    std::vector<mllm::Tensor> outputs{
        mllm::Tensor::empty({1, kHeads, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc(),
    };
    for (int32_t i = 0; i < iterations.get(); ++i) { backend->graphExecute("model.0.s16", inputs, outputs); }
    std::cout << "completed packed-head executions=" << iterations.get() << " seq=" << seq.get() << '\n';
    return 0;
  }

  std::vector<mllm::Tensor> inputs;
  inputs.reserve(kHeads + 2 * kKvHeads + 1);
  for (int32_t i = 0; i < kHeads; ++i) { inputs.push_back(makeZeroU8Tensor({1, 1, seq.get(), kHeadDim})); }
  for (int32_t i = 0; i < kKvHeads; ++i) { inputs.push_back(makeZeroU8Tensor({1, 1, kHeadDim, kv_context.get()})); }
  for (int32_t i = 0; i < kKvHeads; ++i) { inputs.push_back(makeZeroU8Tensor({1, 1, kv_context.get(), kHeadDim})); }
  inputs.push_back(positions_file.isSet() ? loadInt32Tensor({seq.get()}, positions_file.get()) : makePositionTensor(seq.get()));
  std::vector<mllm::Tensor> outputs;
  if (joined_output.get()) {
    outputs.push_back(mllm::Tensor::empty({1, kHeads, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
  } else {
    for (int32_t i = 0; i < kHeads; ++i) {
      outputs.push_back(mllm::Tensor::empty({1, 1, seq.get(), kHeadDim}, mllm::kUInt8, mllm::kQNN).alloc());
    }
  }
  for (int32_t i = 0; i < iterations.get(); ++i) { backend->graphExecute("model.0.s1", inputs, outputs); }
  std::cout << "completed executions=" << iterations.get() << " seq=" << seq.get() << '\n';
  return 0;
});
