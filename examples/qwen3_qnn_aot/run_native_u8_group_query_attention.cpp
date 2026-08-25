// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0017 device runner. Inputs are deterministic real values quantized with
// the accepted layer-14 contracts so A8, native GQA, and W4A16 diagnostics use
// the same mathematical tensors.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Argparse;

namespace {

constexpr int32_t kQueryHeads = 16;
constexpr int32_t kKvHeads = 8;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kSequence = 32;
constexpr int32_t kPast = 992;
constexpr int32_t kContext = 1024;

struct QParams {
  double scale;
  int32_t zero_point;
};

struct Contract {
  QParams query;
  QParams key;
  QParams value;
  QParams output;
};

Contract contract(bool a16) {
  if (a16) {
    return {{0.0011854973854497075, 32133}, {0.32952550053596497, 128},
            {1.407699704170227, 128}, {0.0021023168228566647, 58560}};
  }
  return {{0.32302287220954895, 122}, {0.33071592450141907, 128},
          {1.4026927947998047, 128}, {0.5333649516105652, 229}};
}

uint8_t quantizeU8(double value, QParams q) {
  return static_cast<uint8_t>(std::clamp<int64_t>(std::llround(value / q.scale) + q.zero_point, 0, 255));
}

uint16_t quantizeU16(double value, QParams q) {
  return static_cast<uint16_t>(
      std::clamp<int64_t>(std::llround(value / q.scale) + q.zero_point, 0, 65535));
}

double queryReal(int32_t head, int32_t row, int32_t channel) {
  const int32_t centered = (head * 11 + row * 5 + channel * 7) % 17 - 8;
  return centered * 0.32302287220954895;
}

double keyReal(int32_t head, int32_t depth, int32_t channel) {
  const int32_t centered = (head * 13 + channel * 3 + depth * 5 + (depth / 31) * 7) % 15 - 7;
  return centered * 0.33071592450141907;
}

double valueReal(int32_t head, int32_t depth, int32_t channel) {
  const int32_t centered = (head * 5 + depth * 7 + channel * 11 + (depth / 17) * 3) % 17 - 8;
  return centered * 1.4026927947998047;
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
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? static_cast<double>(values[middle])
                           : (static_cast<double>(values[middle - 1]) + values[middle]) * 0.5;
}

void writeExact(const std::filesystem::path& path, const void* data, size_t bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) throw std::runtime_error("cannot open output: " + path.string());
  stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to write output: " + path.string());
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name = Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& variant = Argparse::add<std::string>("--variant").help("native_gqa or decomposed.");
  auto& activation = Argparse::add<std::string>("--activation").help("a8 or a16.").def("a8");
  auto& timing_path = Argparse::add<std::string>("--timing_csv").help("Classified timing CSV.");
  auto& output_prefix = Argparse::add<std::string>("--output_prefix").help("Raw output prefix.");
  auto& warmup = Argparse::add<int>("--warmup").help("Warmup executions.").def(5);
  auto& iterations = Argparse::add<int>("--iterations").help("Measured executions.").def(30);
  auto& profile_level = Argparse::add<std::string>("--profile_level").help("off or optrace.").def("off");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !variant.isSet() || !timing_path.isSet()
      || !output_prefix.isSet() || (variant.get() != "native_gqa" && variant.get() != "decomposed")
      || (activation.get() != "a8" && activation.get() != "a16")
      || (variant.get() == "native_gqa" && activation.get() != "a8") || warmup.get() < 0
      || iterations.get() <= 0
      || (profile_level.get() != "off" && profile_level.get() != "optrace")) {
    Argparse::printHelp();
    return 2;
  }

  const bool native_gqa = variant.get() == "native_gqa";
  const bool a16 = activation.get() == "a16";
  const auto q = contract(a16);
  const auto timing_file = std::filesystem::absolute(timing_path.get());
  const auto profile_dir = timing_file.parent_path();
  const auto macro_file = profile_dir / "qnn_macro_profile.csv";
  std::filesystem::create_directories(profile_dir);
  setenv("MLLM_QNN_PROFILE_LEVEL", profile_level.get().c_str(), 1);
  setenv("MLLM_QNN_PROFILE_DIR", profile_dir.c_str(), 1);
  std::ofstream(macro_file, std::ios::trunc)
      << "graph,execution,profiled,captured,graph_execute_us\n";

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(
      mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) throw std::runtime_error("QNN backend is unavailable");

  const auto activation_storage = a16 ? mllm::kUInt16 : mllm::kUInt8;
  std::vector<mllm::Tensor> inputs;
  inputs.reserve(native_gqa ? 36 : 35);
  for (int32_t head = 0; head < kQueryHeads; ++head) {
    auto tensor = mllm::Tensor::empty({1, 1, kSequence, kHeadDim}, activation_storage, mllm::kQNN)
                      .alloc().setName("query_head_" + std::to_string(head));
    for (int32_t row = 0; row < kSequence; ++row) {
      for (int32_t channel = 0; channel < kHeadDim; ++channel) {
        const size_t index = static_cast<size_t>(row) * kHeadDim + channel;
        if (a16) {
          tensor.ptr<uint16_t>()[index] = quantizeU16(queryReal(head, row, channel), q.query);
        } else {
          tensor.ptr<uint8_t>()[index] = quantizeU8(queryReal(head, row, channel), q.query);
        }
      }
    }
    inputs.push_back(tensor);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto tensor = mllm::Tensor::empty({1, 1, kSequence, kHeadDim}, mllm::kUInt8, mllm::kQNN)
                      .alloc().setName("key_head_" + std::to_string(head));
    for (int32_t row = 0; row < kSequence; ++row) {
      for (int32_t channel = 0; channel < kHeadDim; ++channel) {
        tensor.ptr<uint8_t>()[row * kHeadDim + channel] =
            quantizeU8(keyReal(head, kPast + row, channel), q.key);
      }
    }
    inputs.push_back(tensor);
  }
  for (int32_t head = 0; head < kKvHeads; ++head) {
    auto tensor = mllm::Tensor::empty({1, 1, kSequence, kHeadDim}, mllm::kUInt8, mllm::kQNN)
                      .alloc().setName("value_head_" + std::to_string(head));
    for (int32_t row = 0; row < kSequence; ++row) {
      for (int32_t channel = 0; channel < kHeadDim; ++channel) {
        tensor.ptr<uint8_t>()[row * kHeadDim + channel] =
            quantizeU8(valueReal(head, kPast + row, channel), q.value);
      }
    }
    inputs.push_back(tensor);
  }
  auto past_key = mllm::Tensor::empty({1, kKvHeads, kHeadDim, kPast}, mllm::kUInt8, mllm::kQNN)
                      .alloc().setName("past_key");
  for (int32_t head = 0; head < kKvHeads; ++head) {
    for (int32_t channel = 0; channel < kHeadDim; ++channel) {
      for (int32_t depth = 0; depth < kPast; ++depth) {
        const size_t index = (static_cast<size_t>(head) * kHeadDim + channel) * kPast + depth;
        past_key.ptr<uint8_t>()[index] = quantizeU8(keyReal(head, depth, channel), q.key);
      }
    }
  }
  inputs.push_back(past_key);
  auto past_value = mllm::Tensor::empty({1, kKvHeads, kPast, kHeadDim}, mllm::kUInt8, mllm::kQNN)
                        .alloc().setName("past_value");
  for (int32_t head = 0; head < kKvHeads; ++head) {
    for (int32_t depth = 0; depth < kPast; ++depth) {
      for (int32_t channel = 0; channel < kHeadDim; ++channel) {
        const size_t index = (static_cast<size_t>(head) * kPast + depth) * kHeadDim + channel;
        past_value.ptr<uint8_t>()[index] = quantizeU8(valueReal(head, depth, channel), q.value);
      }
    }
  }
  inputs.push_back(past_value);

  if (native_gqa) {
    auto seqlens = mllm::Tensor::empty({1}, mllm::kInt32, mllm::kQNN).alloc().setName("seqlens_k_minus_one");
    seqlens.ptr<int32_t>()[0] = kContext - 1;
    inputs.push_back(seqlens);
    auto total = mllm::Tensor::empty({}, mllm::kInt32, mllm::kQNN).alloc().setName("total_sequence_length");
    total.ptr<int32_t>()[0] = kContext;
    inputs.push_back(total);
  } else {
    auto mask = mllm::Tensor::empty({1, 1, kSequence, kContext}, activation_storage, mllm::kQNN)
                    .alloc().setName("causal_mask");
    for (int32_t row = 0; row < kSequence; ++row) {
      const int32_t valid = kPast + row + 1;
      for (int32_t depth = 0; depth < kContext; ++depth) {
        const size_t index = static_cast<size_t>(row) * kContext + depth;
        if (a16) {
          mask.ptr<uint16_t>()[index] = depth < valid ? 65535 : 0;
        } else {
          mask.ptr<uint8_t>()[index] = depth < valid ? 255 : 0;
        }
      }
    }
    inputs.push_back(mask);
  }

  auto attention = mllm::Tensor::empty({1, kSequence, kQueryHeads * kHeadDim},
                                       activation_storage, mllm::kQNN).alloc().setName("attention_output");
  auto new_key = mllm::Tensor::empty({1, kKvHeads, kHeadDim, kSequence},
                                     mllm::kUInt8, mllm::kQNN).alloc().setName("new_key");
  auto new_value = mllm::Tensor::empty({1, kKvHeads, kSequence, kHeadDim},
                                       mllm::kUInt8, mllm::kQNN).alloc().setName("new_value");
  std::vector<mllm::Tensor> outputs{attention, new_key, new_value};

  const int total_invocations = warmup.get() + iterations.get();
  for (int invocation = 0; invocation < total_invocations; ++invocation) {
    backend->graphExecute(graph_name.get(), inputs, outputs);
  }

  const auto durations = readMacroDurations(macro_file);
  if (durations.size() != static_cast<size_t>(total_invocations)) {
    throw std::runtime_error("macro timer row count mismatch");
  }
  std::vector<uint64_t> measured;
  std::ofstream timing(timing_file, std::ios::trunc);
  timing << "phase,iteration,graph_execute_us\n";
  for (size_t index = 0; index < durations.size(); ++index) {
    const bool is_warmup = index < static_cast<size_t>(warmup.get());
    timing << (is_warmup ? "warmup" : "measured") << ','
           << (is_warmup ? index : index - warmup.get()) << ',' << durations[index] << '\n';
    if (!is_warmup) measured.push_back(durations[index]);
  }

  const auto prefix = std::filesystem::absolute(output_prefix.get());
  writeExact(prefix.string() + (a16 ? ".attention.u16" : ".attention.u8"),
             a16 ? static_cast<const void*>(outputs[0].ptr<uint16_t>())
                 : static_cast<const void*>(outputs[0].ptr<uint8_t>()),
             static_cast<size_t>(kSequence) * kQueryHeads * kHeadDim * (a16 ? 2 : 1));
  writeExact(prefix.string() + ".key.u8", outputs[1].ptr<uint8_t>(),
             static_cast<size_t>(kKvHeads) * kHeadDim * kSequence);
  writeExact(prefix.string() + ".value.u8", outputs[2].ptr<uint8_t>(),
             static_cast<size_t>(kKvHeads) * kSequence * kHeadDim);

  const auto [minimum, maximum] = std::minmax_element(measured.begin(), measured.end());
  const double mean = std::accumulate(measured.begin(), measured.end(), 0.0)
                      / static_cast<double>(measured.size());
  std::cout << "experiment=EXP-0017 variant=" << variant.get() << " activation=" << activation.get()
            << " warmup=" << warmup.get() << " iterations=" << iterations.get()
            << " wall_us_min=" << *minimum << " wall_us_median=" << median(measured)
            << " wall_us_mean=" << mean << " wall_us_max=" << *maximum << '\n';
  return 0;
});
