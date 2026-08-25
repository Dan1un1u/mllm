// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0016 Stage B device runner. It reports measured graph wall latency and
// compares the custom integer-HMX/HVX result with a float64 attention oracle.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

constexpr uint32_t kHeads = 2;
constexpr uint32_t kRows = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kContext = 1024;
constexpr int32_t kQueryZero = 122;
constexpr int32_t kKeyZero = 128;
constexpr int32_t kValueZero = 128;
constexpr int32_t kOutputZero = 229;
constexpr double kQueryScale = 0.32302287220954895;
constexpr double kKeyScale = 0.33071592450141907;
constexpr double kValueScale = 1.4026927947998047;
constexpr double kOutputScale = 0.5333649516105652;
constexpr int32_t kW4A16QueryZero = 32133;
constexpr int32_t kW4A16KeyZero = 128;
constexpr int32_t kW4A16ValueZero = 128;
constexpr int32_t kW4A16OutputZero = 58560;
constexpr double kW4A16QueryScale = 0.0011854973854497075;
constexpr double kW4A16KeyScale = 0.32952550053596497;
constexpr double kW4A16ValueScale = 1.407699704170227;
constexpr double kW4A16OutputScale = 0.0021023168228566647;

uint8_t clampU8(int32_t value) {
  return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

uint16_t clampU16(int64_t value) {
  return static_cast<uint16_t>(std::clamp<int64_t>(value, 0, 65535));
}

uint8_t quantizeU8(double value, double scale, int32_t zero_point) {
  return clampU8(static_cast<int32_t>(std::llround(value / scale)) + zero_point);
}

uint16_t quantizeU16(double value, double scale, int32_t zero_point) {
  return clampU16(std::llround(value / scale) + zero_point);
}

double queryReal(uint32_t head, uint32_t row, uint32_t channel) {
  const int32_t centered = static_cast<int32_t>((head * 11 + row * 5 + channel * 7) % 17) - 8;
  return centered * kQueryScale;
}

double keyReal(uint32_t channel, uint32_t depth) {
  const int32_t centered = static_cast<int32_t>((channel * 3 + depth * 5 + (depth / 31) * 7) % 15) - 7;
  return centered * kKeyScale;
}

double valueReal(uint32_t depth, uint32_t channel) {
  const int32_t centered = static_cast<int32_t>((depth * 7 + channel * 11 + (depth / 17) * 3) % 17) - 8;
  return centered * kValueScale;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name = Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& profile_dir = Argparse::add<std::string>("--profile_dir").help("Profiling output directory.");
  auto& warmup = Argparse::add<int>("--warmup").help("Warmup executions.").def(5);
  auto& iterations = Argparse::add<int>("--iterations").help("Measured executions.").def(30);
  auto& profile_level = Argparse::add<std::string>("--profile_level").help("off or optrace.").def("off");
  auto& activation_variant =
      Argparse::add<std::string>("--activation_variant").help("w4a8 or w4a16 graph I/O.").def("w4a8");
  auto& output_dump =
      Argparse::add<std::string>("--output_dump").help("Optional raw output tensor path.").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !profile_dir.isSet() || warmup.get() < 0 || iterations.get() <= 0
      || (profile_level.get() != "off" && profile_level.get() != "optrace")) {
    Argparse::printHelp();
    return 2;
  }
  if (activation_variant.get() != "w4a8" && activation_variant.get() != "w4a16") {
    Argparse::printHelp();
    return 2;
  }
  const bool w4a16 = activation_variant.get() == "w4a16";

  const auto absolute_profile_dir = std::filesystem::absolute(profile_dir.get());
  std::filesystem::create_directories(absolute_profile_dir);
  setenv("MLLM_QNN_PROFILE_LEVEL", profile_level.get().c_str(), 1);
  setenv("MLLM_QNN_PROFILE_DIR", absolute_profile_dir.c_str(), 1);

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) throw std::runtime_error("QNN backend is unavailable");

  const auto activation_storage_dtype = w4a16 ? mllm::kUInt16 : mllm::kUInt8;
  auto query = mllm::Tensor::empty({1, 2, 32, 128}, activation_storage_dtype, mllm::kQNN).alloc().setName("query");
  auto key = mllm::Tensor::empty({1, 1, 128, 1024}, mllm::kUInt8, mllm::kQNN).alloc().setName("key_transposed");
  auto value = mllm::Tensor::empty({1, 1, 1024, 128}, mllm::kUInt8, mllm::kQNN).alloc().setName("value");
  auto mask = mllm::Tensor::empty({1, 1, 32, 1024}, activation_storage_dtype, mllm::kQNN).alloc().setName("causal_mask");
  auto output = mllm::Tensor::empty({1, 2, 32, 128}, activation_storage_dtype, mllm::kQNN).alloc().setName("output");

  for (uint32_t head = 0; head < kHeads; ++head) {
    for (uint32_t row = 0; row < kRows; ++row) {
      for (uint32_t channel = 0; channel < kHeadDim; ++channel) {
        const size_t index = ((head * kRows + row) * kHeadDim) + channel;
        if (w4a16) {
          query.ptr<uint16_t>()[index] = quantizeU16(queryReal(head, row, channel), kW4A16QueryScale,
                                                    kW4A16QueryZero);
        } else {
          query.ptr<uint8_t>()[index] = quantizeU8(queryReal(head, row, channel), kQueryScale, kQueryZero);
        }
      }
    }
  }
  for (uint32_t channel = 0; channel < kHeadDim; ++channel) {
    for (uint32_t depth = 0; depth < kContext; ++depth) {
      key.ptr<uint8_t>()[channel * kContext + depth] =
          quantizeU8(keyReal(channel, depth), w4a16 ? kW4A16KeyScale : kKeyScale,
                     w4a16 ? kW4A16KeyZero : kKeyZero);
    }
  }
  for (uint32_t depth = 0; depth < kContext; ++depth) {
    for (uint32_t channel = 0; channel < kHeadDim; ++channel) {
      value.ptr<uint8_t>()[depth * kHeadDim + channel] =
          quantizeU8(valueReal(depth, channel), w4a16 ? kW4A16ValueScale : kValueScale,
                     w4a16 ? kW4A16ValueZero : kValueZero);
    }
  }
  for (uint32_t row = 0; row < kRows; ++row) {
    const uint32_t valid = 993 + row;
    for (uint32_t depth = 0; depth < kContext; ++depth) {
      if (w4a16) {
        mask.ptr<uint16_t>()[row * kContext + depth] = depth < valid ? 65535 : 0;
      } else {
        mask.ptr<uint8_t>()[row * kContext + depth] = depth < valid ? 255 : 0;
      }
    }
  }

  std::vector<mllm::Tensor> inputs{query, key, value, mask};
  std::vector<mllm::Tensor> outputs{output};
  for (int invocation = 0; invocation < warmup.get(); ++invocation) backend->graphExecute(graph_name.get(), inputs, outputs);

  std::vector<double> wall_microseconds;
  wall_microseconds.reserve(iterations.get());
  for (int invocation = 0; invocation < iterations.get(); ++invocation) {
    const auto start = std::chrono::steady_clock::now();
    backend->graphExecute(graph_name.get(), inputs, outputs);
    const auto end = std::chrono::steady_clock::now();
    wall_microseconds.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }

  std::vector<uint32_t> reference(kHeads * kRows * kHeadDim);
  std::vector<double> scores(kContext);
  std::vector<double> probabilities(kContext);
  for (uint32_t head = 0; head < kHeads; ++head) {
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint32_t valid = 993 + row;
      double maximum = -1.0e300;
      for (uint32_t depth = 0; depth < valid; ++depth) {
        double score = 0.0;
        for (uint32_t channel = 0; channel < kHeadDim; ++channel) {
          score += queryReal(head, row, channel) * keyReal(channel, depth);
        }
        scores[depth] = score / std::sqrt(static_cast<double>(kHeadDim));
        maximum = std::max(maximum, scores[depth]);
      }
      double denominator = 0.0;
      for (uint32_t depth = 0; depth < valid; ++depth) {
        probabilities[depth] = std::exp(scores[depth] - maximum);
        denominator += probabilities[depth];
      }
      for (uint32_t depth = 0; depth < valid; ++depth) probabilities[depth] /= denominator;
      for (uint32_t channel = 0; channel < kHeadDim; ++channel) {
        double result = 0.0;
        for (uint32_t depth = 0; depth < valid; ++depth) {
          result += probabilities[depth] * valueReal(depth, channel);
        }
        reference[(head * kRows + row) * kHeadDim + channel] = w4a16
            ? quantizeU16(result, kW4A16OutputScale, kW4A16OutputZero)
            : quantizeU8(result, kOutputScale, kOutputZero);
      }
    }
  }

  int32_t maximum_error = 0;
  uint64_t absolute_error_sum = 0;
  size_t errors_gt_two = 0;
  size_t output_saturated = 0;
  for (size_t index = 0; index < reference.size(); ++index) {
    const uint32_t actual = w4a16 ? output.ptr<uint16_t>()[index] : output.ptr<uint8_t>()[index];
    const int32_t error = std::abs(static_cast<int32_t>(actual) - static_cast<int32_t>(reference[index]));
    maximum_error = std::max(maximum_error, error);
    absolute_error_sum += static_cast<uint64_t>(error);
    errors_gt_two += error > 2;
    output_saturated += actual == 0 || actual == (w4a16 ? 65535u : 255u);
  }
  if (!output_dump.get().empty()) {
    std::ofstream stream(output_dump.get(), std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("unable to open output dump");
    const size_t output_bytes = reference.size() * (w4a16 ? sizeof(uint16_t) : sizeof(uint8_t));
    const auto* output_data = w4a16 ? reinterpret_cast<const char*>(output.ptr<uint16_t>())
                                    : reinterpret_cast<const char*>(output.ptr<uint8_t>());
    stream.write(output_data, static_cast<std::streamsize>(output_bytes));
    if (!stream) throw std::runtime_error("unable to write output dump");
  }
  const double mean_error = static_cast<double>(absolute_error_sum) / reference.size();
  const auto [minimum_wall, maximum_wall] = std::minmax_element(wall_microseconds.begin(), wall_microseconds.end());
  const double mean_wall = std::accumulate(wall_microseconds.begin(), wall_microseconds.end(), 0.0)
                           / static_cast<double>(wall_microseconds.size());
  std::cout << std::fixed << std::setprecision(3)
            << "experiment=EXP-0016 stage=B variant=" << activation_variant.get()
            << " warmup=" << warmup.get() << " iterations=" << iterations.get()
            << " wall_us_min=" << *minimum_wall << " wall_us_median=" << median(wall_microseconds)
            << " wall_us_mean=" << mean_wall << " wall_us_max=" << *maximum_wall
            << " reference_max_error_lsb=" << maximum_error << " reference_mean_error_lsb=" << mean_error
            << " reference_errors_gt_2=" << errors_gt_two << " output_saturated=" << output_saturated << '\n';
  return 0;
});
