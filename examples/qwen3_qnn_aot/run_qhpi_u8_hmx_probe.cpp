// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Argparse;

namespace {

constexpr uint32_t kSpatial = 64;
constexpr uint32_t kChannels = 32;
constexpr uint32_t kPhaseWordBytes = sizeof(uint32_t);

size_t packedWeightOffset(uint32_t input_channel, uint32_t output_channel) {
  return (static_cast<size_t>(input_channel / 4) * kChannels + output_channel) * 4 + input_channel % 4;
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("--context").help("Cached QNN context.");
  auto& graph_name = Argparse::add<std::string>("--graph").help("QNN graph name.");
  auto& profile_dir = Argparse::add<std::string>("--profile_dir").help("Profiling output directory.");
  auto& iterations = Argparse::add<int>("--iterations").help("Measured executions.").def(1);
  auto& profile_level = Argparse::add<std::string>("--profile_level").help("off or optrace.").def("optrace");
  auto& pipeline = Argparse::add<std::string>("--pipeline").help("single or mixed-resource.").def("single");
  auto& pattern = Argparse::add<std::string>("--pattern")
                      .help("structured, identity, permuted-signed, or map-weight diagnostic inputs.")
                      .def("structured");
  auto& map_offset = Argparse::add<int>("--map_offset").help("Single packed-weight byte to dump in map-weight mode.").def(-1);
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!context_path.isSet() || !graph_name.isSet() || !profile_dir.isSet() || iterations.get() <= 0
      || (profile_level.get() != "off" && profile_level.get() != "optrace")
      || (pipeline.get() != "single" && pipeline.get() != "mixed-resource")
      || (pattern.get() != "structured" && pattern.get() != "identity" && pattern.get() != "permuted-signed"
          && pattern.get() != "map-weight")
      || map_offset.get() < -1 || map_offset.get() >= static_cast<int>(kChannels * kChannels)
      || (pipeline.get() == "mixed-resource" && pattern.get() == "map-weight")) {
    Argparse::printHelp();
    return 2;
  }

  const auto absolute_profile_dir = std::filesystem::absolute(profile_dir.get());
  std::filesystem::create_directories(absolute_profile_dir);
  setenv("MLLM_QNN_PROFILE_LEVEL", profile_level.get().c_str(), 1);
  setenv("MLLM_QNN_PROFILE_DIR", absolute_profile_dir.c_str(), 1);

  mllm::initQnnBackend(context_path.get());
  auto backend = std::static_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
  if (!backend) throw std::runtime_error("QNN backend is unavailable");

  auto activation = mllm::Tensor::empty({1, 8, 8, 32}, mllm::kUInt8, mllm::kQNN).alloc();
  auto weight = mllm::Tensor::empty({1, 1, 32, 32}, mllm::kInt8, mllm::kQNN).alloc();
  auto output = mllm::Tensor::empty({1, 8, 8, 32}, mllm::kUInt8, mllm::kQNN).alloc();
  activation.setName("activation");
  weight.setName("weight_packed");
  output.setName("output");

  constexpr int32_t kInputZeroPoint = 128;
  for (uint32_t spatial = 0; spatial < kSpatial; ++spatial) {
    for (uint32_t input_channel = 0; input_channel < kChannels; ++input_channel) {
      const int32_t centered_value =
          pattern.get() == "map-weight"        ? static_cast<int32_t>(input_channel) + 1 - kInputZeroPoint
          : pattern.get() == "identity"        ? static_cast<int32_t>((spatial * 11 + input_channel * 7) % 64)
          : pattern.get() == "permuted-signed" ? static_cast<int32_t>((spatial * 11 + input_channel * 7) % 64) - 32
                                               : static_cast<int32_t>((spatial * 3 + input_channel * 5) % 4);
      activation.ptr<uint8_t>()[spatial * kChannels + input_channel] = static_cast<uint8_t>(kInputZeroPoint + centered_value);
    }
  }
  std::fill(weight.ptr<int8_t>(), weight.ptr<int8_t>() + kChannels * kChannels, 0);
  if (pattern.get() != "map-weight") {
    for (uint32_t input_channel = 0; input_channel < kChannels; ++input_channel) {
      for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
        int8_t coefficient = 0;
        if (pattern.get() == "identity") {
          coefficient = static_cast<int8_t>(input_channel == output_channel);
        } else if (pattern.get() == "permuted-signed") {
          const uint32_t selected_input = (output_channel * 7 + 3) % kChannels;
          coefficient = input_channel == selected_input ? static_cast<int8_t>(output_channel % 2 == 0 ? 1 : -1) : 0;
        } else {
          coefficient = static_cast<int8_t>((input_channel * 7 + output_channel * 3) % 4 - 1);
        }
        weight.ptr<int8_t>()[packedWeightOffset(input_channel, output_channel)] = coefficient;
      }
    }
    // EXP-0015 Stage A uses the final packed word as a TCM-only phase word.
    // Keep its four logical coefficients at zero; the device sees 0x80808080
    // after QNN applies the S8 tensor's physical offset.
    std::fill(weight.ptr<int8_t>() + kChannels * kChannels - kPhaseWordBytes,
              weight.ptr<int8_t>() + kChannels * kChannels, 0);
  }

  std::vector<mllm::Tensor> inputs{activation, weight};
  std::vector<mllm::Tensor> outputs{output};
  if (pattern.get() == "map-weight") {
    size_t invalid_mappings = 0;
    const size_t first_offset = map_offset.get() >= 0 ? static_cast<size_t>(map_offset.get()) : 0;
    const size_t last_offset = map_offset.get() >= 0 ? first_offset + 1 : kChannels * kChannels;
    for (size_t offset = first_offset; offset < last_offset; ++offset) {
      std::fill(weight.ptr<int8_t>(), weight.ptr<int8_t>() + kChannels * kChannels, 0);
      weight.ptr<int8_t>()[offset] = 1;
      backend->graphExecute(graph_name.get(), inputs, outputs);

      int32_t mapped_input = -1;
      int32_t mapped_output = -1;
      size_t nonzero_channels = 0;
      bool spatially_uniform = true;
      for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
        const uint8_t value = output.ptr<uint8_t>()[output_channel];
        if (value != 0) {
          mapped_input = static_cast<int32_t>(value) - 1;
          mapped_output = static_cast<int32_t>(output_channel);
          ++nonzero_channels;
        }
        for (uint32_t spatial = 1; spatial < kSpatial; ++spatial) {
          spatially_uniform &= output.ptr<uint8_t>()[spatial * kChannels + output_channel] == value;
        }
      }
      if (map_offset.get() >= 0) {
        for (uint32_t spatial = 0; spatial < kSpatial; ++spatial) {
          for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
            const uint8_t value = output.ptr<uint8_t>()[spatial * kChannels + output_channel];
            if (value != 0) {
              std::cout << "weight_value," << offset << ',' << spatial << ',' << output_channel << ','
                        << static_cast<int32_t>(value) << '\n';
            }
          }
        }
      }
      const bool valid =
          nonzero_channels == 1 && spatially_uniform && mapped_input >= 0 && mapped_input < static_cast<int32_t>(kChannels);
      invalid_mappings += !valid;
      std::cout << "weight_map," << offset << ',' << mapped_input << ',' << mapped_output << ',' << nonzero_channels << ','
                << static_cast<int32_t>(spatially_uniform) << '\n';
    }
    std::cout << "pattern=map-weight invalid_mappings=" << invalid_mappings << '\n';
    return invalid_mappings == 0 ? 0 : 1;
  }

  for (int invocation = 0; invocation < iterations.get(); ++invocation) {
    backend->graphExecute(graph_name.get(), inputs, outputs);
  }

  std::vector<uint8_t> first_stage(kSpatial * kChannels);
  std::vector<uint8_t> hvx_stage(kSpatial * kChannels);
  std::vector<uint8_t> reference(kSpatial * kChannels);
  for (uint32_t spatial = 0; spatial < kSpatial; ++spatial) {
    for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
      int32_t accumulator = 0;
      for (uint32_t input_channel = 0; input_channel < kChannels; ++input_channel) {
        const int32_t value = activation.ptr<uint8_t>()[spatial * kChannels + input_channel];
        const int32_t coefficient = weight.ptr<int8_t>()[packedWeightOffset(input_channel, output_channel)];
        accumulator += (value - kInputZeroPoint) * coefficient;
      }
      const size_t index = spatial * kChannels + output_channel;
      first_stage[index] = static_cast<uint8_t>(std::clamp(accumulator, 0, 255));
      hvx_stage[index] = static_cast<uint8_t>(std::min(static_cast<int32_t>(first_stage[index]) + 1, 255));
    }
  }
  if (pipeline.get() == "mixed-resource") {
    for (uint32_t spatial = 0; spatial < kSpatial; ++spatial) {
      for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
        int32_t accumulator = 0;
        for (uint32_t input_channel = 0; input_channel < kChannels; ++input_channel) {
          const int32_t value = hvx_stage[spatial * kChannels + input_channel];
          const int32_t coefficient = weight.ptr<int8_t>()[packedWeightOffset(input_channel, output_channel)];
          accumulator += value * coefficient;
        }
        reference[spatial * kChannels + output_channel] = static_cast<uint8_t>(std::clamp(accumulator, 0, 255));
      }
    }
  } else {
    reference = first_stage;
  }

  size_t mismatches = 0;
  int32_t minimum_reference = 255;
  int32_t maximum_reference = 0;
  int32_t minimum_output = 255;
  int32_t maximum_output = 0;
  uint64_t output_sum = 0;
  std::vector<std::pair<int32_t, int32_t>> first_pairs;
  for (uint32_t spatial = 0; spatial < kSpatial; ++spatial) {
    for (uint32_t output_channel = 0; output_channel < kChannels; ++output_channel) {
      const size_t index = spatial * kChannels + output_channel;
      const auto expected = reference[index];
      const auto actual = output.ptr<uint8_t>()[index];
      minimum_reference = std::min(minimum_reference, static_cast<int32_t>(expected));
      maximum_reference = std::max(maximum_reference, static_cast<int32_t>(expected));
      minimum_output = std::min(minimum_output, static_cast<int32_t>(actual));
      maximum_output = std::max(maximum_output, static_cast<int32_t>(actual));
      output_sum += actual;
      if (first_pairs.size() < 64) first_pairs.emplace_back(actual, expected);
      if (actual != expected) ++mismatches;
    }
  }
  std::cout << "pipeline=" << pipeline.get() << " pattern=" << pattern.get() << " iterations=" << iterations.get()
            << " output_first=" << static_cast<int32_t>(output.ptr<uint8_t>()[0]) << " output_min=" << minimum_output
            << " output_max=" << maximum_output << " output_sum=" << output_sum << " reference_min=" << minimum_reference
            << " reference_max=" << maximum_reference << " mismatches=" << mismatches << '\n';
  std::cout << "first_actual_reference=";
  for (const auto& [actual, reference] : first_pairs) { std::cout << actual << ':' << reference << ','; }
  std::cout << '\n';
  if (pipeline.get() == "mixed-resource") {
    std::array<size_t, 16> resource_markers{};
    for (size_t index = 0; index < kSpatial * kChannels; ++index) {
      const uint8_t value = output.ptr<uint8_t>()[index];
      if (value >= 240) ++resource_markers[value - 240];
    }
    std::cout << "resource_marker_counts=";
    for (size_t marker = 0; marker < resource_markers.size(); ++marker) {
      std::cout << (240 + marker) << ':' << resource_markers[marker] << ',';
    }
    std::cout << '\n';
  }
  return mismatches == 0 ? 0 : 1;
});
