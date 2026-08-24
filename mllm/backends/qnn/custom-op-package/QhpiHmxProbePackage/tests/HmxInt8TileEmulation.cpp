// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/mman.h>

#include "HmxInt8Tile.hpp"

namespace probe = mllm::qnn::qhpi_hmx_probe;

int main() {
  libnative_use_hmx_v1();
  auto* activation = static_cast<uint8_t*>(std::aligned_alloc(2048, probe::kActivationBytes));
  auto* weight = static_cast<int8_t*>(std::aligned_alloc(2048, 2048));
  // libnative models Hexagon's 32-bit address operands, so keep the host-side
  // output below 4 GiB. Real Hexagon pointers are natively 32-bit.
  auto* output =
      static_cast<uint8_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0));
  if (activation == nullptr || weight == nullptr || output == MAP_FAILED) {
    std::cerr << "aligned allocation failed\n";
    return 2;
  }

  constexpr int32_t kInputZeroPoint = 128;
  for (uint32_t spatial = 0; spatial < probe::kSpatial; ++spatial) {
    for (uint32_t input_channel = 0; input_channel < probe::kInputChannels; ++input_channel) {
      activation[probe::activationOffset(spatial, input_channel)] =
          static_cast<uint8_t>(kInputZeroPoint + ((spatial * 3 + input_channel * 5) % 4));
    }
  }
  std::memset(weight, 0, 2048);
  for (uint32_t input_channel = 0; input_channel < probe::kInputChannels; ++input_channel) {
    for (uint32_t output_channel = 0; output_channel < probe::kOutputChannels; ++output_channel) {
      weight[probe::weightOffset(input_channel, output_channel)] =
          static_cast<int8_t>((input_channel * 7 + output_channel * 3) % 4 - 1);
    }
  }

  // Match the QHPI kernel's zero-DDR scratch strategy: stage the signed
  // weight and bias in the output crouton, then overwrite it with the result.
  std::memcpy(output, weight, probe::kWeightBytes);
  auto* staged_weight = reinterpret_cast<int8_t*>(output);
  auto* staged_bias = reinterpret_cast<uint32_t*>(output + probe::kWeightBytes);
  probe::fillAsymmetricBias(staged_weight, kInputZeroPoint, staged_bias);
  probe::executeU8S8Tile(activation, staged_weight, staged_bias, output);

  size_t mismatches = 0;
  int32_t minimum_reference = 255;
  int32_t maximum_reference = 0;
  for (uint32_t spatial = 0; spatial < probe::kSpatial; ++spatial) {
    for (uint32_t output_channel = 0; output_channel < probe::kOutputChannels; ++output_channel) {
      int32_t accumulator = 0;
      for (uint32_t input_channel = 0; input_channel < probe::kInputChannels; ++input_channel) {
        const int32_t value = activation[probe::activationOffset(spatial, input_channel)];
        const int32_t coefficient = weight[probe::weightOffset(input_channel, output_channel)];
        accumulator += (value - kInputZeroPoint) * coefficient;
      }
      const auto reference = static_cast<uint8_t>(std::clamp(accumulator, 0, 255));
      minimum_reference = std::min(minimum_reference, static_cast<int32_t>(reference));
      maximum_reference = std::max(maximum_reference, static_cast<int32_t>(reference));
      if (output[probe::outputOffset(spatial, output_channel)] != reference) ++mismatches;
    }
  }

  std::cout << "hmx_integer_u8xs8=1 asymmetric_zero_point=" << kInputZeroPoint << " reference_min=" << minimum_reference
            << " reference_max=" << maximum_reference << " mismatches=" << mismatches << '\n';

  munmap(output, 4096);
  std::free(weight);
  std::free(activation);
  return mismatches == 0 ? 0 : 1;
}
