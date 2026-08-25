// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <sys/mman.h>

#include "HmxInt8Tile.hpp"

namespace probe = mllm::qnn::qhpi_hmx_probe;

int main() {
  libnative_use_hmx_v1();
  auto* activation = static_cast<uint8_t*>(std::aligned_alloc(2048, probe::kActivationBytes));
  auto* weight = static_cast<int8_t*>(std::aligned_alloc(2048, 2048));
  auto* output =
      static_cast<uint8_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0));
  if (activation == nullptr || weight == nullptr || output == MAP_FAILED) return 2;

  std::memset(activation, 129, probe::kActivationBytes);
  std::memset(weight, 1, 2048);
  auto* bias = reinterpret_cast<uint32_t*>(output + probe::kWeightBytes);

  for (uint32_t exponent = 18; exponent <= 25; ++exponent) {
    for (uint32_t significand : {0u, 256u, 512u, 768u}) {
      std::memset(output, 0, 4096);
      const uint32_t lower_word = (exponent << 10) | significand;
      for (uint32_t channel = 0; channel < probe::kOutputChannels; ++channel) {
        bias[channel] = lower_word;
        bias[probe::kOutputChannels + channel] = static_cast<uint32_t>(-128 * 32);
      }
      probe::executeU8S8Tile(activation, weight, bias, output);
      std::cout << "exponent=" << exponent << " significand=" << significand
                << " lower=0x" << std::hex << lower_word << std::dec
                << " output=" << static_cast<unsigned>(output[0]) << '\n';
    }
  }

  // A four-tile deep load must accumulate into the same HMX state before one
  // converting store. Scale 1/2 and a pre-scale +256 bias encode
  // output_zero=128, so 4 * 32 centered MACs become 192.
  auto* deep_activation = static_cast<uint8_t*>(std::aligned_alloc(2048, 4 * probe::kActivationBytes));
  auto* deep_weight = static_cast<int8_t*>(std::aligned_alloc(2048, 4 * probe::kActivationBytes));
  if (deep_activation == nullptr || deep_weight == nullptr) return 2;
  std::memset(deep_activation, 129, 4 * probe::kActivationBytes);
  for (uint32_t weight_stride : {probe::kWeightBytes, probe::kActivationBytes}) {
    std::memset(deep_weight, 0, 4 * probe::kActivationBytes);
    for (uint32_t tile = 0; tile < 4; ++tile) std::memset(deep_weight + tile * weight_stride, 1, probe::kWeightBytes);
    for (uint32_t tiles : {1u, 2u, 4u}) {
      std::memset(output, 0, 4096);
      for (uint32_t channel = 0; channel < probe::kOutputChannels; ++channel) {
        bias[channel] = 23u << 10;
        bias[probe::kOutputChannels + channel] = static_cast<uint32_t>(-128 * 32 * tiles + 256);
      }
      Q6_bias_mxmem2_A(bias);
      Q6_mxclracc();
      Q6_activation_ub_mxmem_RR_deep_cm(reinterpret_cast<uintptr_t>(deep_activation),
                                        tiles * probe::kActivationBytes - 1);
      Q6_weight_b_mxmem_RR_deep(reinterpret_cast<uintptr_t>(deep_weight), tiles * weight_stride - 1);
      Q6_mxmem_AR_after_cm_sat_ub(output, probe::kWriteRt);
      std::cout << "deep_tiles=" << tiles << " weight_stride=" << weight_stride
                << " output=" << static_cast<unsigned>(output[0])
                << " expected=" << (128 + 16 * tiles) << '\n';
    }
  }

  std::free(deep_weight);
  std::free(deep_activation);
  munmap(output, 4096);
  std::free(weight);
  std::free(activation);
  return 0;
}
