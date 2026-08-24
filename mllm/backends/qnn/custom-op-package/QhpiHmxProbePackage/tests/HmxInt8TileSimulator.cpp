// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdio>

#include "HmxInt8Tile.hpp"

namespace probe = mllm::qnn::qhpi_hmx_probe;

alignas(2048) static uint8_t activation[probe::kActivationBytes];
alignas(2048) static int8_t weight[2048];
alignas(256) static uint32_t bias[probe::kBiasBytes / sizeof(uint32_t)];
alignas(2048) static uint8_t output[probe::kOutputBytes];

int main() {
  for (auto& value : activation) value = 1;
  for (auto& value : weight) value = 1;
  for (auto& value : output) value = 0xa5;

  probe::fillAsymmetricBias(weight, 0, bias);
  probe::executeU8S8Tile(activation, weight, bias, output);

  size_t changed = 0;
  uint8_t minimum = 255;
  uint8_t maximum = 0;
  uint64_t sum = 0;
  for (const uint8_t value : output) {
    changed += value != 0xa5;
    minimum = value < minimum ? value : minimum;
    maximum = value > maximum ? value : maximum;
    sum += value;
  }
  std::printf("changed=%u min=%u max=%u sum=%llu first=%u\n", static_cast<unsigned>(changed), static_cast<unsigned>(minimum),
              static_cast<unsigned>(maximum), static_cast<unsigned long long>(sum), static_cast<unsigned>(output[0]));
  return changed == probe::kOutputBytes ? 0 : 1;
}
