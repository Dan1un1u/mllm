// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/mman.h>

#include "HmxGqaCore.hpp"

namespace gqa = mllm::qnn::qhpi_hmx_probe::gqa;
namespace probe = mllm::qnn::qhpi_hmx_probe;

namespace {

uint8_t requantize(int32_t accumulator, int32_t bias, float scale) {
  const int32_t scaled_accumulator = static_cast<int32_t>(std::floor(accumulator * scale));
  const int32_t scaled_bias = static_cast<int32_t>(std::floor(bias * scale));
  return static_cast<uint8_t>(std::clamp(scaled_accumulator + scaled_bias, 0, 255));
}

}  // namespace

int main() {
  libnative_use_hmx_v1();
  auto* query = static_cast<uint8_t*>(std::aligned_alloc(2048, gqa::kQkDepthTiles * gqa::kPhysicalTileBytes));
  auto* key = static_cast<uint8_t*>(std::aligned_alloc(2048, gqa::kQkDepthTiles * gqa::kPackedWeightBytes));
  auto* weight_scratch = static_cast<int8_t*>(std::aligned_alloc(2048, gqa::kQkDepthTiles * gqa::kPhysicalTileBytes));
  auto* output =
      static_cast<uint8_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0));
  auto* bias = static_cast<uint32_t*>(std::aligned_alloc(256, probe::kBiasBytes));
  if (query == nullptr || key == nullptr || weight_scratch == nullptr || output == MAP_FAILED || bias == nullptr) return 2;

  std::memset(query, gqa::kQueryZeroPoint, gqa::kQkDepthTiles * gqa::kPhysicalTileBytes);
  for (uint32_t tile = 0; tile < gqa::kQkDepthTiles; ++tile) {
    for (uint32_t row = 0; row < 32; ++row) {
      for (uint32_t channel = 0; channel < 32; ++channel) {
        query[tile * gqa::kPhysicalTileBytes + row * 32 + channel] =
            static_cast<uint8_t>(gqa::kQueryZeroPoint + ((row * 3 + channel * 5 + tile * 7) % 31) - 15);
        key[tile * gqa::kPackedWeightBytes + row * 32 + channel] =
            static_cast<uint8_t>(gqa::kKeyZeroPoint + ((row * 11 + channel * 7 + tile * 5) % 29) - 14);
      }
    }
    gqa::packSignedWeightTileHvx(key + tile * gqa::kPackedWeightBytes,
                                weight_scratch + tile * gqa::kPhysicalTileBytes);
  }

  size_t permutation_mismatches = 0;
  for (uint32_t tile = 0; tile < gqa::kQkDepthTiles; ++tile) {
    for (uint32_t k = 0; k < 32; ++k) {
      for (uint32_t n = 0; n < 32; ++n) {
        const int8_t expected = static_cast<int8_t>(static_cast<int32_t>(key[tile * 1024 + k * 32 + n]) - 128);
        const int8_t actual = weight_scratch[tile * 2048 + probe::weightOffset(k, n)];
        permutation_mismatches += actual != expected;
      }
    }
  }

  // QNN's logical H=1 tensors expose each eight-row W tile as a separate
  // 8x8 Crouton block. Verify the Stage-B gather discards the 56 padding rows
  // in every source block and reconstructs exactly one logical 32x32 tile.
  auto* crouton_sources = static_cast<uint8_t*>(std::aligned_alloc(2048, 4 * gqa::kPhysicalTileBytes));
  auto* gathered_weight = static_cast<int8_t*>(std::aligned_alloc(2048, gqa::kPhysicalTileBytes));
  if (crouton_sources == nullptr || gathered_weight == nullptr) return 2;
  std::memset(crouton_sources, 0x5a, 4 * gqa::kPhysicalTileBytes);
  const void* source_blocks[4];
  for (uint32_t chunk = 0; chunk < 4; ++chunk) {
    auto* block = crouton_sources + chunk * gqa::kPhysicalTileBytes;
    source_blocks[chunk] = block;
    for (uint32_t row = 0; row < 8; ++row) {
      for (uint32_t n = 0; n < 32; ++n) {
        block[row * 32 + n] = static_cast<uint8_t>(128 + ((chunk * 17 + row * 7 + n * 5) % 31) - 15);
      }
    }
  }
  std::memset(gathered_weight, 0, gqa::kPhysicalTileBytes);
  gqa::packFourCroutonRowsHvx(source_blocks, gathered_weight);
  size_t crouton_gather_mismatches = 0;
  for (uint32_t k = 0; k < 32; ++k) {
    for (uint32_t n = 0; n < 32; ++n) {
      const uint32_t chunk = k / 8;
      const uint32_t row = k % 8;
      const auto* block = crouton_sources + chunk * gqa::kPhysicalTileBytes;
      const int8_t expected = static_cast<int8_t>(static_cast<int32_t>(block[row * 32 + n]) - 128);
      crouton_gather_mismatches += gathered_weight[probe::weightOffset(k, n)] != expected;
    }
  }

  for (uint32_t n = 0; n < 32; ++n) {
    int32_t weight_sum = 0;
    for (uint32_t tile = 0; tile < gqa::kQkDepthTiles; ++tile) {
      for (uint32_t k = 0; k < 32; ++k) weight_sum += weight_scratch[tile * 2048 + probe::weightOffset(k, n)];
    }
    bias[n] = gqa::kQkConvertLowerWord;
    bias[32 + n] = static_cast<uint32_t>(-gqa::kQueryZeroPoint * weight_sum + gqa::kQkOutputOffsetInAccumulator);
  }
  gqa::executeU8S8Deep(query, weight_scratch, gqa::kQkDepthTiles, bias, output);

  size_t qk_emulator_mismatches = 0;
  size_t qk_target_errors_gt_one = 0;
  int32_t qk_target_max_error = 0;
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t n = 0; n < 32; ++n) {
      int32_t accumulator = 0;
      int32_t weight_sum = 0;
      for (uint32_t tile = 0; tile < gqa::kQkDepthTiles; ++tile) {
        for (uint32_t k = 0; k < 32; ++k) {
          const int32_t q = query[tile * 2048 + row * 32 + k];
          const int32_t w = weight_scratch[tile * 2048 + probe::weightOffset(k, n)];
          accumulator += q * w;
          weight_sum += w;
        }
      }
      const int32_t total_bias = -gqa::kQueryZeroPoint * weight_sum + gqa::kQkOutputOffsetInAccumulator;
      const uint8_t expected_emulator = requantize(accumulator, total_bias, gqa::kQkEffectiveScale);
      const int32_t centered_accumulator = accumulator - gqa::kQueryZeroPoint * weight_sum;
      const uint8_t expected_target = static_cast<uint8_t>(std::clamp(
          static_cast<int32_t>(std::floor(centered_accumulator * gqa::kQkTargetScale + 0.5F)) + gqa::kScoreZeroPoint,
          0, 255));
      const uint8_t actual = output[row * 32 + n];
      qk_emulator_mismatches += actual != expected_emulator;
      const int32_t target_error = std::abs(static_cast<int32_t>(actual) - static_cast<int32_t>(expected_target));
      qk_target_max_error = std::max(qk_target_max_error, target_error);
      qk_target_errors_gt_one += target_error > 1;
    }
  }

  auto* probability = static_cast<uint8_t*>(std::aligned_alloc(2048, gqa::kAvDepthTiles * gqa::kPhysicalTileBytes));
  auto* value = static_cast<uint8_t*>(std::aligned_alloc(2048, gqa::kAvDepthTiles * gqa::kPackedWeightBytes));
  auto* av_weight = static_cast<int8_t*>(std::aligned_alloc(2048, gqa::kAvDepthTiles * gqa::kPhysicalTileBytes));
  if (probability == nullptr || value == nullptr || av_weight == nullptr) return 2;
  std::memset(probability, 0, gqa::kAvDepthTiles * gqa::kPhysicalTileBytes);
  for (uint32_t tile = 0; tile < gqa::kAvDepthTiles; ++tile) {
    for (uint32_t row = 0; row < 32; ++row) {
      for (uint32_t channel = 0; channel < 32; ++channel) {
        probability[tile * 2048 + row * 32 + channel] =
            static_cast<uint8_t>((row * 13 + channel * 3 + tile * 5) % 17);
        value[tile * 1024 + row * 32 + channel] =
            static_cast<uint8_t>(gqa::kValueZeroPoint + ((row * 7 + channel * 11 + tile * 3) % 25) - 12);
      }
    }
    gqa::packSignedWeightTileHvx(value + tile * 1024, av_weight + tile * 2048);
  }
  for (uint32_t n = 0; n < 32; ++n) {
    bias[n] = gqa::kAvConvertLowerWord;
    bias[32 + n] = static_cast<uint32_t>(gqa::kAvOutputOffsetInAccumulator);
  }
  gqa::executeU8S8Deep(probability, av_weight, gqa::kAvDepthTiles, bias, output);

  int32_t av_target_max_error = 0;
  size_t av_target_errors_gt_one = 0;
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t n = 0; n < 32; ++n) {
      int32_t accumulator = 0;
      for (uint32_t tile = 0; tile < gqa::kAvDepthTiles; ++tile) {
        for (uint32_t k = 0; k < 32; ++k) {
          accumulator += probability[tile * 2048 + row * 32 + k]
                         * av_weight[tile * 2048 + probe::weightOffset(k, n)];
        }
      }
      const uint8_t target = static_cast<uint8_t>(std::clamp(
          static_cast<int32_t>(std::floor(accumulator * gqa::kAvTargetScale + 0.5F)) + gqa::kOutputZeroPoint, 0, 255));
      const int32_t error = std::abs(static_cast<int32_t>(output[row * 32 + n]) - static_cast<int32_t>(target));
      av_target_max_error = std::max(av_target_max_error, error);
      av_target_errors_gt_one += error > 1;
    }
  }

  std::cout << "permutation_mismatches=" << permutation_mismatches
            << " crouton_gather_mismatches=" << crouton_gather_mismatches
            << " qk_emulator_mismatches=" << qk_emulator_mismatches
            << " qk_target_max_error=" << qk_target_max_error
            << " qk_target_errors_gt_one=" << qk_target_errors_gt_one
            << " av_target_max_error=" << av_target_max_error
            << " av_target_errors_gt_one=" << av_target_errors_gt_one << '\n';
  std::free(av_weight);
  std::free(value);
  std::free(probability);
  std::free(bias);
  munmap(output, 4096);
  std::free(weight_scratch);
  std::free(key);
  std::free(query);
  std::free(gathered_weight);
  std::free(crouton_sources);
  return permutation_mismatches == 0 && crouton_gather_mismatches == 0 && qk_target_errors_gt_one == 0
                 && av_target_errors_gt_one == 0
             ? 0
             : 1;
}
