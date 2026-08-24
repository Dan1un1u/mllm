// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

namespace mllm::qnn::qhpi_hmx_probe {

// V79 integer HMX consumes one U8 activation crouton (64 spatial rows by
// 32 channels) and one S8 weight tile (32 input by 32 output channels).
inline constexpr uint32_t kSpatial = 64;
inline constexpr uint32_t kInputChannels = 32;
inline constexpr uint32_t kOutputChannels = 32;
inline constexpr uint32_t kActivationBytes = 2048;
inline constexpr uint32_t kWeightBytes = 1024;
inline constexpr uint32_t kOutputBytes = 2048;
inline constexpr uint32_t kBiasBytes = 256;

// QHPI's Crouton8 signature exposes the channel-major view consumed and
// produced by HMX's :cm instructions. The signed weight tile keeps HMX's
// native K4/N packing.
inline constexpr size_t activationOffset(uint32_t spatial, uint32_t channel) {
  return static_cast<size_t>(spatial) * kInputChannels + channel;
}

inline constexpr size_t weightOffset(uint32_t input_channel, uint32_t output_channel) {
  return (static_cast<size_t>(input_channel / 4) * kOutputChannels + output_channel) * 4 + input_channel % 4;
}

inline constexpr size_t outputOffset(uint32_t spatial, uint32_t channel) {
  return static_cast<size_t>(spatial) * kOutputChannels + channel;
}

// U8 has six spatial bits. 111000 selects an 8x8 YYYXXX crouton. The
// remaining Rt fields select channels 0..31 and exactly one crouton.
inline constexpr uint32_t kSpatialMask = 0b111000;
inline constexpr uint32_t kActivationRt = ((kSpatialMask >> 2) << 7) | (31u << 2) | (kSpatialMask & 0x3u);
inline constexpr uint32_t kWeightRt = kWeightBytes - 1;
inline constexpr uint32_t kWriteRt = ((kSpatialMask >> 2) << 7) | (kSpatialMask & 0x3u);

// Integer HMX stores one 64-bit bias register per output channel as all 32
// lower words followed by all 32 upper words. For an identity accumulator to
// U8 conversion, exponent=24 and significand=0 encode scale 1. The upper word
// is the signed accumulator bias.
inline constexpr uint32_t kIdentityConvertLowerWord = 24u << 10;

inline void fillAsymmetricBias(const int8_t* packed_weight, int32_t input_zero_point, uint32_t* bias_words) {
  for (uint32_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    int32_t weight_sum = 0;
    for (uint32_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
      weight_sum += packed_weight[weightOffset(input_channel, output_channel)];
    }
    bias_words[output_channel] = kIdentityConvertLowerWord;
    bias_words[kOutputChannels + output_channel] = static_cast<uint32_t>(-input_zero_point * weight_sum);
  }
}

inline void executeU8S8Tile(const uint8_t* activation, const int8_t* weight, const uint32_t* bias_words, uint8_t* output) {
  // HMX bias memory and input/output croutons have architectural alignment
  // requirements. The QHPI wrapper validates these before entering here.
  Q6_bias_mxmem2_A(const_cast<uint32_t*>(bias_words));
  Q6_mxclracc();
#if defined(__hexagon__)
  // Activation and weight must be consecutive and in one instruction packet.
  // The :cm load and :after:cm store are required to match QHPI Crouton8;
  // the default HMX view interleaves four spatial bytes per 32-bit word.
  asm volatile("{ activation.ub = mxmem(%0, %1):cm\n"
               "  weight.b = mxmem(%2, %3) }\n"
               :
               : "r"(activation), "r"(kActivationRt), "r"(weight), "r"(kWeightRt)
               : "memory");
  Q6_mxmem_AR_after_cm_sat_ub(output, kWriteRt);
#else
  // Hexagon Tools libnative emulates the same V79 integer-HMX state machine.
  Q6_activation_ub_mxmem_RR_cm(reinterpret_cast<uintptr_t>(activation), kActivationRt);
  Q6_weight_b_mxmem_RR(reinterpret_cast<uintptr_t>(weight), kWeightRt);
  Q6_mxmem_AR_after_cm_sat_ub(output, kWriteRt);
#endif
}

}  // namespace mllm::qnn::qhpi_hmx_probe
