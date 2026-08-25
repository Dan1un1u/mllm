// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "HmxInt8Tile.hpp"

namespace mllm::qnn::qhpi_hmx_probe::gqa {

inline constexpr uint32_t kPhysicalTileBytes = 2048;
inline constexpr uint32_t kPackedWeightBytes = 1024;
inline constexpr uint32_t kRows = 32;
inline constexpr uint32_t kHeadDim = 128;
inline constexpr uint32_t kContext = 1024;
inline constexpr uint32_t kQkDepthTiles = kHeadDim / kInputChannels;
inline constexpr uint32_t kAvDepthTiles = kContext / kInputChannels;
inline constexpr uint32_t kScoreBlocks = kContext / kOutputChannels;
inline constexpr uint32_t kOutputBlocks = kHeadDim / kOutputChannels;
inline constexpr uint32_t kCroutonSpatialEdge = 8;
inline constexpr uint32_t kCroutonSpatialRows = kCroutonSpatialEdge * kCroutonSpatialEdge;
inline constexpr uint32_t kRowsPerPackedWeightVector = sizeof(HVX_Vector) / kOutputChannels;

// Layer-14 s32 contract from the accepted QAIRT 2.49 RMSNorm-A8 baseline.
inline constexpr int32_t kQueryZeroPoint = 122;
inline constexpr int32_t kKeyZeroPoint = 128;
inline constexpr int32_t kValueZeroPoint = 128;
inline constexpr int32_t kScoreZeroPoint = 156;
inline constexpr int32_t kProbabilityZeroPoint = 0;
inline constexpr int32_t kOutputZeroPoint = 229;
inline constexpr float kQueryScale = 0.32302287220954895F;
inline constexpr float kKeyScale = 0.33071592450141907F;
inline constexpr float kValueScale = 1.4026927947998047F;
inline constexpr float kMaskScale = 3.9215688047988815e-6F;
inline constexpr float kProbabilityScale = 1.0F / 255.0F;
inline constexpr float kOutputScale = 0.5333649516105652F;

// HMX converter scale = (1 + significand / 1024) * 2^(exponent - 24).
// QK folds 1/sqrt(128) into the converter. AV consumes U8 probability
// scale 1/255. The chosen encodings approximate the baseline qparams by
// 0.0213% and 0.0159%, respectively.
inline constexpr uint32_t kQkConvertLowerWord = (19u << 10) | 176u;
inline constexpr uint32_t kAvConvertLowerWord = (17u << 10) | 328u;
inline constexpr int32_t kQkOutputOffsetInAccumulator = 4260;
inline constexpr int32_t kAvOutputOffsetInAccumulator = 22201;
inline constexpr float kQkEffectiveScale = 0.03662109375F;
inline constexpr float kAvEffectiveScale = 0.01031494140625F;
inline constexpr float kQkTargetScale = 0.03661327941174323F;
inline constexpr float kAvTargetScale = 0.010313306191995176F;
inline constexpr float kScoreScale = 0.2578960955142975F;
inline constexpr uint8_t kSoftmaxExponentCoefficient = 95;

// Generated with Qualcomm's Vdelta_Helper for the permutation
//   destination[4*n + k] = source[32*k + n]
// which converts four Crouton rows into HMX's native K4/N S8 packing.
alignas(128) inline constexpr uint8_t kWeightVrdeltaControl[128] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
    0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15,
    0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
    0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
};

alignas(128) inline constexpr uint8_t kWeightVdeltaControl[128] = {
    0x00, 0x04, 0x08, 0x0c, 0x11, 0x15, 0x19, 0x1d, 0x22, 0x26, 0x2a, 0x2e, 0x33, 0x37, 0x3b, 0x3f,
    0x24, 0x20, 0x2c, 0x28, 0x35, 0x31, 0x3d, 0x39, 0x06, 0x02, 0x0e, 0x0a, 0x17, 0x13, 0x1f, 0x1b,
    0x08, 0x0c, 0x00, 0x04, 0x19, 0x1d, 0x11, 0x15, 0x2a, 0x2e, 0x22, 0x26, 0x3b, 0x3f, 0x33, 0x37,
    0x2c, 0x28, 0x24, 0x20, 0x3d, 0x39, 0x35, 0x31, 0x0e, 0x0a, 0x06, 0x02, 0x1f, 0x1b, 0x17, 0x13,
    0x30, 0x34, 0x38, 0x3c, 0x21, 0x25, 0x29, 0x2d, 0x12, 0x16, 0x1a, 0x1e, 0x03, 0x07, 0x0b, 0x0f,
    0x14, 0x10, 0x1c, 0x18, 0x05, 0x01, 0x0d, 0x09, 0x36, 0x32, 0x3e, 0x3a, 0x27, 0x23, 0x2f, 0x2b,
    0x38, 0x3c, 0x30, 0x34, 0x29, 0x2d, 0x21, 0x25, 0x1a, 0x1e, 0x12, 0x16, 0x0b, 0x0f, 0x03, 0x07,
    0x1c, 0x18, 0x14, 0x10, 0x0d, 0x09, 0x05, 0x01, 0x3e, 0x3a, 0x36, 0x32, 0x2f, 0x2b, 0x27, 0x23,
};

inline HVX_Vector loadAlignedVector(const void* pointer) {
  return *reinterpret_cast<const HVX_Vector*>(pointer);
}

inline void storeAlignedVector(void* pointer, HVX_Vector value) {
  *reinterpret_cast<HVX_Vector*>(pointer) = value;
}

inline void packSignedWeightTileHvx(const uint8_t* source_row_major, int8_t* destination_k4n) {
  const HVX_Vector vrdelta_control = loadAlignedVector(kWeightVrdeltaControl);
  const HVX_Vector vdelta_control = loadAlignedVector(kWeightVdeltaControl);
  const HVX_Vector sign_flip = Q6_V_vsplat_R(0x80808080);
  for (uint32_t offset = 0; offset < kPackedWeightBytes; offset += sizeof(HVX_Vector)) {
    HVX_Vector value = loadAlignedVector(source_row_major + offset);
    value = Q6_V_vrdelta_VV(value, vrdelta_control);
    value = Q6_V_vdelta_VV(value, vdelta_control);
    value = Q6_V_vxor_VV(value, sign_flip);
    storeAlignedVector(destination_k4n + offset, value);
  }
}

// A logical [32, 32] K/N weight tile is split across four QNN Crouton8
// blocks when the logical spatial height is one: each source block contains
// eight real rows at y=0 followed by 56 padding rows. Gather those four
// eight-row fragments while applying the same row-major -> HMX K4/N
// permutation and asymmetric-U8 -> signed-S8 conversion as the flat helper.
inline void packFourCroutonRowsHvx(const void* const source_blocks[4], int8_t* destination_k4n) {
  const HVX_Vector vrdelta_control = loadAlignedVector(kWeightVrdeltaControl);
  const HVX_Vector vdelta_control = loadAlignedVector(kWeightVdeltaControl);
  const HVX_Vector sign_flip = Q6_V_vsplat_R(0x80808080);
  static_assert(kRowsPerPackedWeightVector == 4);
  for (uint32_t group = 0; group < kInputChannels / kRowsPerPackedWeightVector; ++group) {
    const uint32_t source_block = group / 2;
    const uint32_t source_group = group & 1;
    const auto* source = static_cast<const uint8_t*>(source_blocks[source_block])
                         + source_group * sizeof(HVX_Vector);
    HVX_Vector value = loadAlignedVector(source);
    value = Q6_V_vrdelta_VV(value, vrdelta_control);
    value = Q6_V_vdelta_VV(value, vdelta_control);
    value = Q6_V_vxor_VV(value, sign_flip);
    storeAlignedVector(destination_k4n + group * sizeof(HVX_Vector), value);
  }
}

inline void executeU8S8Deep(const uint8_t* activation_tiles, const int8_t* weight_tiles, uint32_t tiles,
                            const uint32_t* bias_words, uint8_t* output) {
  Q6_bias_mxmem2_A(const_cast<uint32_t*>(bias_words));
  Q6_mxclracc();
#if defined(__hexagon__)
  const uint32_t activation_limit = tiles * kPhysicalTileBytes - 1;
  const uint32_t weight_limit = tiles * kPhysicalTileBytes - 1;
  asm volatile("{ activation.ub = mxmem(%0, %1):deep:cm\n"
               "  weight.b = mxmem(%2, %3):deep }\n"
               :
               : "r"(activation_tiles), "r"(activation_limit), "r"(weight_tiles), "r"(weight_limit)
               : "memory");
#else
  Q6_activation_ub_mxmem_RR_deep_cm(reinterpret_cast<uintptr_t>(activation_tiles),
                                    tiles * kPhysicalTileBytes - 1);
  Q6_weight_b_mxmem_RR_deep(reinterpret_cast<uintptr_t>(weight_tiles), tiles * kPhysicalTileBytes - 1);
#endif
  Q6_mxmem_AR_after_cm_sat_ub(output, kWriteRt);
}

}  // namespace mllm::qnn::qhpi_hmx_probe::gqa
