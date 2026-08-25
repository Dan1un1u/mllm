// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// EXP-0016 Stage B: one QHPI HMX resource node that executes the complete
// first GQA group of layer-14 s32 attention as QK -> masked U8 Softmax -> AV.
// QK and AV use V79 integer HMX. The in-place Softmax is HVX SIMD. All tensor
// and scratch storage is constrained to indirect Crouton8 TCM blocks.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "HTP/core/qhpi.h"
#include "HmxGqaCore.hpp"

#if !defined(REFERENCE_OP)
#include <hexagon_protos.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

namespace gqa = mllm::qnn::qhpi_hmx_probe::gqa;
namespace probe = mllm::qnn::qhpi_hmx_probe;

namespace {

constexpr uint32_t kTokenTiles = gqa::kRows / gqa::kCroutonSpatialEdge;
constexpr uint32_t kHeadDimSpatialTiles = gqa::kHeadDim / gqa::kCroutonSpatialEdge;
constexpr uint32_t kContextSpatialTiles = gqa::kContext / gqa::kCroutonSpatialEdge;
constexpr uint32_t kQueryBlocks = kTokenTiles * gqa::kQkDepthTiles;
constexpr uint32_t kKeyBlocks = kHeadDimSpatialTiles * gqa::kScoreBlocks;
constexpr uint32_t kValueBlocks = kContextSpatialTiles * gqa::kOutputBlocks;
constexpr uint32_t kMaskBlocks = kTokenTiles * gqa::kScoreBlocks;
constexpr uint32_t kOutputBlocks = kQueryBlocks;

#if !defined(REFERENCE_OP)
constexpr uint32_t kVectorBytes = sizeof(HVX_Vector);
constexpr uint32_t kRowsPerVector = kVectorBytes / probe::kOutputChannels;
constexpr uint8_t kMaskedCode = 15;
constexpr uint8_t kLargestExponentCode = 14;
constexpr uint32_t kReciprocalFractionBits = 23;

static_assert(kVectorBytes == 128);
static_assert(kRowsPerVector == 4);

constexpr uint16_t kExponentNumerator[16] = {
    16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0,
};

union alignas(128) VectorStorage {
  HVX_Vector vector;
  uint8_t bytes[kVectorBytes];
  uint16_t hwords[kVectorBytes / 2];
};
#endif

bool shapeEquals(const QHPI_Shape& shape, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
  return shape.rank == 4 && shape.dims[0] == d0 && shape.dims[1] == d1 && shape.dims[2] == d2
         && shape.dims[3] == d3;
}

bool quantEquals(const QHPI_Tensor* tensor, int32_t zero_offset, float step_size) {
  const auto quant = qhpi_tensor_quant_parameters(tensor);
  return quant.zero_offset == zero_offset
         && std::fabs(quant.stepsize - step_size) <= std::max(1.0e-8F, std::fabs(step_size) * 1.0e-6F);
}

#if !defined(REFERENCE_OP)
bool aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr && (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool contiguous(void* const* blocks, uint32_t start, uint32_t count) {
  if (blocks == nullptr || count == 0 || blocks[start] == nullptr || !aligned(blocks[start], gqa::kPhysicalTileBytes)) {
    return false;
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(blocks[start]);
  for (uint32_t index = 1; index < count; ++index) {
    if (reinterpret_cast<uintptr_t>(blocks[start + index]) != base + index * gqa::kPhysicalTileBytes) return false;
  }
  return true;
}

inline HVX_Vector loadVector(const void* pointer) {
  return *reinterpret_cast<const HVX_Vector*>(pointer);
}

inline void storeVector(void* pointer, HVX_Vector value) {
  *reinterpret_cast<HVX_Vector*>(pointer) = value;
}

inline HVX_Vector splatU8(uint8_t value) { return Q6_V_vsplat_R(Q6_R_vsplatb_R(value)); }

inline HVX_Vector xorPermute(HVX_Vector value, uint8_t byte_offset) {
  return Q6_V_vdelta_VV(value, splatU8(byte_offset));
}

inline HVX_Vector reduceMaxWithinRows(HVX_Vector value) {
  value = Q6_Vub_vmax_VubVub(value, xorPermute(value, 1));
  value = Q6_Vub_vmax_VubVub(value, xorPermute(value, 2));
  value = Q6_Vub_vmax_VubVub(value, xorPermute(value, 4));
  value = Q6_Vub_vmax_VubVub(value, xorPermute(value, 8));
  value = Q6_Vub_vmax_VubVub(value, xorPermute(value, 16));
  return value;
}

inline HVX_Vector sumAdjacentQ14Halfwords(HVX_Vector values) {
  const HVX_Vector low_bytes = Q6_V_vand_VV(values, Q6_V_vsplat_R(0x00ff00ff));
  const HVX_Vector high_bytes = Q6_Vuh_vlsr_VuhR(values, 8);
  const HVX_Vector low_sum = Q6_Vuw_vrmpy_VubRub(low_bytes, 0x01010101);
  const HVX_Vector high_sum = Q6_Vuw_vrmpy_VubRub(high_bytes, 0x01010101);
  return Q6_Vw_vadd_VwVw(low_sum, Q6_Vw_vasl_VwR(high_sum, 8));
}

inline HVX_Vector reduceSumWithinRows(HVX_Vector value) {
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 4));
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 8));
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 16));
  return value;
}

inline int probabilityTableOffset(int row, int code) {
  return (row & 1 ? 64 : 0) + 2 * code + (row >= 2 ? 1 : 0);
}

void fillDeepBias(const int8_t* weight_tiles, uint32_t tiles, int32_t input_zero_point,
                  uint32_t converter_word, int32_t output_offset, uint32_t* bias_words) {
  for (uint32_t output_channel = 0; output_channel < probe::kOutputChannels; ++output_channel) {
    int32_t weight_sum = 0;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
      const int8_t* weight = weight_tiles + tile * gqa::kPhysicalTileBytes;
      for (uint32_t input_channel = 0; input_channel < probe::kInputChannels; ++input_channel) {
        weight_sum += weight[probe::weightOffset(input_channel, output_channel)];
      }
    }
    bias_words[output_channel] = converter_word;
    bias_words[probe::kOutputChannels + output_channel] =
        static_cast<uint32_t>(-input_zero_point * weight_sum + output_offset);
  }
}

void copyBlockHvx(const void* source, void* destination) {
  for (uint32_t offset = 0; offset < gqa::kPhysicalTileBytes; offset += kVectorBytes) {
    storeVector(static_cast<uint8_t*>(destination) + offset,
                loadVector(static_cast<const uint8_t*>(source) + offset));
  }
}

uint8_t maskByte(void* const* mask_blocks, uint32_t row, uint32_t depth) {
  const uint32_t token_tile = row / gqa::kCroutonSpatialEdge;
  const uint32_t token_lane = row % gqa::kCroutonSpatialEdge;
  const uint32_t depth_block = depth / probe::kOutputChannels;
  const auto* block = static_cast<const uint8_t*>(
      mask_blocks[token_tile * gqa::kScoreBlocks + depth_block]);
  return block[token_lane * probe::kOutputChannels + depth % probe::kOutputChannels];
}

bool recoverCausalPrefixLengths(void* const* mask_blocks, uint16_t* valid_counts) {
  for (uint32_t row = 0; row < gqa::kRows; ++row) {
    int32_t depth = static_cast<int32_t>(gqa::kContext) - 1;
    while (depth >= 0 && maskByte(mask_blocks, row, static_cast<uint32_t>(depth)) != 255) --depth;
    valid_counts[row] = static_cast<uint16_t>(depth + 1);
    if (valid_counts[row] == 0) return false;
    // The fixed s32 attention contract is a causal prefix. Checking the two
    // transition bytes catches an inverted or non-prefix mask without a full
    // second tensor scan.
    if (maskByte(mask_blocks, row, 0) != 255) return false;
    if (valid_counts[row] < gqa::kContext && maskByte(mask_blocks, row, valid_counts[row]) == 255) return false;
  }
  return true;
}

HVX_VectorPred validLanes(uint32_t depth_block, const uint16_t* valid_counts,
                          VectorStorage* lane_index, VectorStorage* thresholds) {
  const uint32_t depth_base = depth_block * probe::kOutputChannels;
  const uint32_t depth_end = depth_base + probe::kOutputChannels;
  uint32_t minimum = gqa::kContext;
  uint32_t maximum = 0;
  for (uint32_t row = 0; row < kRowsPerVector; ++row) {
    minimum = std::min<uint32_t>(minimum, valid_counts[row]);
    maximum = std::max<uint32_t>(maximum, valid_counts[row]);
  }
  const HVX_Vector zero = Q6_V_vzero();
  if (depth_end <= minimum) return Q6_Q_vcmp_eq_VbVb(zero, zero);
  if (depth_base >= maximum) return Q6_Q_vcmp_gt_VubVub(zero, zero);

  for (uint32_t row = 0; row < kRowsPerVector; ++row) {
    const uint32_t count = valid_counts[row];
    const uint8_t lanes = static_cast<uint8_t>(count <= depth_base ? 0 : std::min<uint32_t>(32, count - depth_base));
    std::fill_n(thresholds->bytes + row * probe::kOutputChannels, probe::kOutputChannels, lanes);
  }
  return Q6_Q_vcmp_gt_VubVub(thresholds->vector, lane_index->vector);
}

void softmaxInPlace(void* const* score_blocks, const uint16_t* valid_counts, uint8_t* scratch) {
  auto* exponent_lut = reinterpret_cast<VectorStorage*>(scratch);
  auto* lane_index = reinterpret_cast<VectorStorage*>(scratch + 128);
  auto* thresholds = reinterpret_cast<VectorStorage*>(scratch + 256);
  auto* probability_lut = reinterpret_cast<VectorStorage*>(scratch + 384);
  auto* row_index_base = reinterpret_cast<VectorStorage*>(scratch + 512);
  *exponent_lut = {};
  for (int code = 0; code <= kMaskedCode; ++code) exponent_lut->hwords[2 * code] = kExponentNumerator[code];
  for (uint32_t row = 0; row < kRowsPerVector; ++row) {
    for (uint32_t lane = 0; lane < probe::kOutputChannels; ++lane) {
      lane_index->bytes[row * probe::kOutputChannels + lane] = static_cast<uint8_t>(lane);
      row_index_base->bytes[row * probe::kOutputChannels + lane] = static_cast<uint8_t>(row * 32);
    }
  }

  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector max_code = splatU8(kLargestExponentCode);
  const HVX_Vector masked_code = splatU8(kMaskedCode);
  const HVX_Vector round_q8_h = Q6_Vh_vsplat_R(128);
  const HVX_VectorPair round_q8 = Q6_W_vcombine_VV(round_q8_h, round_q8_h);

  for (uint32_t token_tile = 0; token_tile < kTokenTiles; ++token_tile) {
    // y=0 carries head 0 and y=1 carries head 1. Each physical 128-byte
    // vector covers four adjacent x positions, so the four useful groups are
    // offsets 0, 4, 8, and 12 spatial rows inside the 8x8 Crouton block.
    for (uint32_t physical_row_base = 0; physical_row_base < 16;
         physical_row_base += kRowsPerVector) {
      const uint32_t token_base = token_tile * gqa::kCroutonSpatialEdge
                                  + (physical_row_base % gqa::kCroutonSpatialEdge);
      const uint16_t group_valid_counts[kRowsPerVector] = {
          valid_counts[token_base], valid_counts[token_base + 1],
          valid_counts[token_base + 2], valid_counts[token_base + 3],
      };

      HVX_Vector row_max = zero;
      for (uint32_t depth_block = 0; depth_block < gqa::kScoreBlocks; ++depth_block) {
        const auto* pointer = static_cast<const uint8_t*>(
                                  score_blocks[token_tile * gqa::kScoreBlocks + depth_block])
                              + physical_row_base * probe::kOutputChannels;
        const HVX_Vector score = loadVector(pointer);
        const HVX_VectorPred valid =
            validLanes(depth_block, group_valid_counts, lane_index, thresholds);
        row_max = Q6_Vub_vmax_VubVub(row_max, Q6_V_vmux_QVV(valid, score, zero));
      }
      row_max = reduceMaxWithinRows(row_max);

      HVX_Vector row_sum_vector = zero;
      for (uint32_t depth_block = 0; depth_block < gqa::kScoreBlocks; ++depth_block) {
        auto* pointer = static_cast<uint8_t*>(
                            score_blocks[token_tile * gqa::kScoreBlocks + depth_block])
                        + physical_row_base * probe::kOutputChannels;
        const HVX_Vector score = loadVector(pointer);
        const HVX_VectorPred valid =
            validLanes(depth_block, group_valid_counts, lane_index, thresholds);
        const HVX_Vector difference = Q6_Vub_vsub_VubVub_sat(row_max, score);
        HVX_VectorPair product =
            Q6_Wuh_vmpy_VubRub(difference, Q6_R_vsplatb_R(gqa::kSoftmaxExponentCoefficient));
        product = Q6_Wh_vadd_WhWh(product, round_q8);
        HVX_Vector code = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(product), Q6_V_lo_W(product));
        code = Q6_Vub_vmin_VubVub(code, max_code);
        code = Q6_V_vmux_QVV(valid, code, masked_code);
        storeVector(pointer, code);

        const HVX_VectorPair numerator = Q6_Wh_vlut16_VbVhR(code, exponent_lut->vector, 0);
        const HVX_Vector even_sum = sumAdjacentQ14Halfwords(Q6_V_lo_W(numerator));
        const HVX_Vector odd_sum = sumAdjacentQ14Halfwords(Q6_V_hi_W(numerator));
        row_sum_vector = Q6_Vw_vadd_VwVw(row_sum_vector, Q6_Vw_vadd_VwVw(even_sum, odd_sum));
      }
      row_sum_vector = reduceSumWithinRows(row_sum_vector);
      const uint32_t row_sum[kRowsPerVector] = {
          static_cast<uint32_t>(Q6_R_vextract_VR(row_sum_vector, 0)),
          static_cast<uint32_t>(Q6_R_vextract_VR(row_sum_vector, 32)),
          static_cast<uint32_t>(Q6_R_vextract_VR(row_sum_vector, 64)),
          static_cast<uint32_t>(Q6_R_vextract_VR(row_sum_vector, 96)),
      };

      std::fill_n(probability_lut->bytes, kVectorBytes, 0);
      for (uint32_t row = 0; row < kRowsPerVector; ++row) {
        if (row_sum[row] == 0) continue;
        const uint32_t reciprocal_q23 =
            ((255u << kReciprocalFractionBits) + row_sum[row] / 2) / row_sum[row];
        for (int code = 0; code <= kLargestExponentCode; ++code) {
          const uint32_t probability =
              (static_cast<uint32_t>(kExponentNumerator[code]) * reciprocal_q23
               + (1u << (kReciprocalFractionBits - 1)))
              >> kReciprocalFractionBits;
          probability_lut->bytes[probabilityTableOffset(static_cast<int>(row), code)] =
              static_cast<uint8_t>(std::min<uint32_t>(255, probability));
        }
      }

      for (uint32_t depth_block = 0; depth_block < gqa::kScoreBlocks; ++depth_block) {
        auto* pointer = static_cast<uint8_t*>(
                            score_blocks[token_tile * gqa::kScoreBlocks + depth_block])
                        + physical_row_base * probe::kOutputChannels;
        const HVX_Vector code = loadVector(pointer);
        const HVX_Vector row_index = Q6_V_vor_VV(code, row_index_base->vector);
        HVX_Vector probability = Q6_Vb_vlut32_VbVbR(row_index, probability_lut->vector, 0);
        probability = Q6_Vb_vlut32or_VbVbVbR(probability, row_index, probability_lut->vector, 1);
        probability = Q6_Vb_vlut32or_VbVbVbR(probability, row_index, probability_lut->vector, 2);
        probability = Q6_Vb_vlut32or_VbVbVbR(probability, row_index, probability_lut->vector, 3);
        storeVector(pointer, probability);
      }
    }
  }
}

#endif

uint32_t fusedGqaHmxSoftmaxAv(QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs,
                              uint32_t num_inputs, const QHPI_Tensor* const* inputs) {
  if (handle == nullptr || num_inputs != 4 || num_outputs != 1 || inputs == nullptr || outputs == nullptr
      || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr || inputs[3] == nullptr
      || outputs[0] == nullptr) {
    return QHPI_ERROR_FATAL;
  }
  const uint32_t resources = qhpi_thread_resources(handle);
  if (resources != 0 && resources != QHPI_RESOURCE_HMX) return QHPI_ERROR_FATAL;
  if (!shapeEquals(qhpi_tensor_shape(inputs[0]), 1, 2, gqa::kRows, gqa::kHeadDim)
      || !shapeEquals(qhpi_tensor_shape(inputs[1]), 1, 1, gqa::kHeadDim, gqa::kContext)
      || !shapeEquals(qhpi_tensor_shape(inputs[2]), 1, 1, gqa::kContext, gqa::kHeadDim)
      || !shapeEquals(qhpi_tensor_shape(inputs[3]), 1, 1, gqa::kRows, gqa::kContext)
      || !shapeEquals(qhpi_tensor_shape(outputs[0]), 1, 2, gqa::kRows, gqa::kHeadDim)
      || !quantEquals(inputs[0], gqa::kQueryZeroPoint, gqa::kQueryScale)
      || !quantEquals(inputs[1], gqa::kKeyZeroPoint, gqa::kKeyScale)
      || !quantEquals(inputs[2], gqa::kValueZeroPoint, gqa::kValueScale)
      || !quantEquals(inputs[3], 255, gqa::kMaskScale)
      || !quantEquals(outputs[0], gqa::kOutputZeroPoint, gqa::kOutputScale)
      || qhpi_tensor_block_table_length(inputs[0]) != kQueryBlocks
      || qhpi_tensor_block_table_length(inputs[1]) != kKeyBlocks
      || qhpi_tensor_block_table_length(inputs[2]) != kValueBlocks
      || qhpi_tensor_block_table_length(inputs[3]) != kMaskBlocks
      || qhpi_tensor_block_table_length(outputs[0]) != kOutputBlocks) {
    return QHPI_ERROR_FATAL;
  }

#if defined(REFERENCE_OP)
  return QHPI_UNSUPPORTED;
#else
  auto** query_blocks = qhpi_tensor_block_table(inputs[0]);
  auto** key_blocks = qhpi_tensor_block_table(inputs[1]);
  auto** value_blocks = qhpi_tensor_block_table(inputs[2]);
  auto** mask_blocks = qhpi_tensor_block_table(inputs[3]);
  auto** output_blocks = qhpi_tensor_block_table(outputs[0]);
  if (!contiguous(query_blocks, 0, kQueryBlocks) || !contiguous(key_blocks, 0, kKeyBlocks)
      || !contiguous(value_blocks, 0, kValueBlocks) || !contiguous(mask_blocks, 0, kMaskBlocks)
      || !contiguous(output_blocks, 0, kOutputBlocks)) {
    return QHPI_ERROR_FATAL;
  }

  // Output blocks are dead until the final copy. Use blocks 0..3 for packed
  // QK weights and block 4 for the 256-byte HMX bias plus HVX Softmax scratch.
  auto* qk_weight = reinterpret_cast<int8_t*>(output_blocks[0]);
  auto* bias = reinterpret_cast<uint32_t*>(output_blocks[4]);
  auto* valid_counts = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(output_blocks[4]) + 512);
  auto* softmax_scratch = static_cast<uint8_t*>(output_blocks[4]) + 768;
  if (!aligned(qk_weight, 2048) || !aligned(bias, 256) || !aligned(softmax_scratch, 128)
      || !recoverCausalPrefixLengths(mask_blocks, valid_counts)) {
    return QHPI_ERROR_FATAL;
  }

  // QK: each packed query crouton contains both heads (physical y=0 and
  // y=1). Pack each logical 128x32 K slice once, then run it over all four
  // eight-token tiles. One HMX call computes both heads simultaneously.
  for (uint32_t score_block = 0; score_block < gqa::kScoreBlocks; ++score_block) {
    for (uint32_t tile = 0; tile < gqa::kQkDepthTiles; ++tile) {
      const void* sources[4];
      for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        sources[chunk] = key_blocks[(tile * 4 + chunk) * gqa::kScoreBlocks + score_block];
      }
      gqa::packFourCroutonRowsHvx(sources, qk_weight + tile * gqa::kPhysicalTileBytes);
    }
    fillDeepBias(qk_weight, gqa::kQkDepthTiles, gqa::kQueryZeroPoint, gqa::kQkConvertLowerWord,
                 gqa::kQkOutputOffsetInAccumulator, bias);
    for (uint32_t token_tile = 0; token_tile < kTokenTiles; ++token_tile) {
      gqa::executeU8S8Deep(static_cast<const uint8_t*>(query_blocks[token_tile * gqa::kQkDepthTiles]),
                           qk_weight, gqa::kQkDepthTiles, bias,
                           static_cast<uint8_t*>(mask_blocks[token_tile * gqa::kScoreBlocks + score_block]));
    }
  }

  // Both heads' score/probability tensors share the physical y=0/y=1 rows of
  // the mask croutons. No second score tensor or intermediate allocation is
  // required.
  softmaxInPlace(mask_blocks, valid_counts, softmax_scratch);

  // AV: K is now dead. Reuse its first 32 contiguous blocks as the 64 KiB
  // deep-weight scratch. The probability croutons again carry both heads.
  auto* av_weight = reinterpret_cast<int8_t*>(key_blocks[0]);
  for (uint32_t output_block = 0; output_block < gqa::kOutputBlocks; ++output_block) {
    for (uint32_t tile = 0; tile < gqa::kAvDepthTiles; ++tile) {
      const void* sources[4];
      for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        sources[chunk] = value_blocks[(tile * 4 + chunk) * gqa::kOutputBlocks + output_block];
      }
      gqa::packFourCroutonRowsHvx(sources, av_weight + tile * gqa::kPhysicalTileBytes);
    }
    fillDeepBias(av_weight, gqa::kAvDepthTiles, gqa::kProbabilityZeroPoint, gqa::kAvConvertLowerWord,
                 gqa::kAvOutputOffsetInAccumulator, bias);
    for (uint32_t token_tile = 0; token_tile < kTokenTiles; ++token_tile) {
      gqa::executeU8S8Deep(static_cast<const uint8_t*>(mask_blocks[token_tile * gqa::kScoreBlocks]),
                           av_weight, gqa::kAvDepthTiles, bias,
                           static_cast<uint8_t*>(query_blocks[token_tile * gqa::kOutputBlocks + output_block]));
    }
  }

  for (uint32_t block = 0; block < kOutputBlocks; ++block) copyBlockHvx(query_blocks[block], output_blocks[block]);
  return QHPI_SUCCESS;
#endif
}

QHPI_Tensor_Signature_v1 input_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Tensor_Signature_v1 output_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 kernels[] = {{.function_name = THIS_PKG_NAME_STR "::fusedGqaHmxSoftmaxAv",
                             .function = fusedGqaHmxSoftmaxAv,
                             .resources = QHPI_RESOURCE_HMX,
                             .source_destructive = false,
                             .multithreaded = false,
                             .variable_inputs = false,
                             .variable_outputs = false,
                             .min_inputs = 4,
                             .input_signature = input_signatures,
                             .min_outputs = 1,
                             .output_signature = output_signatures,
                             .cost_function = nullptr,
                             .sync_block_size = 0,
                             .precomputed_data_size = 0,
                             .do_precomputation_function = nullptr,
                             .function_with_precomputed_data = nullptr,
                             .predicate = nullptr,
                             .Reserved_1 = nullptr,
                             .Reserved_2 = nullptr,
                             .Reserved_3 = nullptr,
                             .Reserved_4 = nullptr}};

QHPI_OpInfo_v1 op_info[] = {{.name = THIS_PKG_NAME_STR "::FusedGqaHmxSoftmaxAv",
                             .num_kernels = 1,
                             .kernels = kernels,
                             .early_rewrite = nullptr,
                             .shape_required = nullptr,
                             .shape_legalized = nullptr,
                             .tile_output = 0,
                             .build_tile = nullptr,
                             .late_rewrite = nullptr,
                             .Reserved_1 = nullptr,
                             .Reserved_2 = nullptr,
                             .Reserved_3 = nullptr,
                             .Reserved_4 = nullptr}};

}  // namespace

const QHPI_OpInfo_v1* fused_gqa_hmx_softmax_av_op_info() { return op_info; }
