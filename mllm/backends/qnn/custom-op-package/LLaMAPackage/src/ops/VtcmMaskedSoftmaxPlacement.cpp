// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// TCM-only, U8-in/U8-out causal E2Softmax specialized for Qwen3 head_dim=128.
// Position IDs encode the valid causal prefix, replacing repeated scans of the
// full [rows, 1024] mask. Fully masked 32-key blocks are never read.

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "HTP/core/intrinsics.h"
#ifdef MLLM_VTCM_SOFTMAX_QHPI_TRANSLATION_UNIT
#include "HTP/core/qhpi.h"
#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)
using Idx = int32_t;
#else
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#endif

#include <hexagon_types.h>

#ifndef MLLM_VTCM_SOFTMAX_QHPI_TRANSLATION_UNIT
BEGIN_PKG_OP_DEFINITION(PKG_VtcmCausalE2SoftmaxHd128);
#endif

enum class SoftmaxStatus : uint32_t {
  Success = 0,
  ErrorDimensions = 1,
};

namespace {

constexpr int kDepthLanes = 32;
constexpr int kVectorBytes = sizeof(HVX_Vector);
constexpr int kRowsPerVector = kVectorBytes / kDepthLanes;
constexpr uint8_t kMaskedCode = 15;
constexpr uint8_t kLargestExponentCode = 14;
constexpr uint32_t kReciprocalFractionBits = 23;
constexpr uint32_t kReciprocalQ30LinearA = 3031741621u;  // round((48 / 17) * 2^30)
constexpr uint32_t kReciprocalQ30LinearB = 2021161080u;  // round((32 / 17) * 2^30)
constexpr float kHeadDim128AttentionBeta = 0.08838834764831845f;

static_assert(kVectorBytes == 128, "The Crouton row kernel requires 128-byte HVX");
static_assert(kRowsPerVector == 4);

constexpr uint16_t kExponentNumerator[16] = {
    16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0,
};

union alignas(128) VectorStorage {
  HVX_Vector vector;
  uint8_t bytes[kVectorBytes];
  uint16_t hwords[kVectorBytes / 2];
  uint32_t words[kVectorBytes / 4];
};

inline uint8_t clampU8(int32_t value) { return static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, value))); }

inline HVX_Vector splatU8(uint8_t value) { return Q6_V_vsplat_R(Q6_R_vsplatb_R(value)); }

inline HVX_Vector loadVector(const uint8_t* ptr) { return q6op_V_vldu_A(ptr); }

inline void storeVector(uint8_t* ptr, HVX_Vector value) { q6op_vstu_AV(reinterpret_cast<HVX_Vector*>(ptr), value); }

inline uint8_t quantizedExponentCoefficient(float score_scale, float beta) {
  // SOLE's 1/ln(2) approximation: 1.4375 = 1 + 1/2 - 1/16.
  const int32_t q8 = static_cast<int32_t>(score_scale * beta * 1.4375f * 256.0f + 0.5f);
  return clampU8(std::max<int32_t>(1, q8));
}

inline uint32_t exactReciprocalQ23(uint32_t output_levels, uint32_t denominator) {
  // Divide-free reciprocal for the row normalization.  Normalize the positive
  // denominator to Q31 [0.5, 1), use the minimax linear seed
  //   1/x ~= 48/17 - (32/17)x,
  // then run three Q30 Newton-Raphson refinements.  The final remainder check
  // preserves the exact rounded integer result of
  //   ((output_levels << 23) + denominator / 2) / denominator.
  const int normalization_shift = __builtin_clz(denominator) - 1;
  const uint32_t normalized = denominator << normalization_shift;
  uint32_t reciprocal_q30 =
      kReciprocalQ30LinearA - static_cast<uint32_t>((uint64_t(kReciprocalQ30LinearB) * normalized + (1ull << 30)) >> 31);
#pragma unroll
  for (int iteration = 0; iteration < 3; ++iteration) {
    const uint32_t product_q30 = static_cast<uint32_t>((uint64_t(normalized) * reciprocal_q30 + (1ull << 30)) >> 31);
    const uint32_t correction_q30 = 0x80000000u - product_q30;
    reciprocal_q30 = static_cast<uint32_t>((uint64_t(reciprocal_q30) * correction_q30 + (1ull << 29)) >> 30);
  }

  const int quotient_shift = 38 - normalization_shift;
  uint32_t quotient =
      static_cast<uint32_t>((uint64_t(output_levels) * reciprocal_q30 + (1ull << (quotient_shift - 1))) >> quotient_shift);
  const uint64_t rounded_numerator = (uint64_t(output_levels) << kReciprocalFractionBits) + denominator / 2;
  uint64_t reconstructed = uint64_t(quotient) * denominator;
  // Three refinements bound the estimate to at most one quotient unit for
  // this op's denominator range (1 .. 1024 * 16384), so a single correction
  // is sufficient and prevents the compiler from recognizing a division loop.
  if (reconstructed > rounded_numerator) {
    --quotient;
  } else if (reconstructed + denominator <= rounded_numerator) {
    ++quotient;
  }
  return quotient;
}

inline HVX_Vector xorPermute(HVX_Vector value, uint8_t byte_offset) {
  // A single set bit in every vdelta control byte selects exactly one
  // butterfly stage. Offsets below 32 never cross a logical 32-byte row.
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
  // Each vrmpy reduces four U8 lanes. Masking/shifting the two bytes of each
  // Q14 halfword yields exact sums of two numerators per U32 lane without
  // unpacking the vector to scalar memory.
  const HVX_Vector low_bytes = Q6_V_vand_VV(values, Q6_V_vsplat_R(0x00ff00ff));
  const HVX_Vector high_bytes = Q6_Vuh_vlsr_VuhR(values, 8);
  const HVX_Vector low_sum = Q6_Vuw_vrmpy_VubRub(low_bytes, 0x01010101);
  const HVX_Vector high_sum = Q6_Vuw_vrmpy_VubRub(high_bytes, 0x01010101);
  return Q6_Vw_vadd_VwVw(low_sum, Q6_Vw_vasl_VwR(high_sum, 8));
}

inline HVX_Vector reduceSumWithinRows(HVX_Vector value) {
  // There are eight U32 partial sums in each 32-byte logical row.
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 4));
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 8));
  value = Q6_Vw_vadd_VwVw(value, xorPermute(value, 16));
  return value;
}

inline HVX_Vector validLaneCounts(const int32_t counts[kRowsPerVector]) {
  VectorStorage storage{};
  for (int row = 0; row < kRowsPerVector; ++row) {
    std::fill_n(storage.bytes + row * kDepthLanes, kDepthLanes, clampU8(counts[row]));
  }
  return storage.vector;
}

inline int probabilityTableOffset(int row, int code) {
  // In 128-byte mode vlut32 selects four interleaved 32-entry tables:
  // row 0/2 occupy low/high bytes of hwords 0..31; row 1/3 use 32..63.
  return (row & 1 ? 64 : 0) + 2 * code + (row >= 2 ? 1 : 0);
}

template<int Blocks, typename OutputTensor, typename ScoresTensor>
inline void runRegisterPrefix(OutputTensor& out, const ScoresTensor& scores, Idx batch, Idx height, Idx row_base, Idx depth,
                              int active_rows, const int32_t valid_lengths[kRowsPerVector], uint8_t output_zero,
                              uint32_t output_levels, uint8_t exponent_coefficient, HVX_Vector exponent_lut,
                              HVX_Vector lane_index, HVX_Vector row_index_base) {
  static_assert(Blocks == 1 || Blocks == 2, "Register prefix is specialized for one or two key blocks");

  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector voutput_zero = splatU8(output_zero);
  const HVX_Vector vmax_code = splatU8(kLargestExponentCode);
  const HVX_Vector vmasked_code = splatU8(kMaskedCode);
  const HVX_Vector vround_q8_h = Q6_Vh_vsplat_R(128);
  const HVX_VectorPair vround_q8 = Q6_W_vcombine_VV(vround_q8_h, vround_q8_h);

  HVX_Vector cached_scores[Blocks];
  HVX_VectorPred cached_valid[Blocks];
  HVX_Vector row_max = vzero;
#pragma unroll
  for (int block = 0; block < Blocks; ++block) {
    const Idx d = block * kDepthLanes;
    int32_t lane_counts[kRowsPerVector];
#pragma unroll
    for (int row = 0; row < kRowsPerVector; ++row) {
      lane_counts[row] = std::max<int32_t>(0, std::min<int32_t>(kDepthLanes, valid_lengths[row] - d));
    }
    cached_scores[block] = loadVector(scores.get_raw_addr(batch, height, row_base, d));
    cached_valid[block] = Q6_Q_vcmp_gt_VubVub(validLaneCounts(lane_counts), lane_index);
    row_max = Q6_Vub_vmax_VubVub(row_max, Q6_V_vmux_QVV(cached_valid[block], cached_scores[block], vzero));
  }
  row_max = reduceMaxWithinRows(row_max);

  HVX_Vector cached_codes[Blocks];
  HVX_Vector row_sum_vector = vzero;
#pragma unroll
  for (int block = 0; block < Blocks; ++block) {
    const HVX_Vector difference = Q6_Vub_vsub_VubVub_sat(row_max, cached_scores[block]);
    HVX_VectorPair product = Q6_Wuh_vmpy_VubRub(difference, Q6_R_vsplatb_R(exponent_coefficient));
    product = Q6_Wh_vadd_WhWh(product, vround_q8);
    HVX_Vector code = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(product), Q6_V_lo_W(product));
    code = Q6_Vub_vmin_VubVub(code, vmax_code);
    code = Q6_V_vmux_QVV(cached_valid[block], code, vmasked_code);
    cached_codes[block] = code;

    const HVX_VectorPair numerator = Q6_Wh_vlut16_VbVhR(code, exponent_lut, 0);
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

  VectorStorage probability_lut{};
  std::fill_n(probability_lut.bytes, kVectorBytes, output_zero);
  for (int row = 0; row < active_rows; ++row) {
    if (row_sum[row] == 0) { continue; }
    const uint32_t reciprocal_q23 = exactReciprocalQ23(output_levels, row_sum[row]);
    for (int code = 0; code <= kLargestExponentCode; ++code) {
      const uint32_t probability = (uint32_t(kExponentNumerator[code]) * reciprocal_q23 + (1u << (kReciprocalFractionBits - 1)))
                                   >> kReciprocalFractionBits;
      probability_lut.bytes[probabilityTableOffset(row, code)] = clampU8(static_cast<int32_t>(probability) + output_zero);
    }
  }

#pragma unroll
  for (int block = 0; block < Blocks; ++block) {
    const HVX_Vector index = Q6_V_vor_VV(cached_codes[block], row_index_base);
    HVX_Vector probability = Q6_Vb_vlut32_VbVbR(index, probability_lut.vector, 0);
    probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 1);
    probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 2);
    probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 3);
    storeVector(out.get_raw_addr(batch, height, row_base, block * kDepthLanes), probability);
  }
  for (Idx d = Blocks * kDepthLanes; d < depth; d += kDepthLanes) {
    storeVector(out.get_raw_addr(batch, height, row_base, d), voutput_zero);
  }
}

}  // namespace

template<typename OutputTensor, typename ScoresTensor, typename PositionsTensor>
SoftmaxStatus runVtcmCausalE2SoftmaxHd128(OutputTensor& out, const ScoresTensor& scores, const PositionsTensor& positions,
                                          uint32_t slice_number, uint32_t num_slices) {
  out.set_dims(scores);
  if (num_slices == 0 || slice_number >= num_slices) { return SoftmaxStatus::ErrorDimensions; }
  if (scores.dim(3) == 0 || scores.dim(3) % kDepthLanes != 0 || positions.dim(3) != 1
      || (positions.dim(0) != 1 && positions.dim(0) != scores.dim(0))
      || (positions.dim(1) != 1 && positions.dim(1) != scores.dim(1))
      || (positions.dim(2) != 1 && positions.dim(2) != scores.dim(2))) {
    return SoftmaxStatus::ErrorDimensions;
  }

  const auto [batches, heights, rows, depth] = scores.dims();
  const uint8_t output_zero = clampU8(out.interface_offset());
  const int32_t output_levels_i = static_cast<int32_t>(out.interface_scale_recip() + 0.5f);
  const uint32_t output_levels = static_cast<uint32_t>(std::max<int32_t>(1, std::min<int32_t>(255, output_levels_i)));
  const uint8_t exponent_coefficient = quantizedExponentCoefficient(scores.interface_scale(), kHeadDim128AttentionBeta);

  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector voutput_zero = splatU8(output_zero);
  const HVX_Vector vmax_code = splatU8(kLargestExponentCode);
  const HVX_Vector vmasked_code = splatU8(kMaskedCode);
  const HVX_Vector vround_q8_h = Q6_Vh_vsplat_R(128);
  const HVX_VectorPair vround_q8 = Q6_W_vcombine_VV(vround_q8_h, vround_q8_h);

  VectorStorage exponent_lut{};
  for (int code = 0; code <= kMaskedCode; ++code) {
    // Rt=0 selects the low halfword of each word for indices 0..15.
    exponent_lut.hwords[2 * code] = kExponentNumerator[code];
  }

  VectorStorage row_index_base{};
  VectorStorage lane_index{};
  for (int row = 0; row < kRowsPerVector; ++row) {
    std::fill_n(row_index_base.bytes + row * kDepthLanes, kDepthLanes, static_cast<uint8_t>(row * 32));
    for (int lane = 0; lane < kDepthLanes; ++lane) { lane_index.bytes[row * kDepthLanes + lane] = static_cast<uint8_t>(lane); }
  }

  uint32_t row_group = 0;
  for (Idx batch = 0; batch < batches; ++batch) {
    for (Idx height = 0; height < heights; ++height) {
      for (Idx row_base = 0; row_base < rows; row_base += kRowsPerVector) {
        if (row_group++ % num_slices != slice_number) { continue; }
        const int active_rows = static_cast<int>(std::min<Idx>(kRowsPerVector, rows - row_base));
        int32_t valid_lengths[kRowsPerVector] = {0, 0, 0, 0};
        int32_t maximum_valid_length = 0;
        for (int row = 0; row < active_rows; ++row) {
          const Idx position_batch = positions.dim(0) == 1 ? 0 : batch;
          const Idx position_height = positions.dim(1) == 1 ? 0 : height;
          const Idx position_row = positions.dim(2) == 1 ? 0 : row_base + row;
          const int32_t position = *positions.get_raw_addr(position_batch, position_height, position_row, 0);
          valid_lengths[row] = std::max<int32_t>(0, std::min<int32_t>(depth, position + 1));
          maximum_valid_length = std::max(maximum_valid_length, valid_lengths[row]);
        }
        const Idx active_depth = static_cast<Idx>((maximum_valid_length + kDepthLanes - 1) / kDepthLanes * kDepthLanes);

        // Real decode/prefill prefixes normally occupy one or two 32-key
        // blocks. Keep their scores and exponent codes in vector registers so
        // the only VTCM traffic is one score read and the final probability
        // write. Longer prefixes retain the general three-pass path below.
        if (active_depth == kDepthLanes) {
          runRegisterPrefix<1>(out, scores, batch, height, row_base, depth, active_rows, valid_lengths, output_zero,
                               output_levels, exponent_coefficient, exponent_lut.vector, lane_index.vector,
                               row_index_base.vector);
          continue;
        }
        // Pass 1: vector max across depth followed by five butterfly stages
        // that broadcast each 32-byte row maximum within its segment.
        HVX_Vector row_max = vzero;
        for (Idx d = 0; d < active_depth; d += kDepthLanes) {
          const HVX_Vector score = loadVector(scores.get_raw_addr(batch, height, row_base, d));
          int32_t lane_counts[kRowsPerVector];
          for (int row = 0; row < kRowsPerVector; ++row) {
            lane_counts[row] = std::max<int32_t>(0, std::min<int32_t>(kDepthLanes, valid_lengths[row] - d));
          }
          const HVX_VectorPred valid = Q6_Q_vcmp_gt_VubVub(validLaneCounts(lane_counts), lane_index.vector);
          row_max = Q6_Vub_vmax_VubVub(row_max, Q6_V_vmux_QVV(valid, score, vzero));
        }
        row_max = reduceMaxWithinRows(row_max);

        // Pass 2: generate/stage U8 exponent codes and accumulate the exact
        // Q14 denominator entirely in vector registers.
        HVX_Vector row_sum_vector = vzero;
        for (Idx d = 0; d < active_depth; d += kDepthLanes) {
          const HVX_Vector score = loadVector(scores.get_raw_addr(batch, height, row_base, d));
          auto* code_ptr = out.get_raw_addr(batch, height, row_base, d);

          int32_t lane_counts[kRowsPerVector];
          for (int row = 0; row < kRowsPerVector; ++row) {
            lane_counts[row] = std::max<int32_t>(0, std::min<int32_t>(kDepthLanes, valid_lengths[row] - d));
          }
          const HVX_VectorPred valid = Q6_Q_vcmp_gt_VubVub(validLaneCounts(lane_counts), lane_index.vector);

          const HVX_Vector difference = Q6_Vub_vsub_VubVub_sat(row_max, score);
          HVX_VectorPair product = Q6_Wuh_vmpy_VubRub(difference, Q6_R_vsplatb_R(exponent_coefficient));
          product = Q6_Wh_vadd_WhWh(product, vround_q8);
          HVX_Vector code = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(product), Q6_V_lo_W(product));
          code = Q6_Vub_vmin_VubVub(code, vmax_code);
          code = Q6_V_vmux_QVV(valid, code, vmasked_code);
          storeVector(code_ptr, code);

          const HVX_VectorPair numerator = Q6_Wh_vlut16_VbVhR(code, exponent_lut.vector, 0);
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

        // Build four row-specific 32-entry byte tables in one HVX vector.
        // Only four vlut32 selections are then needed for every 128 codes.
        VectorStorage probability_lut{};
        std::fill_n(probability_lut.bytes, kVectorBytes, output_zero);
        for (int row = 0; row < active_rows; ++row) {
          if (row_sum[row] == 0) { continue; }
          const uint32_t reciprocal_q23 = exactReciprocalQ23(output_levels, row_sum[row]);
          for (int code = 0; code <= kLargestExponentCode; ++code) {
            const uint32_t probability =
                (uint32_t(kExponentNumerator[code]) * reciprocal_q23 + (1u << (kReciprocalFractionBits - 1)))
                >> kReciprocalFractionBits;
            probability_lut.bytes[probabilityTableOffset(row, code)] = clampU8(static_cast<int32_t>(probability) + output_zero);
          }
        }

        for (Idx d = 0; d < active_depth; d += kDepthLanes) {
          auto* output_ptr = out.get_raw_addr(batch, height, row_base, d);
          const HVX_Vector code = loadVector(output_ptr);
          const HVX_Vector index = Q6_V_vor_VV(code, row_index_base.vector);
          HVX_Vector probability = Q6_Vb_vlut32_VbVbR(index, probability_lut.vector, 0);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 1);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 2);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 3);
          storeVector(output_ptr, probability);
        }
        for (Idx d = active_depth; d < depth; d += kDepthLanes) {
          storeVector(out.get_raw_addr(batch, height, row_base, d), voutput_zero);
        }
      }
    }
  }
  return SoftmaxStatus::Success;
}

#ifndef MLLM_VTCM_SOFTMAX_QHPI_TRANSLATION_UNIT
GraphStatus vtcmCausalE2SoftmaxHd128(QUint8CroutonTensor_TCM& out, const QUint8CroutonTensor_TCM& scores,
                                     const Int32Tensor_TCM& positions) {
  const SoftmaxStatus status = runVtcmCausalE2SoftmaxHd128(out, scores, positions, 0, 1);
  return status == SoftmaxStatus::Success ? GraphStatus::Success : GraphStatus::ErrorDimensions;
}
#else

namespace {

class QhpiCroutonU8 final {
 public:
  explicit QhpiCroutonU8(const QHPI_Tensor* tensor)
      : shape_(qhpi_tensor_shape(tensor)),
        blocks_(reinterpret_cast<uint8_t**>(qhpi_tensor_block_table(tensor))),
        quant_(qhpi_tensor_quant_parameters(tensor)) {
    depth_blocks_ = (shape_.dims[3] + 31) / 32;
    width_blocks_ = depth_blocks_ * ((shape_.dims[2] + 7) / 8);
    height_blocks_ = width_blocks_ * ((shape_.dims[1] + 7) / 8);
  }

  template<typename OtherTensor>
  void set_dims(const OtherTensor&) {}

  Idx dim(int axis) const { return static_cast<Idx>(shape_.dims[axis]); }

  auto dims() const { return std::tuple<Idx, Idx, Idx, Idx>{dim(0), dim(1), dim(2), dim(3)}; }

  float interface_scale() const { return quant_.stepsize; }
  float interface_scale_recip() const { return 1.0f / quant_.stepsize; }
  int32_t interface_offset() const { return quant_.zero_offset; }

  uint8_t* get_raw_addr(Idx batch, Idx height, Idx width, Idx depth) const {
    const uint32_t block = static_cast<uint32_t>(batch) * height_blocks_ + (static_cast<uint32_t>(height) / 8) * width_blocks_
                           + (static_cast<uint32_t>(width) / 8) * depth_blocks_ + static_cast<uint32_t>(depth) / 32;
    const uint32_t offset = (static_cast<uint32_t>(height) % 8) * 8 * 32 + (static_cast<uint32_t>(width) % 8) * 32
                            + static_cast<uint32_t>(depth) % 32;
    return blocks_[block] + offset;
  }

 private:
  QHPI_Shape shape_{};
  uint8_t** blocks_ = nullptr;
  QHPI_Quant_Parameters quant_{};
  uint32_t depth_blocks_ = 0;
  uint32_t width_blocks_ = 0;
  uint32_t height_blocks_ = 0;
};

class QhpiFlatInt32 final {
 public:
  explicit QhpiFlatInt32(const QHPI_Tensor* tensor)
      : shape_(qhpi_tensor_shape(tensor)), data_(static_cast<int32_t*>(qhpi_tensor_raw_data(tensor))) {
    multipliers_[2] = shape_.dims[3];
    multipliers_[1] = multipliers_[2] * shape_.dims[2];
    multipliers_[0] = multipliers_[1] * shape_.dims[1];
  }

  Idx dim(int axis) const { return static_cast<Idx>(shape_.dims[axis]); }

  int32_t* get_raw_addr(Idx batch, Idx height, Idx width, Idx depth) const {
    const uint32_t offset = static_cast<uint32_t>(batch) * multipliers_[0] + static_cast<uint32_t>(height) * multipliers_[1]
                            + static_cast<uint32_t>(width) * multipliers_[2] + static_cast<uint32_t>(depth);
    return data_ + offset;
  }

 private:
  QHPI_Shape shape_{};
  int32_t* data_ = nullptr;
  uint32_t multipliers_[3]{};
};

uint32_t vtcmCausalE2SoftmaxHd128Mt(QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs,
                                    uint32_t num_inputs, const QHPI_Tensor* const* inputs) {
  if (num_outputs != 1 || num_inputs != 2 || outputs == nullptr || inputs == nullptr) { return QHPI_ERROR_FATAL; }
  QhpiCroutonU8 output(outputs[0]);
  const QhpiCroutonU8 scores(inputs[0]);
  const QhpiFlatInt32 positions(inputs[1]);
  const SoftmaxStatus status =
      runVtcmCausalE2SoftmaxHd128(output, scores, positions, qhpi_slice_number(handle), qhpi_num_slices(handle));
  return status == SoftmaxStatus::Success ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
}

QHPI_Tensor_Signature_v1 mt_input_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_INT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Tensor_Signature_v1 mt_output_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 mt_kernels[] = {
    {.function_name = "vtcmCausalE2SoftmaxHd128Mt",
     .function = vtcmCausalE2SoftmaxHd128Mt,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 2,
     .input_signature = mt_input_signatures,
     .min_outputs = 1,
     .output_signature = mt_output_signatures,
     .cost_function = nullptr,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr,
     .Reserved_1 = nullptr,
     .Reserved_2 = nullptr,
     .Reserved_3 = nullptr,
     .Reserved_4 = nullptr},
};

QHPI_OpInfo_v1 mt_ops[] = {
    {.name = THIS_PKG_NAME_STR "::VtcmCausalE2SoftmaxHd128Mt",
     .num_kernels = 1,
     .kernels = mt_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .tile_output = 0,
     .build_tile = nullptr,
     .late_rewrite = nullptr,
     .Reserved_1 = nullptr,
     .Reserved_2 = nullptr,
     .Reserved_3 = nullptr,
     .Reserved_4 = nullptr},
};

}  // namespace

extern "C" const char* qhpi_init() {
  qhpi_register_ops_v1(1, mt_ops, THIS_PKG_NAME_STR);
  return THIS_PKG_NAME_STR;
}
#endif

#ifndef MLLM_VTCM_SOFTMAX_QHPI_TRANSLATION_UNIT
// No MainMemory or generic Tensor registration: failed TCM placement must
// abort graph finalization instead of silently introducing DRAM traffic.
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((vtcmCausalE2SoftmaxHd128), "VtcmCausalE2SoftmaxHd128", FAST, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("VtcmCausalE2SoftmaxHd128", "scores", "positions"), Crouton("*", "scores"),
                      Tcm("*", "scores", "positions"))

END_PKG_OP_DEFINITION(PKG_VtcmCausalE2SoftmaxHd128);
#endif
