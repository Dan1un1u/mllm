// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// TCM-only, U8-in/U8-out masked E2Softmax. The integer approximation and
// rounding contract are unchanged from the first zero-DRAM experiment; this
// version replaces its scalar row scans and compare/mux output cascade with
// HVX reductions plus vlut16/vlut32 register lookup tables.

#include <algorithm>
#include <cstdint>

#include "HTP/core/constraints.h"
#include "HTP/core/intrinsics.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

#include <hexagon_types.h>

BEGIN_PKG_OP_DEFINITION(PKG_VtcmMaskedE2Softmax);

namespace {

constexpr int kDepthLanes = 32;
constexpr int kVectorBytes = sizeof(HVX_Vector);
constexpr int kRowsPerVector = kVectorBytes / kDepthLanes;
constexpr uint8_t kMaskedCode = 15;
constexpr uint8_t kLargestExponentCode = 14;
constexpr uint32_t kReciprocalFractionBits = 23;

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

inline uint8_t clampU8(int32_t value) {
  return static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, value)));
}

inline HVX_Vector splatU8(uint8_t value) { return Q6_V_vsplat_R(Q6_R_vsplatb_R(value)); }

inline HVX_Vector loadVector(const uint8_t* ptr) { return q6op_V_vldu_A(ptr); }

inline void storeVector(uint8_t* ptr, HVX_Vector value) {
  q6op_vstu_AV(reinterpret_cast<HVX_Vector*>(ptr), value);
}

inline uint8_t quantizedExponentCoefficient(float score_scale) {
  // SOLE's 1/ln(2) approximation: 1.4375 = 1 + 1/2 - 1/16.
  const int32_t q8 = static_cast<int32_t>(score_scale * 1.4375f * 256.0f + 0.5f);
  return clampU8(std::max<int32_t>(1, q8));
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

inline int probabilityTableOffset(int row, int code) {
  // In 128-byte mode vlut32 selects four interleaved 32-entry tables:
  // row 0/2 occupy low/high bytes of hwords 0..31; row 1/3 use 32..63.
  return (row & 1 ? 64 : 0) + 2 * code + (row >= 2 ? 1 : 0);
}

}  // namespace

GraphStatus vtcmMaskedE2Softmax(QUint8CroutonTensor_TCM& out, const QUint8CroutonTensor_TCM& scores,
                                const QUint8CroutonTensor_TCM& mask) {
  out.set_dims(scores);
  if (scores.dims() != mask.dims() || scores.dim(3) == 0 || scores.dim(3) % kDepthLanes != 0) {
    return GraphStatus::ErrorDimensions;
  }

  const auto [batches, heights, rows, depth] = scores.dims();
  const uint8_t mask_zero = clampU8(mask.interface_offset());
  const uint8_t output_zero = clampU8(out.interface_offset());
  const int32_t output_levels_i = static_cast<int32_t>(out.interface_scale_recip() + 0.5f);
  const uint32_t output_levels =
      static_cast<uint32_t>(std::max<int32_t>(1, std::min<int32_t>(255, output_levels_i)));
  const uint8_t exponent_coefficient = quantizedExponentCoefficient(scores.interface_scale());

  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector vmask_zero = splatU8(mask_zero);
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
  for (int row = 0; row < kRowsPerVector; ++row) {
    std::fill_n(row_index_base.bytes + row * kDepthLanes, kDepthLanes, static_cast<uint8_t>(row * 32));
  }

  for (Idx batch = 0; batch < batches; ++batch) {
    for (Idx height = 0; height < heights; ++height) {
      for (Idx row_base = 0; row_base < rows; row_base += kRowsPerVector) {
        const int active_rows = static_cast<int>(std::min<Idx>(kRowsPerVector, rows - row_base));

        // Pass 1: vector max across depth followed by five butterfly stages
        // that broadcast each 32-byte row maximum within its segment.
        HVX_Vector row_max = vzero;
        for (Idx d = 0; d < depth; d += kDepthLanes) {
          const HVX_Vector score = loadVector(scores.get_raw_addr(batch, height, row_base, d));
          const HVX_Vector mask_value = loadVector(mask.get_raw_addr(batch, height, row_base, d));
          const HVX_VectorPred valid = Q6_Q_vcmp_eq_VbVb(mask_value, vmask_zero);
          row_max = Q6_Vub_vmax_VubVub(row_max, Q6_V_vmux_QVV(valid, score, vzero));
        }
        row_max = reduceMaxWithinRows(row_max);

        // Pass 2: generate/stage U8 exponent codes and accumulate the exact
        // Q14 denominator entirely in vector registers.
        HVX_Vector row_sum_vector = vzero;
        for (Idx d = 0; d < depth; d += kDepthLanes) {
          const HVX_Vector score = loadVector(scores.get_raw_addr(batch, height, row_base, d));
          const HVX_Vector mask_value = loadVector(mask.get_raw_addr(batch, height, row_base, d));
          auto* code_ptr = out.get_raw_addr(batch, height, row_base, d);

          const HVX_Vector difference = Q6_Vub_vsub_VubVub_sat(row_max, score);
          HVX_VectorPair product = Q6_Wuh_vmpy_VubRub(difference, Q6_R_vsplatb_R(exponent_coefficient));
          product = Q6_Wh_vadd_WhWh(product, vround_q8);
          HVX_Vector code = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(product), Q6_V_lo_W(product));
          code = Q6_Vub_vmin_VubVub(code, vmax_code);
          const HVX_VectorPred valid = Q6_Q_vcmp_eq_VbVb(mask_value, vmask_zero);
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
          const uint32_t reciprocal_q23 =
              ((output_levels << kReciprocalFractionBits) + row_sum[row] / 2) / row_sum[row];
          for (int code = 0; code <= kLargestExponentCode; ++code) {
            const uint32_t probability =
                (static_cast<uint32_t>(kExponentNumerator[code]) * reciprocal_q23
                 + (1u << (kReciprocalFractionBits - 1)))
                >> kReciprocalFractionBits;
            probability_lut.bytes[probabilityTableOffset(row, code)] =
                clampU8(static_cast<int32_t>(probability) + output_zero);
          }
        }

        for (Idx d = 0; d < depth; d += kDepthLanes) {
          auto* output_ptr = out.get_raw_addr(batch, height, row_base, d);
          const HVX_Vector code = loadVector(output_ptr);
          const HVX_Vector index = Q6_V_vor_VV(code, row_index_base.vector);
          HVX_Vector probability = Q6_Vb_vlut32_VbVbR(index, probability_lut.vector, 0);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 1);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 2);
          probability = Q6_Vb_vlut32or_VbVbVbR(probability, index, probability_lut.vector, 3);
          storeVector(output_ptr, probability);
        }
      }
    }
  }
  return GraphStatus::Success;
}

// No MainMemory or generic Tensor registration: failed TCM placement must
// abort graph finalization instead of silently introducing DRAM traffic.
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((vtcmMaskedE2Softmax), "VtcmMaskedE2Softmax", FAST, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("VtcmMaskedE2Softmax", "scores", "mask"), Crouton("*", "scores", "mask"),
                      Tcm("*", "scores", "mask"))

END_PKG_OP_DEFINITION(PKG_VtcmMaskedE2Softmax);
