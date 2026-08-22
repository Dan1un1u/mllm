// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// TCM-only, U8-in/U8-out masked Softmax derived from SOLE E2Softmax.
//
// The implementation deliberately keeps the QK -> Softmax -> PV edge in
// Crouton VTCM. It uses the paper's integer base-2 exponent code
//
//   code = round((row_max - score) * 1.4375)
//
// clipped to fourteen bits of dynamic range. Code 15 is reserved for a
// mathematically exact causal-mask zero, avoiding the probability leakage
// that would result from treating a masked element as 2^-15. The output
// tensor is also used as the exponent-code staging buffer, so the op needs no
// external scratch tensor and cannot spill an intermediate to DRAM.

#include <algorithm>
#include <cstdint>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/intrinsics.h"

#include <hexagon_types.h>

BEGIN_PKG_OP_DEFINITION(PKG_VtcmMaskedE2Softmax);

namespace {

constexpr int kDepthLanes = 32;
constexpr int kVectorBytes = sizeof(HVX_Vector);
constexpr int kRowsPerVector = kVectorBytes / kDepthLanes;
constexpr uint8_t kMaskedCode = 15;
constexpr uint8_t kLargestExponentCode = 14;
constexpr uint32_t kReciprocalFractionBits = 23;

static_assert(kVectorBytes == 128, "The Crouton row-vector kernel requires 128-byte HVX");
static_assert(kRowsPerVector == 4);

// Q14 numerators for 2^-code. Code 15 is the exact masked zero extension to
// the paper's 4-bit exponent representation.
constexpr uint16_t kExponentNumerator[16] = {
    16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0,
};

union alignas(128) VectorBytes {
  HVX_Vector vector;
  uint8_t bytes[kVectorBytes];
};

inline uint8_t clampU8(int32_t value) { return static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, value))); }

inline HVX_Vector splatU8(uint8_t value) { return Q6_V_vsplat_R(Q6_R_vsplatb_R(value)); }

inline HVX_Vector loadVector(const uint8_t* ptr) { return q6op_V_vldu_A(ptr); }

inline void storeVector(uint8_t* ptr, HVX_Vector value) { q6op_vstu_AV(reinterpret_cast<HVX_Vector*>(ptr), value); }

inline uint8_t quantizedExponentCoefficient(float score_scale) {
  // SOLE replaces 1/ln(2) with 1.4375 = 1 + 1/2 - 1/16. A Q8
  // coefficient lets HVX convert all 128 U8 differences in one multiply.
  const int32_t q8 = static_cast<int32_t>(score_scale * 1.4375f * 256.0f + 0.5f);
  return clampU8(std::max<int32_t>(1, q8));
}

}  // namespace

GraphStatus vtcmMaskedE2Softmax(QUint8CroutonTensor_TCM& out, const QUint8CroutonTensor_TCM& scores,
                                const QUint8CroutonTensor_TCM& mask) {
  out.set_dims(scores);
  if (scores.dims() != mask.dims() || scores.dim(3) == 0 || scores.dim(3) % kDepthLanes != 0) {
    return GraphStatus::ErrorDimensions;
  }

  const auto [batches, heights, rows, depth] = scores.dims();
  // HTP tensor interfaces expose the encoded U8 zero point directly (the
  // offset used in q = round(real / scale) + offset), not QNN's serialized
  // negative scale-offset field. The causal mask's valid value is therefore
  // raw 255 for the accepted model encoding.
  const uint8_t mask_zero = clampU8(mask.interface_offset());
  const uint8_t output_zero = clampU8(out.interface_offset());
  const int32_t output_levels_i = static_cast<int32_t>(out.interface_scale_recip() + 0.5f);
  const uint32_t output_levels = static_cast<uint32_t>(std::max<int32_t>(1, std::min<int32_t>(255, output_levels_i)));
  const uint8_t exponent_coefficient = quantizedExponentCoefficient(scores.interface_scale());

  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector vmask_zero = splatU8(mask_zero);
  const HVX_Vector vmax_code = splatU8(kLargestExponentCode);
  const HVX_Vector vmasked_code = splatU8(kMaskedCode);
  const HVX_Vector vround_q8_h = Q6_Vh_vsplat_R(128);
  const HVX_VectorPair vround_q8 = Q6_W_vcombine_VV(vround_q8_h, vround_q8_h);

  for (Idx batch = 0; batch < batches; ++batch) {
    for (Idx height = 0; height < heights; ++height) {
      for (Idx row_base = 0; row_base < rows; row_base += kRowsPerVector) {
        const int active_rows = static_cast<int>(std::min<Idx>(kRowsPerVector, rows - row_base));

        // Pass 1: four independent row maxima are accumulated in a single
        // HVX vector. Each 32-byte segment is one logical Crouton row.
        HVX_Vector vmax_lanes = vzero;
        for (Idx d = 0; d < depth; d += kDepthLanes) {
          const auto* score_ptr = scores.get_raw_addr(batch, height, row_base, d);
          const auto* mask_ptr = mask.get_raw_addr(batch, height, row_base, d);
          const HVX_Vector score_vector = loadVector(score_ptr);
          const HVX_Vector mask_vector = loadVector(mask_ptr);
          const HVX_VectorPred valid = Q6_Q_vcmp_eq_VbVb(mask_vector, vmask_zero);
          const HVX_Vector valid_score = Q6_V_vmux_QVV(valid, score_vector, vzero);
          vmax_lanes = Q6_Vub_vmax_VubVub(vmax_lanes, valid_score);
        }

        VectorBytes max_lanes{};
        max_lanes.vector = vmax_lanes;
        VectorBytes row_max_vector{};
        for (int row = 0; row < active_rows; ++row) {
          uint8_t value = 0;
          for (int lane = 0; lane < kDepthLanes; ++lane) { value = std::max(value, max_lanes.bytes[row * kDepthLanes + lane]); }
          std::fill_n(row_max_vector.bytes + row * kDepthLanes, kDepthLanes, value);
        }

        // Pass 2: generate the paper's 4-bit base-2 exponent code with HVX,
        // stage it in the U8 output, and accumulate one exact integer sum per
        // row. Reserving code 15 for mask means masked lanes add exactly 0.
        uint32_t row_sum[kRowsPerVector] = {};
        for (Idx d = 0; d < depth; d += kDepthLanes) {
          const auto* score_ptr = scores.get_raw_addr(batch, height, row_base, d);
          const auto* mask_ptr = mask.get_raw_addr(batch, height, row_base, d);
          auto* code_ptr = out.get_raw_addr(batch, height, row_base, d);

          const HVX_Vector score_vector = loadVector(score_ptr);
          const HVX_Vector mask_vector = loadVector(mask_ptr);
          const HVX_Vector difference = Q6_Vub_vsub_VubVub_sat(row_max_vector.vector, score_vector);
          HVX_VectorPair product = Q6_Wuh_vmpy_VubRub(difference, Q6_R_vsplatb_R(exponent_coefficient));
          product = Q6_Wh_vadd_WhWh(product, vround_q8);
          HVX_Vector code = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(product), Q6_V_lo_W(product));
          code = Q6_Vub_vmin_VubVub(code, vmax_code);

          const HVX_VectorPred valid = Q6_Q_vcmp_eq_VbVb(mask_vector, vmask_zero);
          code = Q6_V_vmux_QVV(valid, code, vmasked_code);
          storeVector(code_ptr, code);

          VectorBytes code_bytes{};
          code_bytes.vector = code;
          for (int row = 0; row < active_rows; ++row) {
            const uint8_t* row_codes = code_bytes.bytes + row * kDepthLanes;
            uint32_t partial_sum = 0;
            for (int lane = 0; lane < kDepthLanes; ++lane) { partial_sum += kExponentNumerator[row_codes[lane]]; }
            row_sum[row] += partial_sum;
          }
        }

        // Build the fifteen row-specific output levels once. Pass 3 applies
        // them with vector compares/muxes, avoiding a scalar divide per token.
        VectorBytes probability_by_code[15]{};
        for (int row = 0; row < active_rows; ++row) {
          if (row_sum[row] == 0) {
            for (int code = 0; code <= kLargestExponentCode; ++code) {
              std::fill_n(probability_by_code[code].bytes + row * kDepthLanes, kDepthLanes, output_zero);
            }
            continue;
          }

          const uint32_t reciprocal_q23 = ((output_levels << kReciprocalFractionBits) + row_sum[row] / 2) / row_sum[row];
          for (int code = 0; code <= kLargestExponentCode; ++code) {
            const uint32_t probability =
                (static_cast<uint32_t>(kExponentNumerator[code]) * reciprocal_q23 + (1u << (kReciprocalFractionBits - 1)))
                >> kReciprocalFractionBits;
            const uint8_t quantized = clampU8(static_cast<int32_t>(probability) + output_zero);
            std::fill_n(probability_by_code[code].bytes + row * kDepthLanes, kDepthLanes, quantized);
          }
        }

        for (Idx d = 0; d < depth; d += kDepthLanes) {
          auto* output_ptr = out.get_raw_addr(batch, height, row_base, d);
          const HVX_Vector code = loadVector(output_ptr);
          HVX_Vector probability = splatU8(output_zero);
          for (int exponent = 0; exponent <= kLargestExponentCode; ++exponent) {
            const HVX_VectorPred selected = Q6_Q_vcmp_eq_VbVb(code, splatU8(exponent));
            probability = Q6_V_vmux_QVV(selected, probability_by_code[exponent].vector, probability);
          }
          storeVector(output_ptr, probability);
        }
      }
    }
  }
  return GraphStatus::Success;
}

// Deliberately register no MainMemory or generic Tensor implementation. A
// placement failure must abort graph finalization rather than silently insert
// a DRAM-backed fallback kernel.
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((vtcmMaskedE2Softmax), "VtcmMaskedE2Softmax", FAST, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("VtcmMaskedE2Softmax", "scores", "mask"), Crouton("*", "scores", "mask"), Tcm("*", "scores", "mask"))

END_PKG_OP_DEFINITION(PKG_VtcmMaskedE2Softmax);
