// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>

#include "HTP/core/qhpi.h"
#include "HmxInt8Tile.hpp"

#if !defined(REFERENCE_OP)
#include <hexagon_protos.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

namespace probe = mllm::qnn::qhpi_hmx_probe;

namespace {

#if !defined(REFERENCE_OP)
constexpr uint32_t kWeightReady = 2;
constexpr uint32_t kFirstHmxReady = 3;
constexpr uint32_t kHvxReady = 4;
constexpr uint32_t kSecondHmxReady = 5;
constexpr uint32_t kSpinLimit = 200000000;
// The final packed weight word is reserved as a TCM-resident phase word. The
// host fixes the four corresponding logical coefficients to zero, so QNN
// presents its initial physical value as four U8 zero-points (0x80808080).
// This avoids QHPI's sync block, which QAIRT 2.49 serializes as a DDR tensor.
constexpr uint32_t kInitialPhase = 0x80808080U;
constexpr uint32_t kPhaseWordOffset = probe::kWeightBytes - sizeof(uint32_t);
#endif

bool shapeEquals(const QHPI_Shape& shape, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
  return shape.rank == 4 && shape.dims[0] == d0 && shape.dims[1] == d1 && shape.dims[2] == d2 && shape.dims[3] == d3;
}

bool exactQuantization(const QHPI_Tensor* tensor, int32_t zero_offset) {
  const auto quant = qhpi_tensor_quant_parameters(tensor);
  return quant.zero_offset == zero_offset && quant.stepsize == 1.0F;
}

#if !defined(REFERENCE_OP)
bool aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr && (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void publishPhase(uint32_t* phase_word, uint32_t phase) { __atomic_store_n(phase_word, phase, __ATOMIC_RELEASE); }

bool waitForPhase(const uint32_t* phase_word, uint32_t phase) {
  for (uint32_t spin = 0; spin < kSpinLimit; ++spin) {
    if (__atomic_load_n(phase_word, __ATOMIC_ACQUIRE) == phase) return true;
    asm volatile("nop" ::: "memory");
  }
  return false;
}

void packSignedWeightHvx(const uint8_t* encoded_weight, int8_t* signed_weight) {
  const HVX_Vector sign_flip = Q6_V_vsplat_R(0x80808080);
  for (uint32_t offset = 0; offset < probe::kWeightBytes; offset += sizeof(HVX_Vector)) {
    const HVX_Vector encoded = *reinterpret_cast<const HVX_Vector*>(encoded_weight + offset);
    *reinterpret_cast<HVX_Vector*>(signed_weight + offset) = Q6_V_vxor_VV(encoded, sign_flip);
  }
  *reinterpret_cast<uint32_t*>(signed_weight + kPhaseWordOffset) = 0;
}

void addOneHvx(uint8_t* data) {
  const HVX_Vector one = Q6_V_vsplat_R(0x01010101);
  for (uint32_t offset = 0; offset < probe::kOutputBytes; offset += sizeof(HVX_Vector)) {
    const HVX_Vector input = *reinterpret_cast<const HVX_Vector*>(data + offset);
    *reinterpret_cast<HVX_Vector*>(data + offset) = Q6_Vub_vadd_VubVub_sat(input, one);
  }
}
#endif

uint32_t mixedResourceHmxHvxHmx(QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs,
                                uint32_t num_inputs, const QHPI_Tensor* const* inputs) {
  if (handle == nullptr || num_inputs != 2 || num_outputs != 1 || inputs == nullptr || outputs == nullptr
      || inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return QHPI_ERROR_FATAL;
  }
  if (!shapeEquals(qhpi_tensor_shape(inputs[0]), 1, 8, 8, 32)
      || !shapeEquals(qhpi_tensor_shape(inputs[1]), 1, 1, 32, 32)
      || !shapeEquals(qhpi_tensor_shape(outputs[0]), 1, 8, 8, 32) || !exactQuantization(inputs[0], 128)
      || !exactQuantization(inputs[1], 128) || !exactQuantization(outputs[0], 0)
      || qhpi_tensor_block_table_length(inputs[0]) != 1 || qhpi_tensor_block_table_length(outputs[0]) != 1) {
    return QHPI_ERROR_FATAL;
  }

#if defined(REFERENCE_OP)
  return QHPI_UNSUPPORTED;
#else
  auto** activation_blocks = qhpi_tensor_block_table(inputs[0]);
  auto** output_blocks = qhpi_tensor_block_table(outputs[0]);
  auto* encoded_weight = static_cast<uint8_t*>(qhpi_tensor_raw_data(inputs[1]));
  if (activation_blocks == nullptr || output_blocks == nullptr || !aligned(activation_blocks[0], 2048)
      || !aligned(output_blocks[0], 2048) || !aligned(encoded_weight, 128)) {
    return QHPI_ERROR_FATAL;
  }

  auto* activation_storage = static_cast<uint8_t*>(activation_blocks[0]);
  auto* output_storage = static_cast<uint8_t*>(output_blocks[0]);
  auto* phase_word = reinterpret_cast<uint32_t*>(encoded_weight + kPhaseWordOffset);
  auto* first_weight = reinterpret_cast<int8_t*>(output_storage);
  auto* first_bias = reinterpret_cast<uint32_t*>(output_storage + probe::kWeightBytes);
  auto* second_weight = reinterpret_cast<int8_t*>(activation_storage);
  auto* second_bias = reinterpret_cast<uint32_t*>(activation_storage + probe::kWeightBytes);
  if (!aligned(first_weight, 128) || !aligned(first_bias, 256) || !aligned(second_weight, 128)
      || !aligned(second_bias, 256) || !aligned(phase_word, alignof(uint32_t))) {
    return QHPI_ERROR_FATAL;
  }

  const uint32_t resources = qhpi_thread_resources(handle);
#if defined(QHPI_EXCLUSIVE_SINGLE_HMX_AUDIT)
  // Diagnostic-only build: prove whether an exclusive main-control callback
  // may safely issue the exact HMX tile already accepted in EXP-0014.
  if (resources == QHPI_RESOURCE_MAIN || resources == 0) {
    for (uint32_t index = 0; index < probe::kWeightBytes; ++index) {
      first_weight[index] = static_cast<int8_t>(static_cast<int32_t>(encoded_weight[index]) - 128);
    }
    *reinterpret_cast<uint32_t*>(first_weight + kPhaseWordOffset) = 0;
    probe::fillAsymmetricBias(first_weight, 128, first_bias);
    probe::executeU8S8Tile(activation_storage, first_weight, first_bias, output_storage);
  }
  return QHPI_SUCCESS;
#endif
#if defined(QHPI_MIXED_RESOURCE_AUDIT)
  // Diagnostic-only build used to enumerate which callbacks QAIRT dispatches
  // for QHPI_RESOURCE_EXCLUSIVE. Each resource owns a disjoint 128-byte band.
  const uint32_t band = resources == 0                      ? 0
                        : resources == QHPI_RESOURCE_HVX    ? 1
                        : resources == QHPI_RESOURCE_HMX    ? 2
                                                           : 3;
  const uint8_t marker = static_cast<uint8_t>(240U + (resources & 0x0fU));
  for (uint32_t index = 0; index < 128; ++index) output_storage[band * 128 + index] = marker;
  return QHPI_SUCCESS;
#endif
  if (resources == QHPI_RESOURCE_MAIN || resources == 0) {
    // On QAIRT 2.49/V79, as in EXP-0014, the main-control callback issues
    // HMX instructions; there is no independent HMX worker callback.
    if (!waitForPhase(phase_word, kWeightReady)) return QHPI_ERROR_FATAL;
    probe::executeU8S8Tile(activation_storage, first_weight, first_bias, output_storage);
    publishPhase(phase_word, kFirstHmxReady);
    if (!waitForPhase(phase_word, kHvxReady)) return QHPI_ERROR_FATAL;
    // HMX loads the complete activation and weight tiles before its final
    // channel-major store, so this second tile may safely overwrite itself.
    probe::executeU8S8Tile(output_storage, second_weight, second_bias, output_storage);
    publishPhase(phase_word, kSecondHmxReady);
    return QHPI_SUCCESS;
  }
  if (resources == QHPI_RESOURCE_HVX) {
    if (__atomic_load_n(phase_word, __ATOMIC_ACQUIRE) != kInitialPhase) return QHPI_ERROR_FATAL;
    packSignedWeightHvx(encoded_weight, first_weight);
    probe::fillAsymmetricBias(first_weight, 128, first_bias);
    publishPhase(phase_word, kWeightReady);
    if (!waitForPhase(phase_word, kFirstHmxReady)) return QHPI_ERROR_FATAL;
    addOneHvx(output_storage);
    packSignedWeightHvx(encoded_weight, second_weight);
    probe::fillAsymmetricBias(second_weight, 0, second_bias);
    publishPhase(phase_word, kHvxReady);
    return waitForPhase(phase_word, kSecondHmxReady) ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
  }
  if (resources == QHPI_RESOURCE_HMX) {
    return waitForPhase(phase_word, kSecondHmxReady) ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
  }
  return waitForPhase(phase_word, kSecondHmxReady) ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
#endif
}

QHPI_Tensor_Signature_v1 input_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Tensor_Signature_v1 output_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 kernels[] = {{.function_name = THIS_PKG_NAME_STR "::mixedResourceHmxHvxHmx",
                             .function = mixedResourceHmxHvxHmx,
                             .resources = QHPI_RESOURCE_EXCLUSIVE,
                             .source_destructive = true,
                             .multithreaded = false,
                             .variable_inputs = false,
                             .variable_outputs = false,
                             .min_inputs = 2,
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

QHPI_OpInfo_v1 op_info[] = {{.name = THIS_PKG_NAME_STR "::MixedResourceHmxHvxHmx",
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

const QHPI_OpInfo_v1* mixed_resource_hmx_hvx_hmx_op_info() { return op_info; }
