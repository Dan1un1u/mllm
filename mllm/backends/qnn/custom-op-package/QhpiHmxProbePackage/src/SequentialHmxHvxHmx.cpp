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

#if defined(QHPI_SEQUENTIAL_ALIAS_AUDIT)
constexpr bool kSourceDestructive = true;
#else
constexpr bool kSourceDestructive = false;
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

void packSignedWeightHvx(const uint8_t* encoded_weight, int8_t* signed_weight) {
  const HVX_Vector sign_flip = Q6_V_vsplat_R(0x80808080);
  for (uint32_t offset = 0; offset < probe::kWeightBytes; offset += sizeof(HVX_Vector)) {
    const HVX_Vector encoded = *reinterpret_cast<const HVX_Vector*>(encoded_weight + offset);
    *reinterpret_cast<HVX_Vector*>(signed_weight + offset) = Q6_V_vxor_VV(encoded, sign_flip);
  }
}

void addOneHvx(uint8_t* data) {
  const HVX_Vector one = Q6_V_vsplat_R(0x01010101);
  for (uint32_t offset = 0; offset < probe::kOutputBytes; offset += sizeof(HVX_Vector)) {
    const HVX_Vector input = *reinterpret_cast<const HVX_Vector*>(data + offset);
    *reinterpret_cast<HVX_Vector*>(data + offset) = Q6_Vub_vadd_VubVub_sat(input, one);
  }
}

void copyHvx(const uint8_t* source, uint8_t* destination) {
  for (uint32_t offset = 0; offset < probe::kOutputBytes; offset += sizeof(HVX_Vector)) {
    *reinterpret_cast<HVX_Vector*>(destination + offset) =
        *reinterpret_cast<const HVX_Vector*>(source + offset);
  }
}
#endif

uint32_t sequentialHmxHvxHmx(QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs,
                             uint32_t num_inputs, const QHPI_Tensor* const* inputs) {
  if (handle == nullptr || num_inputs != 2 || num_outputs != 1 || inputs == nullptr || outputs == nullptr
      || inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return QHPI_ERROR_FATAL;
  }
  const uint32_t resources = qhpi_thread_resources(handle);
  // EXP-0014 established that QAIRT 2.49 invokes a non-self-sliced
  // QHPI_RESOURCE_HMX kernel through the main-control callback (reported as
  // zero). Reject every resource class outside that observed contract.
  if (resources != 0 && resources != QHPI_RESOURCE_HMX) return QHPI_ERROR_FATAL;
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
  auto* encoded_weight = static_cast<const uint8_t*>(qhpi_tensor_raw_data(inputs[1]));
  if (activation_blocks == nullptr || output_blocks == nullptr) return QHPI_ERROR_FATAL;

  auto* activation_storage = static_cast<uint8_t*>(activation_blocks[0]);
  auto* output_storage = static_cast<uint8_t*>(output_blocks[0]);
  auto* first_weight = reinterpret_cast<int8_t*>(output_storage);
  auto* first_bias = reinterpret_cast<uint32_t*>(output_storage + probe::kWeightBytes);
  auto* second_weight = reinterpret_cast<int8_t*>(activation_storage);
  auto* second_bias = reinterpret_cast<uint32_t*>(activation_storage + probe::kWeightBytes);
  if (!aligned(activation_storage, 2048) || !aligned(output_storage, 2048) || !aligned(encoded_weight, 128)
      || !aligned(first_weight, 128) || !aligned(first_bias, 256) || !aligned(second_weight, 128)
      || !aligned(second_bias, 256)) {
    return QHPI_ERROR_FATAL;
  }

#if defined(QHPI_SEQUENTIAL_ALIAS_AUDIT)
  const uint8_t alias_marker = activation_storage == output_storage ? 0xa1 : 0xa2;
  for (uint32_t index = 0; index < probe::kOutputBytes; ++index) output_storage[index] = alias_marker;
  return QHPI_SUCCESS;
#endif
  if (activation_storage == output_storage) return QHPI_ERROR_FATAL;

  // The output crouton is dead scratch until the first HMX store. HMX loads
  // the complete signed-weight and bias tiles before overwriting that storage.
  packSignedWeightHvx(encoded_weight, first_weight);
  probe::fillAsymmetricBias(first_weight, 128, first_bias);
  probe::executeU8S8Tile(activation_storage, first_weight, first_bias, output_storage);

  // This is deliberately real HVX SIMD work between the two matrix tiles,
  // not a scalar stand-in. It models the vector stage of fused attention.
  addOneHvx(output_storage);

  // The original activation is now dead, so reuse its TCM crouton for both
  // the second signed-weight/bias tiles and then the second HMX destination.
  // This is the same accepted destination-overwrites-staging pattern as the
  // first tile. In contrast, V79 does not preserve a crouton when it is both
  // the activation source and destination of one HMX tile, so keep the live
  // output crouton as a distinct source and copy the result back with HVX.
  packSignedWeightHvx(encoded_weight, second_weight);
  probe::fillAsymmetricBias(second_weight, 0, second_bias);
  probe::executeU8S8Tile(output_storage, second_weight, second_bias, activation_storage);
  copyHvx(activation_storage, output_storage);
  return QHPI_SUCCESS;
#endif
}

QHPI_Tensor_Signature_v1 input_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Tensor_Signature_v1 output_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 kernels[] = {{.function_name = THIS_PKG_NAME_STR "::sequentialHmxHvxHmx",
                             .function = sequentialHmxHvxHmx,
                             .resources = QHPI_RESOURCE_HMX,
                             // The implementation consumes and then reuses the
                             // first input, but the first input and output must
                             // remain distinct during the two-HMX sequence.
                             .source_destructive = kSourceDestructive,
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

QHPI_OpInfo_v1 op_info[] = {{.name = THIS_PKG_NAME_STR "::SequentialHmxHvxHmx",
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

const QHPI_OpInfo_v1* sequential_hmx_hvx_hmx_op_info() { return op_info; }
