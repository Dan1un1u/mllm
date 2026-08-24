// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>

#include "HTP/core/qhpi.h"
#include "HmxInt8Tile.hpp"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

namespace probe = mllm::qnn::qhpi_hmx_probe;

namespace {

bool shapeEquals(const QHPI_Shape& shape, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
  return shape.rank == 4 && shape.dims[0] == d0 && shape.dims[1] == d1 && shape.dims[2] == d2 && shape.dims[3] == d3;
}

bool exactQuantization(const QHPI_Tensor* tensor, int32_t zero_offset) {
  const auto quant = qhpi_tensor_quant_parameters(tensor);
  return quant.zero_offset == zero_offset && quant.stepsize == 1.0F;
}

[[maybe_unused]] bool aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr && (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

uint32_t failContract(QHPI_Tensor* output, uint8_t marker) {
#if defined(QHPI_HMX_PROBE_DIAGNOSTIC)
  auto** blocks = qhpi_tensor_block_table(output);
  if (blocks != nullptr && qhpi_tensor_block_table_length(output) > 0 && blocks[0] != nullptr) {
    auto* bytes = static_cast<uint8_t*>(blocks[0]);
    for (uint32_t index = 0; index < probe::kOutputBytes; ++index) bytes[index] = marker;
    return QHPI_SUCCESS;
  }
#else
  (void)output;
  (void)marker;
#endif
  return QHPI_ERROR_FATAL;
}

uint32_t u8s8HmxMatMul(QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs, uint32_t num_inputs,
                       const QHPI_Tensor* const* inputs) {
  if (handle == nullptr || num_inputs != 2 || num_outputs != 1 || inputs == nullptr || outputs == nullptr
      || inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return QHPI_ERROR_FATAL;
  }
  const uint32_t thread_resources = qhpi_thread_resources(handle);
  // QAIRT 2.49/V79 invokes this non-self-sliced HMX plugin from the main
  // control thread (observed as 0) while the kernel descriptor reserves HMX.
  // Accept that observed runtime convention only if the final Optrace also
  // proves integer-HMX execution; any other worker class is a hard failure.
  if (thread_resources != 0 && thread_resources != QHPI_RESOURCE_HMX) {
    return failContract(outputs[0], static_cast<uint8_t>(0xc0u | (thread_resources & 0x1fu)));
  }
  if (!shapeEquals(qhpi_tensor_shape(inputs[0]), 1, 8, 8, 32)) return failContract(outputs[0], 0xe2);
  if (!shapeEquals(qhpi_tensor_shape(inputs[1]), 1, 1, 32, 32)) return failContract(outputs[0], 0xe3);
  if (!shapeEquals(qhpi_tensor_shape(outputs[0]), 1, 8, 8, 32)) return failContract(outputs[0], 0xe4);
  if (!exactQuantization(inputs[0], 128)) return failContract(outputs[0], 0xe5);
  if (!exactQuantization(inputs[1], 128)) return failContract(outputs[0], 0xe6);
  if (!exactQuantization(outputs[0], 0)) return failContract(outputs[0], 0xe7);
  if (qhpi_tensor_block_table_length(inputs[0]) != 1) return failContract(outputs[0], 0xe8);
  if (qhpi_tensor_block_table_length(outputs[0]) != 1) return failContract(outputs[0], 0xe9);

#if defined(REFERENCE_OP)
  // Host libraries provide metadata and registration only. The real kernel is
  // dispatched from the V79 DSP package.
  return QHPI_UNSUPPORTED;
#else
  auto** activation_blocks = qhpi_tensor_block_table(inputs[0]);
  auto** output_blocks = qhpi_tensor_block_table(outputs[0]);
  if (activation_blocks == nullptr || output_blocks == nullptr) { return QHPI_ERROR_FATAL; }
  // HTP canonicalizes the signed QNN graph input to QUInt8 storage before
  // QHPI matching. Its zero offset remains 128, so subtracting it recovers
  // the exact signed byte consumed by weight.b. Use the output crouton as
  // transient TCM scratch for both the 1 KiB signed tile and the 256-byte HMX
  // bias. Both are loaded by HMX before the final store overwrites them.
  const auto* encoded_weight = static_cast<const uint8_t*>(qhpi_tensor_raw_data(inputs[1]));
  auto* output_storage = static_cast<uint8_t*>(output_blocks[0]);
  auto* weight = reinterpret_cast<int8_t*>(output_storage);
  auto* bias = reinterpret_cast<uint32_t*>(output_storage + probe::kWeightBytes);
  if (!aligned(activation_blocks[0], 2048) || !aligned(encoded_weight, 128) || !aligned(weight, 128) || !aligned(bias, 256)
      || !aligned(output_blocks[0], 2048)) {
    return failContract(outputs[0], 0xea);
  }

  for (uint32_t index = 0; index < probe::kWeightBytes; ++index) {
    weight[index] = static_cast<int8_t>(static_cast<int32_t>(encoded_weight[index]) - 128);
  }

#if defined(QHPI_HMX_PROBE_DIAGNOSTIC)
  return failContract(outputs[0], 0xd0);
#else
  probe::fillAsymmetricBias(weight, 128, bias);
  probe::executeU8S8Tile(static_cast<const uint8_t*>(activation_blocks[0]), weight, bias, output_storage);
  return QHPI_SUCCESS;
#endif
#endif
}

QHPI_Tensor_Signature_v1 input_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Tensor_Signature_v1 output_signatures[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 kernels[] = {{.function_name = THIS_PKG_NAME_STR "::u8s8HmxMatMul",
                             .function = u8s8HmxMatMul,
                             .resources = QHPI_RESOURCE_HMX,
                             .source_destructive = false,
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

QHPI_OpInfo_v1 op_info[] = {{.name = THIS_PKG_NAME_STR "::U8S8HmxMatMul",
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

const QHPI_OpInfo_v1* u8s8_hmx_matmul_op_info() { return op_info; }
