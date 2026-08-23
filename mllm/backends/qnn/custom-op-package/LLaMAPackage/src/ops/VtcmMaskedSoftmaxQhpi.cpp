// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// QHPI and simple_reg expose incompatible private/public declarations when
// included in one translation unit. Reuse the exact SIMD implementation while
// compiling the QHPI registration independently from the legacy op wrapper.
#define MLLM_VTCM_SOFTMAX_QHPI_TRANSLATION_UNIT 1
#include "VtcmMaskedSoftmaxPlacement.cpp"
