// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Placement-only experiment for an interior QK -> masked-softmax -> PV edge.
// This intentionally is not a Softmax implementation.  Its sole purpose is to
// prove whether an external HTP op can keep both inputs and its output in VTCM.

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_VtcmMaskedSoftmaxPlacement);

GraphStatus vtcmMaskedSoftmaxPlacement(QUint8CroutonTensor_TCM& out,
                                       const QUint8CroutonTensor_TCM& scores,
                                       const QUint8CroutonTensor_TCM& mask) {
  out.set_dims(scores);
  (void)mask;

  // Preserve a data dependency for the placement experiment.  Accessors
  // dequantize/requantize when encodings differ; mathematical Softmax and mask
  // handling are deliberately deferred until the zero-DRAM gate passes.
  for (Idx b = 0; b < scores.dim(0); ++b) {
    for (Idx h = 0; h < scores.dim(1); ++h) {
      for (Idx w = 0; w < scores.dim(2); ++w) {
        for (Idx d = 0; d < scores.dim(3); ++d) { out(b, h, w, d) = scores(b, h, w, d); }
      }
    }
  }
  return GraphStatus::Success;
}

// Deliberately register no MainMemory or generic Tensor implementation.  If
// the central placement pass cannot satisfy this contract, graph finalization
// must fail rather than silently inserting a non-TCM custom kernel.
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((vtcmMaskedSoftmaxPlacement),
                                  "VtcmMaskedSoftmaxPlacement",
                                  FAST,
                                  Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("VtcmMaskedSoftmaxPlacement", "scores", "mask"),
                      Crouton("*", "scores", "mask"),
                      Tcm("*", "scores", "mask"))

END_PKG_OP_DEFINITION(PKG_VtcmMaskedSoftmaxPlacement);
