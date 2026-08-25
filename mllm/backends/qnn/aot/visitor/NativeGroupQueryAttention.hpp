// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot/visitor/Base.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::qnn::aot {

// EXP-0017: lower an experiment-only marker to QAIRT 2.49's native GQA.
class QnnAOTNativeGroupQueryAttentionPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTNativeGroupQueryAttentionPattern>> create() {
    return {OpTypes::kDynamicOp_Start, std::make_shared<QnnAOTNativeGroupQueryAttentionPattern>()};
  }
};

}  // namespace mllm::qnn::aot
