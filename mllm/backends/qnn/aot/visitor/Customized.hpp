// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot/visitor/Base.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::qnn::aot {

class QnnAOTCustomizedPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTCustomizedPattern>> create() {
    return {OpTypes::kDynamicOp_Start, std::make_shared<QnnAOTCustomizedPattern>()};
  }
};

}  // namespace mllm::qnn::aot
