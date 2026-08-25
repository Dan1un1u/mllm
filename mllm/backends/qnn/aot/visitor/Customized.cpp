// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/visitor/Customized.hpp"

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

namespace {
constexpr const char* kFusedGqaType = "FusedGqaHmxSoftmaxAv";
}

bool QnnAOTCustomizedPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  auto customized = op->cast_<mllm::ir::linalg::CustomizedOp>();
  if (!customized || op->getAttr("using_qnn") == nullptr) return false;
  auto* custom_op = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(customized->getAOp());
  return custom_op != nullptr && custom_op->getCustomOpTypeName() == kFusedGqaType;
}

bool QnnAOTCustomizedPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();
  auto customized = op->cast_<mllm::ir::linalg::CustomizedOp>();
  if (!customized || op->inputs().size() != 4 || op->outputs().size() != 1) return false;

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  const auto graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  const auto context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto qnn_op = QnnAOTNodeOperation::create(kFusedGqaType);
  qnn_op->setPackageName("QhpiHmxProbePackage");
  for (auto& input_value : op->inputs()) {
    auto input = input_value->cast_<ir::tensor::TensorValue>();
    MLLM_RETURN_FALSE_IF_NOT(input);
    qnn_op->emplaceInput(env->captureQnnAOTNodeTensor(context_name, graph_name, input));
  }
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  MLLM_RETURN_FALSE_IF_NOT(output);
  qnn_op->emplaceOutput(env->captureQnnAOTNodeTensor(context_name, graph_name, output))
      ->setName(customized->getAOp()->getName());
  env->captureAOTNodeOp(context_name, graph_name, qnn_op);
  return true;
}

}  // namespace mllm::qnn::aot
