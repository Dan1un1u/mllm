// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Elewise.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

#include <cstdlib>

namespace mllm::qnn::aot {

bool QnnAOTAddPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::AddOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTAddPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto add_op = op->cast_<mllm::ir::linalg::AddOp>();
  if (!add_op) {
    MLLM_ERROR("Failed to cast to linalg::AddOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Start to attach!
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto i_1 = (*(std::next(op->inputs().begin())))->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  const char* placement_flag = std::getenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX");
  const bool placement_enabled = placement_flag != nullptr && placement_flag[0] != '\0' && placement_flag[0] != '0';
  const auto& score_shape = i_0->tensor_.shape();
  const auto& position_shape = i_1->tensor_.shape();
  const auto broadcast_compatible = [](const std::vector<int32_t>& lhs, const std::vector<int32_t>& rhs) {
    if (lhs.size() != rhs.size()) { return false; }
    for (size_t dim = 0; dim < lhs.size(); ++dim) {
      if (lhs[dim] != rhs[dim] && lhs[dim] != 1 && rhs[dim] != 1) { return false; }
    }
    return true;
  };
  // Only the experimental [B, H, rows, K] score (K is a 32-lane prefix up
  // to 1024) plus broadcastable
  // [B, H, rows, 1] Int32 position marker may become the causal custom op.
  // Every residual/RoPE/MLP Add stays Qualcomm-native.
  const bool use_vtcm_softmax = placement_enabled && score_shape.size() == 4 && position_shape.size() == 4
                                && score_shape.back() >= 32 && score_shape.back() <= 1024 && score_shape.back() % 32 == 0
                                && position_shape.back() == 1 && i_1->tensor_.dtype() == kInt32
                                && broadcast_compatible(score_shape, position_shape);
  const char* multithreaded_flag = std::getenv("MLLM_QNN_VTCM_MASKED_E2_SOFTMAX_MT");
  const bool use_multithreaded_kernel =
      use_vtcm_softmax && multithreaded_flag != nullptr && multithreaded_flag[0] != '\0' && multithreaded_flag[0] != '0';
  auto qnn_op_node = QnnAOTNodeOperation::create(use_multithreaded_kernel ? "VtcmCausalE2SoftmaxHd128Mt"
                                                 : use_vtcm_softmax       ? "VtcmCausalE2SoftmaxHd128"
                                                                          : "ElementWiseAdd");
  qnn_op_node->setPackageName(use_vtcm_softmax ? "LLaMAPackage" : "qti.aisw");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_1))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(add_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

bool QnnAOTMulPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::MulOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTMulPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto mul_op = op->cast_<mllm::ir::linalg::MulOp>();
  if (!mul_op) {
    MLLM_ERROR("Failed to cast to linalg::MulOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Start to attach!
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto i_1 = (*(std::next(op->inputs().begin())))->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  auto qnn_op_node = QnnAOTNodeOperation::create("ElementWiseMultiply");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_1))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(mul_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

bool QnnAOTNegPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::NegOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTNegPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto neg_op = op->cast_<mllm::ir::linalg::NegOp>();
  if (!neg_op) {
    MLLM_ERROR("Failed to cast to linalg::NegOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  // Start to attach!
  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  auto qnn_op_node = QnnAOTNodeOperation::create("ElementWiseNeg");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(neg_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
