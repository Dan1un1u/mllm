// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/visitor/NativeGroupQueryAttention.hpp"

#include <cmath>

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

namespace {
constexpr const char* kExperimentMarker = "EXP0017NativeGroupQueryAttention";
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadDim = 128;
}  // namespace

bool QnnAOTNativeGroupQueryAttentionPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  auto customized = op->cast_<mllm::ir::linalg::CustomizedOp>();
  if (!customized || op->getAttr("using_qnn") == nullptr) return false;
  auto* custom_op = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(customized->getAOp());
  return custom_op != nullptr && custom_op->getCustomOpTypeName() == kExperimentMarker;
}

bool QnnAOTNativeGroupQueryAttentionPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  (void)writer;
  auto env = AOTCompileContext::getInstance().getEnv();
  auto customized = op->cast_<mllm::ir::linalg::CustomizedOp>();
  if (!customized || (op->inputs().size() != 7 && op->inputs().size() != 10)
      || op->outputs().size() != 3) {
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  const auto graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  const auto context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto qnn_op = QnnAOTNodeOperation::create("GroupQueryAttention");
  qnn_op->setPackageName("qti.aisw");
  for (auto& input_value : op->inputs()) {
    auto input = input_value->cast_<ir::tensor::TensorValue>();
    MLLM_RETURN_FALSE_IF_NOT(input);
    qnn_op->emplaceInput(env->captureQnnAOTNodeTensor(context_name, graph_name, input));
  }
  for (auto& output_value : op->outputs()) {
    auto output = output_value->cast_<ir::tensor::TensorValue>();
    MLLM_RETURN_FALSE_IF_NOT(output);
    qnn_op->emplaceOutput(env->captureQnnAOTNodeTensor(context_name, graph_name, output));
  }
  qnn_op->emplaceParamScalar(QNNParamScalarWrapper::create("num_heads", kQueryHeads));
  qnn_op->emplaceParamScalar(QNNParamScalarWrapper::create("kv_num_heads", kKvHeads));
  const bool do_rotary = op->inputs().size() == 10;
  qnn_op->emplaceParamScalar(
      QNNParamScalarWrapper::create("do_rotary", static_cast<uint32_t>(do_rotary)));
  qnn_op->emplaceParamScalar(QNNParamScalarWrapper::create(
      "scale", do_rotary ? 0.0F : 1.0F / std::sqrt(static_cast<float>(kHeadDim))));
  qnn_op->setName(customized->getAOp()->getName());
  env->captureAOTNodeOp(context_name, graph_name, qnn_op);
  return true;
}

}  // namespace mllm::qnn::aot
