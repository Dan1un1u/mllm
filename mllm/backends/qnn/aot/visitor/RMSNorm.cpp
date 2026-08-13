// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/RMSNorm.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTRMSNormPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::RMSNormOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTRMSNormPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto rms_op = op->cast_<mllm::ir::linalg::RMSNormOp>();
  if (!rms_op) {
    MLLM_ERROR("Failed to cast to linalg::RMSNormOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto a = rms_op->getAOp();
  auto rms_aop = dynamic_cast<mllm::aops::RMSNormOp*>(a);
  if (!rms_aop) {
    MLLM_ERROR("Failed to cast to aops::RMSNormOp");
    return false;
  }

  auto weight =
      writer.getContext()->lookupSymbolTable(a->getName() + ".weight")->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();
  auto input_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(
      i_0->getAttr("quant_recipe")->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  auto output_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(
      o_0->getAttr("quant_recipe")->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  const bool needs_a8_bridge = input_quant_spec->quant_to_type == kUInt8 || output_quant_spec->quant_to_type == kUInt8;

  auto make_uint16_bridge = [&](const ir::tensor::TensorValue::ptr_t& source, const std::string& suffix,
                                const std::shared_ptr<ir::linalg::QuantizationSpecAsymPerTensor>& source_spec) {
    auto tensor = Tensor::empty(source->tensor_.shape(), kUInt16).__unsafeSetDType(kUInt16PerTensorAsy);
    tensor.setName(a->getName() + suffix);
    auto value = writer.getContext()->create<ir::tensor::TensorValue>(tensor);
    auto bridge_spec = ir::linalg::QuantizationSpecAsymPerTensor::create(
        0, 65535, kUInt16, kFloat32, kInt32, source_spec->scale, source_spec->zero_point);
    value->setAttr("quant_recipe", writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(bridge_spec));
    return value;
  };

  auto rms_input = needs_a8_bridge ? make_uint16_bridge(i_0, "_a16_input", input_quant_spec) : i_0;
  auto rms_output = needs_a8_bridge ? make_uint16_bridge(o_0, "_a16_output", output_quant_spec) : o_0;

  // Fake bias, nn module seems to be inconsistent with document (AMAZING!).
  // Keep the parameter-side bias carrier aligned with the preserved UInt16
  // RMSNorm weight. Activation inputs/outputs may independently be A8.
  auto bias_tensor = mllm::Tensor::zeros(weight->tensor_.shape(), kUInt16);
  bias_tensor = bias_tensor.__unsafeSetDType(kUInt16PerTensorAsy);

  MLLM_WARN("Making Fake bias for RMSNorm");
  bias_tensor.setName(a->getName() + "_runtime_bias");
  auto bias_node = writer.getContext()->create<ir::tensor::TensorValue>(bias_tensor);

  // Fake bias quant recipe
  auto bias_scale = Tensor::ones({1}, kFloat32);
  auto bias_zero_point = Tensor::zeros({1}, kInt32);
  auto weight_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(
      weight->getAttr("quant_recipe")->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  bias_scale.at<float>({0}) = weight_quant_spec->scale.item<float>();
  MLLM_RT_ASSERT_EQ(bias_zero_point.item<mllm_int32_t>(), 0);
  auto quant_spec = mllm::ir::linalg::QuantizationSpecAsymPerTensor::create(
      0, 65535, kUInt16, kFloat32, kInt32, bias_scale, bias_zero_point);
  auto quant_attr = mllm::ir::linalg::LinalgIRQuantizatonSpecAttr::build(writer.getContext().get(), quant_spec);
  bias_node->setAttr("quant_recipe", quant_attr);

  // QAIRT requires an UInt16 gamma to run in the INT16 RmsNorm
  // configuration. Keep target Linear activations A8 by making the boundary
  // conversions explicit inside the HTP graph.
  if (needs_a8_bridge) {
    auto pre_convert = QnnAOTNodeOperation::create("Convert");
    pre_convert->setPackageName("qti.aisw");
    pre_convert->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0))
        ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, rms_input))
        ->setName(a->getName() + ".a8_to_a16");
    env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, pre_convert);
  }

  auto qnn_op_node = QnnAOTNodeOperation::create("RmsNorm");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceParamScalar(mllm::qnn::QNNParamScalarWrapper::create("epsilon", rms_aop->options().epsilon));

  std::vector<uint32_t> axes_dims = {1};
  auto axes_param = mllm::qnn::QNNParamTensorWrapper::create("axes", a->getName() + "_axes", QNN_DATATYPE_UINT_32, axes_dims);
  uint32_t* axes_data = (uint32_t*)axes_param->alloc();
  axes_data[0] = i_0->tensor_.shape().size() - 1;
  qnn_op_node->emplaceParamTensor(axes_param);

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, rms_input))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight, true))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, bias_node, true))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, rms_output))
      ->setName(rms_op->getAOp()->getName());

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  if (needs_a8_bridge) {
    auto post_convert = QnnAOTNodeOperation::create("Convert");
    post_convert->setPackageName("qti.aisw");
    post_convert->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, rms_output))
        ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
        ->setName(a->getName() + ".a16_to_a8");
    env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, post_convert);
  }

  return true;
}

}  // namespace mllm::qnn::aot
