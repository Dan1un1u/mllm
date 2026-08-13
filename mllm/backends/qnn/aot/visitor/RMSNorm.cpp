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
  auto input_spec = i_0->getAttr("quant_recipe")->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;
  auto output_spec = o_0->getAttr("quant_recipe")->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;
  MLLM_RETURN_FALSE_IF_NOT(input_spec->type == ir::linalg::QuantizationSpecType::kAsymPerTensor);
  MLLM_RETURN_FALSE_IF_NOT(output_spec->type == ir::linalg::QuantizationSpecType::kAsymPerTensor);
  auto input_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(input_spec);
  auto output_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(output_spec);

  auto weight_spec_attr = weight->getAttr("quant_recipe");
  MLLM_RETURN_FALSE_IF_NOT(weight_spec_attr);
  auto weight_spec_base = weight_spec_attr->cast_<mllm::ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;
  MLLM_RETURN_FALSE_IF_NOT(weight_spec_base->type == ir::linalg::QuantizationSpecType::kAsymPerTensor);
  auto weight_quant_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(weight_spec_base);

  const bool input_is_u8 = input_quant_spec->quant_to_type == kUInt8;
  const bool output_is_u8 = output_quant_spec->quant_to_type == kUInt8;
  const bool gamma_is_u8 = weight_quant_spec->quant_to_type == kUInt8 &&
                          (weight->tensor_.dtype() == kUInt8 || weight->tensor_.dtype() == kUInt8PerTensorAsy);
  const bool native_u8 = input_is_u8 && output_is_u8 && gamma_is_u8;
  const bool needs_a8_bridge = !native_u8 && (input_is_u8 || output_is_u8);

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
  // It is a synthetic all-zero tensor, so the native U8 experiment uses an
  // asymmetric U8 carrier and preserves exact real zero (zero_point=0).
  const auto parameter_dtype = gamma_is_u8 ? kUInt8 : kUInt16;
  const auto parameter_ir_dtype = gamma_is_u8 ? kUInt8PerTensorAsy : kUInt16PerTensorAsy;
  const int32_t parameter_quant_max = gamma_is_u8 ? 255 : 65535;
  auto bias_tensor = mllm::Tensor::zeros(weight->tensor_.shape(), parameter_dtype);
  bias_tensor = bias_tensor.__unsafeSetDType(parameter_ir_dtype);

  MLLM_WARN("Making Fake bias for RMSNorm");
  bias_tensor.setName(a->getName() + "_runtime_bias");
  auto bias_node = writer.getContext()->create<ir::tensor::TensorValue>(bias_tensor);

  // Fake bias quant recipe
  auto bias_scale = Tensor::ones({1}, kFloat32);
  auto bias_zero_point = Tensor::zeros({1}, kInt32);
  bias_scale.at<float>({0}) = weight_quant_spec->scale.item<float>();
  MLLM_RT_ASSERT_EQ(bias_zero_point.item<mllm_int32_t>(), 0);
  auto quant_spec = mllm::ir::linalg::QuantizationSpecAsymPerTensor::create(
      0, parameter_quant_max, parameter_dtype, kFloat32, kInt32, bias_scale, bias_zero_point);
  auto quant_attr = mllm::ir::linalg::LinalgIRQuantizatonSpecAttr::build(writer.getContext().get(), quant_spec);
  bias_node->setAttr("quant_recipe", quant_attr);

  // Use the native HTP U8 configuration only when all RmsNorm data operands
  // are U8. Legacy U16 gamma remains supported through the explicit bridge.
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
