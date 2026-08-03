// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>

#include <QnnTypes.h>
#include <nlohmann/json.hpp>

#include <QnnContext.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpGraph.h>

#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

namespace {

bool envFlagEnabled(const char* name, bool fallback = false) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') { return fallback; }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string safeArtifactName(std::string name) {
  for (auto& c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '_' && c != '-') { c = '_'; }
  }
  return name;
}

const char* qnnDataTypeName(Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_INT_4: return "INT4";
    case QNN_DATATYPE_INT_8: return "INT8";
    case QNN_DATATYPE_INT_16: return "INT16";
    case QNN_DATATYPE_INT_32: return "INT32";
    case QNN_DATATYPE_UINT_4: return "UINT4";
    case QNN_DATATYPE_UINT_8: return "UINT8";
    case QNN_DATATYPE_UINT_16: return "UINT16";
    case QNN_DATATYPE_UINT_32: return "UINT32";
    case QNN_DATATYPE_FLOAT_16: return "FLOAT16";
    case QNN_DATATYPE_FLOAT_32: return "FLOAT32";
    case QNN_DATATYPE_SFIXED_POINT_4: return "SFIXED_POINT_4";
    case QNN_DATATYPE_SFIXED_POINT_8: return "SFIXED_POINT_8";
    case QNN_DATATYPE_SFIXED_POINT_16: return "SFIXED_POINT_16";
    case QNN_DATATYPE_UFIXED_POINT_4: return "UFIXED_POINT_4";
    case QNN_DATATYPE_UFIXED_POINT_8: return "UFIXED_POINT_8";
    case QNN_DATATYPE_UFIXED_POINT_16: return "UFIXED_POINT_16";
    case QNN_DATATYPE_BOOL_8: return "BOOL8";
    default: return "UNDEFINED_OR_OTHER";
  }
}

const char* qnnTensorTypeName(Qnn_TensorType_t type) {
  switch (type) {
    case QNN_TENSOR_TYPE_APP_WRITE: return "APP_WRITE";
    case QNN_TENSOR_TYPE_APP_READ: return "APP_READ";
    case QNN_TENSOR_TYPE_APP_READWRITE: return "APP_READWRITE";
    case QNN_TENSOR_TYPE_NATIVE: return "NATIVE";
    case QNN_TENSOR_TYPE_STATIC: return "STATIC";
    case QNN_TENSOR_TYPE_NULL: return "NULL";
    default: return "UNDEFINED_OR_OTHER";
  }
}

const char* quantRecipeTypeName(ir::linalg::QuantizationSpecType type) {
  using T = ir::linalg::QuantizationSpecType;
  switch (type) {
    case T::kNone: return "none";
    case T::kRaw: return "raw";
    case T::kSymPerTensor: return "symmetric_per_tensor";
    case T::kSymPerChannel: return "symmetric_per_channel";
    case T::kSymPerBlock: return "symmetric_per_block";
    case T::kAsymPerTensor: return "asymmetric_per_tensor";
    case T::kAsymPerChannel: return "asymmetric_per_channel";
    case T::kAsymPerBlock: return "asymmetric_per_block";
    case T::kLPBQ: return "lpbq";
  }
  return "unknown";
}

nlohmann::json quantRecipeJson(const ir::tensor::TensorValue::ptr_t& value) {
  nlohmann::json result = nlohmann::json::object();
  auto attr = value->getAttr("quant_recipe");
  if (!attr) {
    result["type"] = "missing";
    return result;
  }
  auto spec = attr->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;
  result["type"] = quantRecipeTypeName(spec->type);
  result["solved"] = spec->solved;
  using T = ir::linalg::QuantizationSpecType;
  switch (spec->type) {
    case T::kRaw: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecRaw>(spec);
      result["storage_dtype"] = nameOfType(cfg->type_);
      break;
    }
    case T::kSymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kSymPerChannel: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerChannel>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max}, {"axis", cfg->ch_axis},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kSymPerBlock: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerBlock>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max}, {"block_size", cfg->block_size},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kAsymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kAsymPerChannel: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerChannel>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max}, {"axis", cfg->ch_axis},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kAsymPerBlock: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerBlock>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max}, {"block_size", cfg->block_size},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)}});
      break;
    }
    case T::kLPBQ: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(spec);
      result.update({{"quant_min", cfg->quant_min}, {"quant_max", cfg->quant_max}, {"block_size", cfg->block_size},
                     {"channel_axis", cfg->ch_axis}, {"block_scale_bitwidth", cfg->scale_level_0_bitwidth},
                     {"quant_to_dtype", nameOfType(cfg->quant_to_type)},
                     {"channel_scale_dtype", nameOfType(cfg->scale_1_type)}});
      break;
    }
    case T::kNone: break;
  }
  return result;
}

nlohmann::json scaleOffsetStats(const Qnn_ScaleOffset_t* values, uint32_t count) {
  nlohmann::json result{{"count", count}};
  if (values == nullptr || count == 0) { return result; }
  float minScale = std::numeric_limits<float>::max();
  float maxScale = std::numeric_limits<float>::lowest();
  double scaleSum = 0.0;
  int32_t minZeroPoint = std::numeric_limits<int32_t>::max();
  int32_t maxZeroPoint = std::numeric_limits<int32_t>::lowest();
  for (uint32_t i = 0; i < count; ++i) {
    minScale = std::min(minScale, values[i].scale);
    maxScale = std::max(maxScale, values[i].scale);
    scaleSum += values[i].scale;
    const int32_t zeroPoint = -values[i].offset;
    minZeroPoint = std::min(minZeroPoint, zeroPoint);
    maxZeroPoint = std::max(maxZeroPoint, zeroPoint);
  }
  result.update({{"scale_min", minScale}, {"scale_max", maxScale}, {"scale_mean", scaleSum / count},
                 {"zero_point_min", minZeroPoint}, {"zero_point_max", maxZeroPoint}});
  if (count <= 16) {
    result["values"] = nlohmann::json::array();
    for (uint32_t i = 0; i < count; ++i) {
      result["values"].push_back({{"scale", values[i].scale}, {"zero_point", -values[i].offset}});
    }
  }
  return result;
}

nlohmann::json nativeQuantizationJson(const Qnn_Tensor_t* tensor) {
  const auto& quant = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  nlohmann::json result{{"defined", quant.encodingDefinition == QNN_DEFINITION_DEFINED},
                        {"encoding_id", static_cast<int64_t>(quant.quantizationEncoding)}};
  if (quant.encodingDefinition != QNN_DEFINITION_DEFINED) {
    result["encoding"] = "undefined";
    return result;
  }
  switch (quant.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      result["encoding"] = "scale_offset";
      result["scale"] = quant.scaleOffsetEncoding.scale;
      result["zero_point"] = -quant.scaleOffsetEncoding.offset;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      result["encoding"] = "axis_scale_offset";
      result["axis"] = quant.axisScaleOffsetEncoding.axis;
      result["scale_offset_stats"] = scaleOffsetStats(quant.axisScaleOffsetEncoding.scaleOffset,
                                                       quant.axisScaleOffsetEncoding.numScaleOffsets);
      break;
    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION: {
      result["encoding"] = "blockwise_expansion";
      auto* block = quant.blockwiseExpansion;
      if (block == nullptr) { break; }
      const auto* dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
      const uint32_t axisSize = dims == nullptr ? 0 : dims[block->axis];
      const uint64_t blockScaleCount = static_cast<uint64_t>(axisSize) * block->numBlocksPerAxis;
      result.update({{"axis", block->axis}, {"axis_size", axisSize},
                     {"num_blocks_per_axis", block->numBlocksPerAxis},
                     {"block_scale_bitwidth", block->blockScaleBitwidth},
                     {"block_scale_storage_bits",
                      block->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8 ? 8 : 16},
                     {"block_scale_count", blockScaleCount},
                     {"channel_scale_stats", scaleOffsetStats(block->scaleOffsets, axisSize)}});
      if (blockScaleCount != 0) {
        uint32_t minValue = std::numeric_limits<uint32_t>::max();
        uint32_t maxValue = 0;
        double sum = 0.0;
        for (uint64_t i = 0; i < blockScaleCount; ++i) {
          const uint32_t value = block->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8
                                     ? block->blocksScale8[i]
                                     : block->blocksScale16[i];
          minValue = std::min(minValue, value);
          maxValue = std::max(maxValue, value);
          sum += value;
        }
        result["block_scale_stats"] = {{"min", minValue}, {"max", maxValue},
                                         {"mean", sum / blockScaleCount}};
      }
      break;
    }
    default: result["encoding"] = "supported_by_qnn_but_not_expanded_by_manifest"; break;
  }
  return result;
}

}  // namespace

QnnAOTNodeTensor::QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  ir_storage_dtype_ = nameOfType(v->tensor_.dtype());
  quant_recipe_json_ = quantRecipeJson(v).dump();
  auto type = parseQnnTensorTypeFromIR(v);
  auto name = v->name();
  auto quant = parseQnnQuantizeParamFromIR(v);

  if (force_static_weight || type == QNN_TENSOR_TYPE_STATIC) {
    tensor_wrapper_ = mllm::qnn::QNNTensorWrapper::createStaticTensor(name, v->tensor_, quant);
  } else {
    tensor_wrapper_ = mllm::qnn::QNNTensorWrapper::create(name, type, v->tensor_, quant);
  }
  setupComplexTensorQuantization(v);  // per-channel and LPBQ cases
}

Qnn_TensorType_t QnnAOTNodeTensor::parseQnnTensorTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  auto type = v->tensor_.memType();
  Qnn_TensorType_t ret_qnn_tensor_type = QNN_TENSOR_TYPE_UNDEFINED;
  switch (type) {
    case kTensorMemTypes_Start: {
      break;
    }

    // For MLLM Frame work to use
    case kNormal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_NATIVE;
      break;
    }
    case kExtraInput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kExtraOutput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kManual: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kGlobal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // Framework need to judge if this tensor is mmap from disk.
    case kParams_Start:
    case kParamsMMAP:
    case kParamsNormal:
    case kParams_End: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // For QNN Backend to use.
    case kQnnAppRead: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kQnnAppWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kQnnAppReadWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kTensorMemTypes_End: break;
  }

  // Check Attribute. The Attribute priority is higher than tensor type
  if (v->getAttr("qnn_graph_outputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ; }
  if (v->getAttr("qnn_graph_inputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE; }
  if (v->getAttr("constant")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC; }

  return ret_qnn_tensor_type;
}

Qnn_DataType_t QnnAOTNodeTensor::parseQnnDataTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  return mllm::qnn::mllmDataTypeToQnnDataType(v->tensor_.dtype());
}

std::string QnnAOTNodeTensor::parseQnnTensorNameFromIR(const ir::tensor::TensorValue::ptr_t& v) { return v->name(); }

Qnn_QuantizeParams_t QnnAOTNodeTensor::parseQnnQuantizeParamFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  Qnn_QuantizeParams_t ret = QNN_QUANTIZE_PARAMS_INIT;

  MLLM_RT_ASSERT(v);
  MLLM_RT_ASSERT(v->getAttr("quant_recipe"));
  auto quant_spec = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;

  switch (quant_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw:
    case ir::linalg::QuantizationSpecType::kSymPerChannel:
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(quant_spec);
      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      if (!cfg->scale || !cfg->zero_point) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "AsymPerTensor quant recipe has no scale or zero point. tensor: {}", v->name());
      }
      ret.scaleOffsetEncoding =
          Qnn_ScaleOffset_t{.scale = cfg->scale.item<float>(), .offset = -cfg->zero_point.item<int32_t>()};
      MLLM_INFO("Configuring AsymPerTensor quantization for tensor: {}, scale: {}, zero_point: {}", v->name(),
                cfg->scale.item<float>(), cfg->zero_point.item<int32_t>());
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(quant_spec);
      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      if (!cfg->scale) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "SymPerTensor quant recipe has no scale. tensor: {}", v->name());
      }

      MLLM_RT_ASSERT_EQ(cfg->quant_to_type, kUInt8);

      ret.scaleOffsetEncoding = Qnn_ScaleOffset_t{.scale = cfg->scale.item<float>(), .offset = -128};
      MLLM_INFO("Configuring SymPerTensor quantization for tensor: {}, scale: {}", v->name(), cfg->scale.item<float>());
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't handle kNone type");
    }
  }

  return ret;
}

void QnnAOTNodeTensor::setupComplexTensorQuantization(const ir::tensor::TensorValue::ptr_t& v) {
  MLLM_RT_ASSERT(v->getAttr("quant_recipe"));
  auto quant_spec = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;

  switch (quant_spec->type) {
    case ir::linalg::QuantizationSpecType::kSymPerChannel: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerChannel>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(cfg->ch_axis);
      std::vector<Qnn_ScaleOffset_t> scale_offsets(num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale.size(0));
      MLLM_RT_ASSERT_EQ(cfg->scale.dtype(), kFloat32);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_offsets[i].scale = cfg->scale.at<float>({i});
        scale_offsets[i].offset = 0;
      }

      tensor_wrapper_->setScaleOffsetQuantization(scale_offsets, cfg->ch_axis);
      break;
    }
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      MLLM_INFO("Solving LPBQ quantization for tensor: {}", v->tensor_.name());
      // This LPBQ Type is for Conv2D Only !!! Linear has diff layout cmp with conv2d

      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(-1);
      std::vector<Qnn_ScaleOffset_t> scale_offsets(num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale_level_1_fp.size(-1));
      MLLM_RT_ASSERT_EQ(cfg->scale_level_0_int.dtype(), kUInt8);
      MLLM_RT_ASSERT_EQ(cfg->scale_level_1_fp.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(cfg->scale_level_0_int.rank(), 1);
      MLLM_RT_ASSERT_EQ(cfg->scale_level_1_fp.rank(), 1);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_offsets[i].scale = cfg->scale_level_1_fp.at<float>({i});
        scale_offsets[i].offset = 0;
      }

      Qnn_BlockwiseExpansion_t blockwise_expansion;
      blockwise_expansion.axis = v->tensor_.rank() - 1;
      blockwise_expansion.scaleOffsets = nullptr;  // Will be set by setBlockwiseQuantization
      blockwise_expansion.numBlocksPerAxis = v->tensor_.size(-2) / cfg->block_size;
      blockwise_expansion.blockScaleBitwidth = 4;  // 4 bits for uint4 scale
      blockwise_expansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;
      blockwise_expansion.blocksScale8 = cfg->scale_level_0_int.ptr<mllm_uint8_t>();

      tensor_wrapper_->setBlockwiseQuantization(blockwise_expansion, scale_offsets);
      break;
    }
    default: break;
  }
}

// QnnAOTNodeOperation implementations
QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addInputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ins) {
  inputs.insert(inputs.end(), ins.begin(), ins.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addOutputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ous) {
  outputs.insert(outputs.end(), ous.begin(), ous.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceInput(const QnnAOTNodeTensor::ptr_t& input) {
  inputs.push_back(input);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceOutput(const QnnAOTNodeTensor::ptr_t& output) {
  outputs.push_back(output);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamScalar(
    const std::vector<std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>>& params) {
  param_scalar.insert(param_scalar.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamScalar(
    const std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>& param) {
  param_scalar.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamTensor(
    const std::vector<std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>>& params) {
  param_tensor.insert(param_tensor.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamTensor(
    const std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>& param) {
  param_tensor.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setOpName(const std::string& op_name) {
  op_name_ = op_name;
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setName(const std::string& name) {
  name_ = name;
  return shared_from_this();
}

std::string QnnAOTNodeOperation::getName() { return name_; }

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setPackageName(const std::string& package_name) {
  package_name_ = package_name;
  return shared_from_this();
}

QnnAOTGraph::QnnAOTGraph(QNN_INTERFACE_VER_TYPE& qnnInterface, Qnn_BackendHandle_t backendHandle,
                         Qnn_ContextHandle_t contextHandle, Qnn_ProfileHandle_t profileHandle,
                         const std::string& graphName)
    : qnn_interface_(&qnnInterface), profile_handle_(profileHandle), graph_name_(graphName) {
  qnn_model_ = std::make_shared<mllm::qnn::QNNModel>(qnnInterface, backendHandle);

  // Short Depth Conv On HMX Off
  QnnHtpGraph_CustomConfig_t* p_custom_config = nullptr;
  // FIXME: @chenghuaWang The code below will make llm inference slow!!!
  // p_custom_config = (QnnHtpGraph_CustomConfig_t*)malloc(sizeof(QnnHtpGraph_CustomConfig_t));
  // p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
  // p_custom_config->shortDepthConvOnHmxOff = true;
  // htp_graph_configs.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  // Fold Relu Activation Into Conv Off
  p_custom_config = (QnnHtpGraph_CustomConfig_t*)malloc(sizeof(QnnHtpGraph_CustomConfig_t));
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  p_custom_config->foldReluActivationIntoConvOff = true;
  htp_graph_configs.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  // FIXME: If need or not
  p_custom_config = (QnnHtpGraph_CustomConfig_t*)malloc(sizeof(QnnHtpGraph_CustomConfig_t));
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  p_custom_config->precision = QNN_PRECISION_FLOAT16;
  htp_graph_configs.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  // Optimization level
  p_custom_config = (QnnHtpGraph_CustomConfig_t*)malloc(sizeof(QnnHtpGraph_CustomConfig_t));
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  p_custom_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  p_custom_config->optimizationOption.floatValue = 3;
  htp_graph_configs.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  // VTCM Size
  p_custom_config = (QnnHtpGraph_CustomConfig_t*)malloc(sizeof(QnnHtpGraph_CustomConfig_t));
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  p_custom_config->vtcmSizeInMB = 8;
  htp_graph_configs.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  qnn_graph_configs.resize(htp_graph_configs.size());
  qnn_graph_configs.reserve(htp_graph_configs.size() + 1);
  for (int i = 0; i < htp_graph_configs.size(); ++i) {
    qnn_graph_configs[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    qnn_graph_configs[i].customConfig = htp_graph_configs[i];
    qnn_graph_config_pass_in_.push_back(&qnn_graph_configs[i]);
  }

  qnn_graph_config_pass_in_.push_back(nullptr);

  qnn_model_->initialize(contextHandle, graphName.c_str(), false, 1, qnn_graph_config_pass_in_.data());
}

void QnnAOTGraph::addTensor(const QnnAOTNodeTensor::ptr_t& tensor) {
  qnn_model_->addTensorWrapper(tensor->getWrapper());
  all_tensors_.insert({tensor->getWrapper()->getName(), tensor});
}

void QnnAOTGraph::addOperation(const QnnAOTNodeOperation::ptr_t& qnn_op) {
  std::vector<std::string> inputNames;
  for (auto& in : qnn_op->inputs) inputNames.push_back(in->getWrapper()->getName());

  std::vector<std::string> outputNames;
  for (auto& out : qnn_op->outputs) outputNames.push_back(out->getWrapper()->getName());

  for (auto& in : qnn_op->inputs) qnn_model_->addTensorWrapper(in->getWrapper());
  for (auto& out : qnn_op->outputs) qnn_model_->addTensorWrapper(out->getWrapper());

  qnn_model_->addNode(QNN_OPCONFIG_VERSION_1, qnn_op->name_, qnn_op->package_name_, qnn_op->op_name_, qnn_op->param_tensor,
                      qnn_op->param_scalar, inputNames, outputNames);

  op_node_.insert({qnn_op->getName(), qnn_op});
}

bool QnnAOTGraph::compile() {
  if (is_compiled_) { return true; }
  dumpQuantizationManifest();
  bool ret = qnn_model_->finalizeGraph(profile_handle_, nullptr) == mllm::qnn::MODEL_NO_ERROR;
  if (ret && profile_handle_ != nullptr) { dumpOptraceArtifacts(); }
  is_compiled_ = true;
  return ret;
}

void QnnAOTGraph::dumpQuantizationManifest() {
  const char* manifestDirEnv = std::getenv("MLLM_QNN_AOT_QUANT_MANIFEST_DIR");
  const char* optraceDirEnv = std::getenv("MLLM_QNN_AOT_OPTRACE_DIR");
  const char* selectedDir = manifestDirEnv && manifestDirEnv[0] ? manifestDirEnv : optraceDirEnv;
  if (selectedDir == nullptr || selectedDir[0] == '\0') { return; }

  const std::filesystem::path outputDir = selectedDir;
  std::error_code fsError;
  std::filesystem::create_directories(outputDir, fsError);
  if (fsError) {
    MLLM_ERROR("Failed to create quantization manifest directory {}: {}", outputDir.string(), fsError.message());
    return;
  }

  std::map<std::string, QnnAOTNodeTensor::ptr_t> tensors;
  std::map<std::string, std::string> producers;
  std::map<std::string, std::set<std::string>> consumers;
  std::vector<QnnAOTNodeOperation::ptr_t> operations;
  operations.reserve(op_node_.size());
  for (const auto& item : op_node_) {
    const auto& op = item.second;
    operations.push_back(op);
    for (const auto& tensor : op->inputs) {
      const auto& name = tensor->getWrapper()->getName();
      tensors[name] = tensor;
      consumers[name].insert(op->name_);
    }
    for (const auto& tensor : op->outputs) {
      const auto& name = tensor->getWrapper()->getName();
      tensors[name] = tensor;
      producers[name] = op->name_;
    }
  }
  for (const auto& [name, tensor] : all_tensors_) { tensors[name] = tensor; }
  std::sort(operations.begin(), operations.end(), [](const auto& lhs, const auto& rhs) { return lhs->name_ < rhs->name_; });

  nlohmann::json manifest{{"schema_version", 2},
                          {"graph", graph_name_},
                          {"scope", "pre-finalize QNN graph; HTP lowering may change physical execution dtypes"},
                          {"operations", nlohmann::json::array()},
                          {"tensors", nlohmann::json::array()}};
  for (const auto& op : operations) {
    nlohmann::json item{{"name", op->name_}, {"qnn_op_type", op->op_name_}, {"package", op->package_name_},
                        {"inputs", nlohmann::json::array()}, {"outputs", nlohmann::json::array()}};
    for (const auto& tensor : op->inputs) item["inputs"].push_back(tensor->getWrapper()->getName());
    for (const auto& tensor : op->outputs) item["outputs"].push_back(tensor->getWrapper()->getName());
    manifest["operations"].push_back(std::move(item));
  }
  for (const auto& [name, tensor] : tensors) {
    const auto* native = tensor->getWrapper()->getNativeTensor();
    nlohmann::json dimensions = nlohmann::json::array();
    const auto* nativeDimensions = QNN_TENSOR_GET_DIMENSIONS(native);
    for (uint32_t i = 0; i < QNN_TENSOR_GET_RANK(native); ++i) dimensions.push_back(nativeDimensions[i]);
    const auto recipe = nlohmann::json::parse(tensor->getQuantRecipeJson());
    const auto logicalQuantDtype = recipe.value("quant_to_dtype", recipe.value("storage_dtype", tensor->getIRStorageDtype()));
    nlohmann::json item{{"name", name},
                        {"ir_storage_dtype", tensor->getIRStorageDtype()},
                        {"logical_quant_dtype", logicalQuantDtype},
                        {"qnn_dtype", qnnDataTypeName(QNN_TENSOR_GET_DATA_TYPE(native))},
                        {"qnn_dtype_id", static_cast<int64_t>(QNN_TENSOR_GET_DATA_TYPE(native))},
                        {"tensor_type", qnnTensorTypeName(QNN_TENSOR_GET_TYPE(native))},
                        {"dimensions", std::move(dimensions)},
                        {"quant_recipe", recipe},
                        {"qnn_quantization", nativeQuantizationJson(native)},
                        {"producer", producers.count(name) ? nlohmann::json(producers[name]) : nlohmann::json(nullptr)},
                        {"consumers", nlohmann::json::array()}};
    for (const auto& consumer : consumers[name]) item["consumers"].push_back(consumer);
    manifest["tensors"].push_back(std::move(item));
  }

  const auto outputPath = outputDir / (safeArtifactName(graph_name_) + "_quant_manifest.json");
  std::ofstream output(outputPath, std::ios::trunc);
  output << manifest.dump(2) << '\n';
  if (output.good()) {
    MLLM_INFO("Wrote QNN quantization manifest {} ({} ops, {} tensors)", outputPath.string(), operations.size(),
              tensors.size());
  } else {
    MLLM_ERROR("Failed to write QNN quantization manifest {}", outputPath.string());
  }
}

void QnnAOTGraph::dumpOptraceArtifacts() {
  if (qnn_interface_ == nullptr || qnn_interface_->profileGetEvents == nullptr ||
      qnn_interface_->profileGetSubEvents == nullptr || qnn_interface_->profileGetExtendedEventData == nullptr) {
    MLLM_WARN("Optrace artifact extraction APIs are unavailable for graph {}", graph_name_);
    return;
  }

  const char* outputDirEnv = std::getenv("MLLM_QNN_AOT_OPTRACE_DIR");
  const std::filesystem::path outputDir = outputDirEnv && outputDirEnv[0] ? outputDirEnv : ".";
  std::error_code fsError;
  std::filesystem::create_directories(outputDir, fsError);
  if (fsError) {
    MLLM_ERROR("Failed to create Optrace artifact directory {}: {}", outputDir.string(), fsError.message());
    return;
  }

  uint32_t artifactIndex = 0;
  // The HTP backend writes the schematic directly to the current working
  // directory during graph finalization; it is not normally returned as a
  // QNN profile event. Move it to the caller-selected artifact directory.
  const auto emittedSchematic = std::filesystem::current_path() / (graph_name_ + "_schematic.bin");
  if (std::filesystem::exists(emittedSchematic)) {
    const auto outputPath = outputDir / emittedSchematic.filename();
    if (emittedSchematic != outputPath) {
      std::filesystem::rename(emittedSchematic, outputPath, fsError);
      if (fsError) {
        fsError.clear();
        std::filesystem::copy_file(emittedSchematic, outputPath,
                                   std::filesystem::copy_options::overwrite_existing, fsError);
        if (!fsError) { std::filesystem::remove(emittedSchematic, fsError); }
      }
    }
    if (!fsError) {
      MLLM_INFO("Collected QNN Optrace schematic {}", outputPath.string());
      ++artifactIndex;
    } else {
      MLLM_ERROR("Failed to collect QNN Optrace schematic {}: {}", emittedSchematic.string(), fsError.message());
      fsError.clear();
    }
  }

  std::function<void(QnnProfile_EventId_t)> visitEvent;
  visitEvent = [&](QnnProfile_EventId_t eventId) {
    QnnProfile_ExtendedEventData_t extended = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
    if (QNN_PROFILE_NO_ERROR == qnn_interface_->profileGetExtendedEventData(eventId, &extended) &&
        extended.version == QNN_PROFILE_DATA_VERSION_1 && extended.v1.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
      const auto& objectInfo = extended.v1.backendOpaqueObject;
      const auto& object = objectInfo.opaqueObject;
      if (object.data != nullptr && object.len != 0) {
        std::string fileName = objectInfo.fileName ? std::filesystem::path(objectInfo.fileName).filename().string() : "";
        if (fileName.empty()) {
          fileName = safeArtifactName(graph_name_) + "_optrace_" + std::to_string(artifactIndex) + ".bin";
        }
        ++artifactIndex;
        const auto outputPath = outputDir / fileName;
        std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char*>(object.data), object.len);
        if (output.good()) {
          MLLM_INFO("Wrote QNN Optrace artifact {} ({} bytes)", outputPath.string(), object.len);
        } else {
          MLLM_ERROR("Failed to write QNN Optrace artifact {}", outputPath.string());
        }
      }
    }

    const QnnProfile_EventId_t* children = nullptr;
    uint32_t numChildren = 0;
    if (QNN_PROFILE_NO_ERROR == qnn_interface_->profileGetSubEvents(eventId, &children, &numChildren)) {
      for (uint32_t i = 0; i < numChildren; ++i) { visitEvent(children[i]); }
    }
  };

  const QnnProfile_EventId_t* events = nullptr;
  uint32_t numEvents = 0;
  if (QNN_PROFILE_NO_ERROR != qnn_interface_->profileGetEvents(profile_handle_, &events, &numEvents)) {
    MLLM_ERROR("Failed to retrieve Optrace finalize events for graph {}", graph_name_);
    return;
  }
  for (uint32_t i = 0; i < numEvents; ++i) { visitEvent(events[i]); }
  if (artifactIndex == 0) { MLLM_WARN("No Optrace schematic artifact was emitted for graph {}", graph_name_); }
}

const std::vector<std::string> QnnDynSymbolLoader::possible_qnn_dyn_lib_paths_{
    "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
};

QnnDynSymbolLoader::~QnnDynSymbolLoader() {
  for (auto& item : libs_) {
    if (item.second.handle_) { dlclose(item.second.handle_); }
  }
}

bool QnnDynSymbolLoader::loadQnnDynLib(const std::string& lib_name, int flag) {
  for (auto const& path : possible_qnn_dyn_lib_paths_) {
    auto real_path = path + lib_name;
    auto handle = dlopen(real_path.c_str(), flag);
    if (handle) {
      auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
      libs_.insert({lib_name, descriptor});
      MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
      return true;
    } else {
      char* error = dlerror();
      MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
    }
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

bool QnnDynSymbolLoader::loadQnnDynLibAtPath(const std::string& path, const std::string& lib_name, int flag) {
  auto real_path = path + lib_name;
  auto handle = dlopen(real_path.c_str(), flag);
  if (handle) {
    auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
    libs_.insert({lib_name, descriptor});
    MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
    return true;
  } else {
    char* error = dlerror();
    MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

QnnAOTEnv::QnnAOTEnv(const QcomTargetMachine& target_machine) : target_machine_(target_machine) { _setup(); }

QnnAOTEnv::QnnAOTEnv(const std::string& lib_path, const QcomTargetMachine& target_machine) : target_machine_(target_machine) {
  _setup(lib_path);
}

void QnnAOTEnv::_setup(const std::string& path) {
  auto& loader = QnnDynSymbolLoader::instance();
  std::string htp_backend_lib_name = "libQnnHtp.so";
  // GLOBAL Load
  if (path.empty()) {
    if (!loader.loadQnnDynLib(htp_backend_lib_name,
                              QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  } else {
    if (!loader.loadQnnDynLibAtPath(path, htp_backend_lib_name,
                                    QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  }

  auto qnn_interface_get_providers_func =
      loader(htp_backend_lib_name).func<QnnFuncSymbols::QnnInterfaceGetProvidersFuncType>("QnnInterface_getProviders");

  QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;

  MLLM_RT_ASSERT_EQ(qnn_interface_get_providers_func((const QnnInterface_t***)&interface_providers, &num_providers),
                    QNN_SUCCESS);
  MLLM_RT_ASSERT(interface_providers != nullptr);
  MLLM_RT_ASSERT(num_providers != 0);

  MLLM_INFO("QnnAOTEnv::QnnAOTEnv get HTP num_providers: {}", num_providers);

  bool found_valid_interface = false;
  // Get correct provider
  for (size_t provider_id = 0; provider_id < num_providers; provider_id++) {
    if (QNN_API_VERSION_MAJOR == interface_providers[provider_id]->apiVersion.coreApiVersion.major
        && QNN_API_VERSION_MINOR <= interface_providers[provider_id]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_htp_func_symbols_.qnn_interface_ = interface_providers[provider_id]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  MLLM_RT_ASSERT_EQ(found_valid_interface, true);

  // Check if this HTP Backend has specific property
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (status == QNN_PROPERTY_NOT_SUPPORTED) { MLLM_WARN("Device property is not supported"); }

    MLLM_RT_ASSERT(status != QNN_PROPERTY_ERROR_UNKNOWN_KEY);
  }

  // Try to config this target machine
  {
    auto device_custom_config = createDecideCustomConfigInfo();
    QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

    switch (target_machine_.soc_htp_security_pd_session) {
      case QcomSecurityPDSession::kHtpSignedPd: {
        p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
        unreachable_handle_.push_back(p_custom_config);
        p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        p_custom_config->useSignedProcessDomain.useSignedProcessDomain = true;
        p_custom_config->useSignedProcessDomain.deviceId = 0;
        device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));
        break;
      }
      case QcomSecurityPDSession::kHtpUnsignedPd:
      default: break;
    }

    const std::vector<QnnDevice_PlatformInfo_t*> device_platform_info = createDevicePlatformInfo();
    uint32_t num_custom_configs = device_platform_info.size() + device_custom_config.size();
    target_machine_qnn_config_.resize(num_custom_configs);

    for (std::size_t i = 0; i < device_custom_config.size(); ++i) {
      target_machine_qnn_config_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      target_machine_qnn_config_[i].customConfig = device_custom_config[i];
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[i]);
    }

    if (!device_platform_info.empty()) {
      // The length of platform info can only be 1.
      MLLM_RT_ASSERT_EQ(device_platform_info.size(), 1u);
      target_machine_qnn_config_[device_custom_config.size()].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
      target_machine_qnn_config_[device_custom_config.size()].hardwareInfo = device_platform_info.back();
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[device_custom_config.size()]);
    }

    // null terminated
    target_machine_qnn_config_ptrs_.push_back(nullptr);
  }
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::createContext(const std::string& name, bool weights_sharing) {
  // Check if context with this name already exists
  if (contexts_.count(name) > 0) {
    MLLM_WARN("Context '{}' already exists, reusing the existing context", name);
    return contexts_[name];
  }

  std::shared_ptr<QnnDeviceAndContext> context = std::make_shared<QnnDeviceAndContext>();
  context->name_ = name;

  // 1. create logger and register callback.
  // clang-format off
  MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.logCreate(__mllmQnnLoggerCallback,QNN_LOG_LEVEL_VERBOSE, &context->log_), QNN_SUCCESS)
  MLLM_RT_ASSERT_EQ(QNN_BACKEND_NO_ERROR, qnn_htp_func_symbols_.qnn_interface_.backendCreate(context->log_, (const QnnBackend_Config_t**)context->bk_cfg_, &context->bk_handle_))
  // clang-format on

  // 2. Create HTP Device
  // clang-format off
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.deviceCreate) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.deviceCreate(context->log_, target_machine_qnn_config_ptrs_.data(), &context->device_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }
  // clang-format on

  // 3. Create Profile
  {
    auto status = qnn_htp_func_symbols_.qnn_interface_.profileCreate(context->bk_handle_, QNN_PROFILE_LEVEL_DETAILED,
                                                                     &context->profile_bk_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
    context->optrace_enabled_ = envFlagEnabled("MLLM_QNN_AOT_OPTRACE");
    if (context->optrace_enabled_) {
      MLLM_RT_ASSERT(qnn_htp_func_symbols_.qnn_interface_.profileSetConfig != nullptr);
      MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability(
                            QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG),
                        QNN_PROPERTY_SUPPORTED);
      QnnProfile_Config_t optraceConfig = QNN_PROFILE_CONFIG_INIT;
      optraceConfig.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
      optraceConfig.enableOptrace = 1;
      const QnnProfile_Config_t* configs[] = {&optraceConfig, nullptr};
      auto configStatus =
          qnn_htp_func_symbols_.qnn_interface_.profileSetConfig(context->profile_bk_handle_, configs);
      MLLM_RT_ASSERT_EQ(configStatus, QNN_PROFILE_NO_ERROR);
      MLLM_INFO("QNN AOT Optrace enabled for context {}", name);
    }
  }

  // 4. Create Context
  {
    auto cfgs = createContextCustomConfig(weights_sharing);
    if (cfgs.size()) {
      context->qnn_context_config_ = (QnnContext_Config_t**)malloc(sizeof(QnnContext_Config_t*) * (cfgs.size() + 1));
      unreachable_handle_.emplace_back(context->qnn_context_config_);
    }
    for (int i = 0; i < cfgs.size(); ++i) {
      context->qnn_context_config_[i] = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
      context->qnn_context_config_[i]->option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      context->qnn_context_config_[i]->customConfig = cfgs[i];
      unreachable_handle_.emplace_back(context->qnn_context_config_[i]);
    }
    if (cfgs.size()) { context->qnn_context_config_[cfgs.size()] = nullptr; }
    auto status = qnn_htp_func_symbols_.qnn_interface_.contextCreate(context->bk_handle_, context->device_handle_,
                                                                     (const QnnContext_Config_t**)context->qnn_context_config_,
                                                                     &context->qnn_ctx_handle_);
    MLLM_RT_ASSERT_EQ(QNN_CONTEXT_NO_ERROR, status);
  }

  // 5. Register MLLM's Qnn Opset
  // clang-format off
  {
    // FIXME(wch): we need to register our own opset of qnn.
  }
  // clang-format on

  MLLM_RT_ASSERT_EQ(contexts_.count(name), 0);
  contexts_[name] = context;
  return context;
}

void QnnAOTEnv::saveContext(const std::string& name, const std::string& path) {
  if (contexts_.find(name) == contexts_.end()) {
    MLLM_ERROR("QnnAOTEnv::saveContext Context {} not found", name);
    return;
  }
  auto context = contexts_[name];

  uint64_t binarySize = 0;
  uint64_t writtenSize = 0;

  auto status = qnn_htp_func_symbols_.qnn_interface_.contextGetBinarySize(context->qnn_ctx_handle_, &binarySize);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);

  std::vector<uint8_t> binaryBuffer(binarySize);

  status = qnn_htp_func_symbols_.qnn_interface_.contextGetBinary(
      context->qnn_ctx_handle_, reinterpret_cast<void*>(binaryBuffer.data()), binarySize, &writtenSize);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);

  if (binarySize < writtenSize) {
    MLLM_ERROR("QNN context binary size mismatch: expected {} bytes, but wrote {} bytes.", binarySize, writtenSize);
  }

  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    MLLM_ERROR("Failed to open file {} for writing QNN context.", path);
    return;
  }
  file.write(reinterpret_cast<char*>(binaryBuffer.data()), writtenSize);
  file.close();

  MLLM_INFO("QNN context {} saved to {} written {}", name, path, writtenSize);
}

void QnnAOTEnv::destroyContext(const std::string& name) {
  // TODO
}

std::vector<QnnDevice_PlatformInfo_t*> QnnAOTEnv::createDevicePlatformInfo() {
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  QnnDevice_PlatformInfo_t* p_platform_info = nullptr;
  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = nullptr;
  QnnHtpDevice_DeviceInfoExtension_t* p_device_info_extension = nullptr;
  QnnDevice_CoreInfo_t* p_core_info = nullptr;

  p_platform_info = (QnnDevice_PlatformInfo_t*)malloc(sizeof(QnnDevice_PlatformInfo_t));
  unreachable_handle_.push_back(p_platform_info);
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  p_hw_device_info = (QnnDevice_HardwareDeviceInfo_t*)malloc(sizeof(QnnDevice_HardwareDeviceInfo_t));
  unreachable_handle_.push_back(p_hw_device_info);
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = 0;
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = 1;

  p_device_info_extension = (QnnHtpDevice_DeviceInfoExtension_t*)malloc(sizeof(QnnHtpDevice_DeviceInfoExtension_t));
  unreachable_handle_.push_back(p_device_info_extension);
  // clang-format off
  p_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  p_device_info_extension->onChipDevice.vtcmSize = target_machine_.soc_htp_vtcm_total_memory_size;  // in MB
  p_device_info_extension->onChipDevice.signedPdSupport = target_machine_.soc_htp_security_pd_session == QcomSecurityPDSession::kHtpSignedPd;
  p_device_info_extension->onChipDevice.socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  p_device_info_extension->onChipDevice.arch = static_cast<QnnHtpDevice_Arch_t>(target_machine_.soc_htp_arch);
  p_device_info_extension->onChipDevice.dlbcSupport = true;
  p_hw_device_info->v1.deviceInfoExtension = p_device_info_extension;
  // clang-format on

  p_core_info = (QnnDevice_CoreInfo_t*)malloc(sizeof(QnnDevice_CoreInfo_t));
  unreachable_handle_.push_back(p_core_info);
  p_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  p_core_info->v1.coreId = 0;
  p_core_info->v1.coreType = 0;
  p_core_info->v1.coreInfoExtension = nullptr;
  p_hw_device_info->v1.cores = p_core_info;

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.push_back(p_platform_info);

  return ret;
}

std::vector<QnnDevice_CustomConfig_t> QnnAOTEnv::createDecideCustomConfigInfo() {
  std::vector<QnnDevice_CustomConfig_t> ret;

  QnnHtpDevice_CustomConfig_t* p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
  unreachable_handle_.push_back(p_custom_config);
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  ret.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}

std::vector<QnnContext_CustomConfig_t> QnnAOTEnv::createContextCustomConfig(bool weights_sharing) {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnHtpContext_CustomConfig_t* p_custom_config = nullptr;

  if (weights_sharing) {
    p_custom_config = (QnnHtpContext_CustomConfig_t*)malloc(sizeof(QnnHtpContext_CustomConfig_t));
    unreachable_handle_.push_back(p_custom_config);
    p_custom_config->option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p_custom_config->weightSharingEnabled = true;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

  return ret;
}

QnnAOTGraph::ptr_t QnnAOTEnv::captureAOTGraph(const std::string& qnn_context_name, const std::string& g_name) {
  if (contexts_.find(qnn_context_name) == contexts_.end()) {
    MLLM_ERROR("Context {} not found", qnn_context_name);
    return nullptr;
  }
  auto& ctx = contexts_[qnn_context_name];
  if (ctx->graphs_.find(g_name) == ctx->graphs_.end()) {
    ctx->graphs_[g_name] = std::make_shared<QnnAOTGraph>(
        qnn_htp_func_symbols_.qnn_interface_, ctx->bk_handle_, ctx->qnn_ctx_handle_,
        ctx->optrace_enabled_ ? ctx->profile_bk_handle_ : nullptr, g_name);
  }
  return ctx->graphs_[g_name];
}

void QnnAOTEnv::captureAOTNodeOp(const std::string& qnn_context_name, const std::string& graph_name,
                                 const QnnAOTNodeOperation::ptr_t& op) {
  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  contexts_[qnn_context_name]->graphs_[graph_name]->addOperation(op);
}

QnnAOTNodeTensor::ptr_t QnnAOTEnv::captureQnnAOTNodeTensor(const std::string& qnn_context_name, const std::string& graph_name,
                                                           const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  auto __qnn_tensor_name = v->name();

  bool __qnn_enable_static_weight = force_static_weight;

  // Check if this value want static qnn weight. The static qnn weight will be shared through one context in diff graphs!
  if (v->tensor_.memType() == kGlobal || (v->tensor_.memType() <= kParams_End && v->tensor_.memType() >= kParams_Start)
      || v->getAttr("constant")) {
    __qnn_enable_static_weight = true;
  }

  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  auto graph = contexts_[qnn_context_name]->graphs_[graph_name];

  // If normal weight is cached, we return it directly
  if (graph->all_tensors_.count(__qnn_tensor_name)) { return graph->all_tensors_[__qnn_tensor_name]; }

  QnnAOTNodeTensor::ptr_t ret = nullptr;

  // If static weight is cached, we return it directly.
  if (__qnn_enable_static_weight) {
    if (contexts_[qnn_context_name]->static_tensor_.count(__qnn_tensor_name)) {
      ret = contexts_[qnn_context_name]->static_tensor_[__qnn_tensor_name];
    }
  }

  // There has no Tensor in the cache.
  if (ret == nullptr) {
    ret = QnnAOTNodeTensor::create(v, __qnn_enable_static_weight);

    if (__qnn_enable_static_weight) { contexts_[qnn_context_name]->static_tensor_[__qnn_tensor_name] = ret; }
  }

  graph->addTensor(ret);

  return ret;
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::getContext(const std::string& name) { return contexts_[name]; }

}  // namespace mllm::qnn::aot
