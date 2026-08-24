// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <array>
#include <string>

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpDef.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

namespace {

constexpr auto kPackageName = THIS_PKG_NAME_STR;
constexpr auto kOpName = "U8S8HmxMatMul";
constexpr auto kMixedResourceOpName = "MixedResourceHmxHvxHmx";
std::array<const char*, 2> op_names{{kOpName, kMixedResourceOpName}};
Qnn_ApiVersion_t sdk_api_version = QNN_HTP_API_VERSION_INIT;
Qnn_Version_t opset_version = {QNN_OPSET_VERSION_MAJOR, QNN_OPSET_VERSION_MINOR, QNN_OPSET_VERSION_PATCH};
QnnOpPackage_Info_t package_info = {kPackageName,     op_names.data(),  nullptr, op_names.size(), nullptr, 0,
                                    QNN_SDK_BUILD_ID, &sdk_api_version, nullptr, &opset_version,  {0}};
QnnOpPackage_GlobalInfrastructure_t global_infrastructure = nullptr;
bool package_initialized = false;
QnnLog_Callback_t log_callback = nullptr;
QnnLog_Level_t max_log_level = static_cast<QnnLog_Level_t>(0);

}  // namespace

extern const QHPI_OpInfo_v1* u8s8_hmx_matmul_op_info();
extern const QHPI_OpInfo_v1* mixed_resource_hmx_hvx_hmx_op_info();

Qnn_ErrorHandle_t QhpiHmxProbePackageInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  global_infrastructure = infrastructure;
  package_initialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageGetInfo(const QnnOpPackage_Info_t** info) {
  if (!package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (info == nullptr) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  *info = &package_info;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageValidateOpConfig(Qnn_OpConfig_t config) {
  const bool known_op = config.version == QNN_OPCONFIG_VERSION_1 && config.v1.typeName != nullptr
                        && (std::string(config.v1.typeName) == kOpName
                            || std::string(config.v1.typeName) == kMixedResourceOpName);
  if (config.version != QNN_OPCONFIG_VERSION_1 || config.v1.packageName == nullptr || config.v1.typeName == nullptr
      || std::string(config.v1.packageName) != kPackageName || !known_op
      || config.v1.numOfParams != 0 || config.v1.numOfInputs != 2 || config.v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageCreateOpImpl(QnnOpPackage_GraphInfrastructure_t /*infrastructure*/,
                                                  QnnOpPackage_Node_t /*node*/, QnnOpPackage_OpImpl_t* /*implementation*/) {
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageFreeOpImpl(QnnOpPackage_OpImpl_t /*implementation*/) {
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageLogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t level) {
  if (callback == nullptr || level < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  log_callback = callback;
  max_log_level = level;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageLogSetLevel(QnnLog_Level_t level) {
  if (level < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  max_log_level = level;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageLogTerminate() {
  log_callback = nullptr;
  max_log_level = static_cast<QnnLog_Level_t>(0);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QhpiHmxProbePackageTerminate() {
  if (!package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  global_infrastructure = nullptr;
  package_initialized = false;
  return QNN_SUCCESS;
}

extern "C" QNN_API Qnn_ErrorHandle_t QhpiHmxProbePackageInterfaceProvider(QnnOpPackage_Interface_t* interface) {
  if (interface == nullptr) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = QhpiHmxProbePackageInit;
  interface->v1_4.terminate = QhpiHmxProbePackageTerminate;
  interface->v1_4.getInfo = QhpiHmxProbePackageGetInfo;
  interface->v1_4.validateOpConfig = QhpiHmxProbePackageValidateOpConfig;
  interface->v1_4.createOpImpl = QhpiHmxProbePackageCreateOpImpl;
  interface->v1_4.freeOpImpl = QhpiHmxProbePackageFreeOpImpl;
  interface->v1_4.logInitialize = QhpiHmxProbePackageLogInitialize;
  interface->v1_4.logSetLevel = QhpiHmxProbePackageLogSetLevel;
  interface->v1_4.logTerminate = QhpiHmxProbePackageLogTerminate;
  return QNN_SUCCESS;
}

extern "C" QNN_API const char* qhpi_init() {
  static std::array<QHPI_OpInfo_v1, 2> registered_ops{{*u8s8_hmx_matmul_op_info(),
                                                       *mixed_resource_hmx_hvx_hmx_op_info()}};
  qhpi_register_ops_v1(registered_ops.size(), registered_ops.data(), kPackageName);
  return kPackageName;
}
