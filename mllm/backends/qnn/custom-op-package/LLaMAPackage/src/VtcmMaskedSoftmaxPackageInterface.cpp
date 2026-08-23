// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Minimal x86 prepare package for the QAIRT 2.49 Softmax experiment.  The
// device package keeps the normal LLaMAPackage interface; this host-only
// interface avoids loading unrelated historical op definitions into the
// version-isolated graph finalizer.

#include <array>
#include <string>

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_VtcmMaskedE2SoftmaxHd128)
END_PKG_OPS_OPTS_LIST()

namespace {

constexpr auto kPackageName = THIS_PKG_NAME_STR;
constexpr auto kOpName = "VtcmMaskedE2SoftmaxHd128";
std::array<const char*, 1> op_names{{kOpName}};
Qnn_ApiVersion_t sdk_api_version = QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t package_info = QNN_OP_PACKAGE_INFO_INIT;
QnnOpPackage_GlobalInfrastructure_t global_infrastructure = nullptr;
bool package_initialized = false;
QnnLog_Callback_t log_callback = nullptr;
QnnLog_Level_t max_log_level = static_cast<QnnLog_Level_t>(0);
bool log_initialized = false;

}  // namespace

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()
INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t softmaxSimdPackageInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (package_initialized) { return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED; }
  REGISTER_PACKAGE_PARAM_ORDERS()
  REGISTER_PACKAGE_AXIS_PARAMS()
  REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
  global_infrastructure = infrastructure;
  package_initialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageGetInfo(const QnnOpPackage_Info_t** info) {
  if (!package_initialized) { return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED; }
  if (info == nullptr) { return QNN_OP_PACKAGE_ERROR_INVALID_INFO; }
  package_info = QNN_OP_PACKAGE_INFO_INIT;
  package_info.packageName = kPackageName;
  package_info.operationNames = op_names.data();
  package_info.numOperations = op_names.size();
  package_info.sdkBuildId = QNN_SDK_BUILD_ID;
  package_info.sdkApiVersion = &sdk_api_version;
  *info = &package_info;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageValidateOpConfig(Qnn_OpConfig_t config) {
  if (config.v1.packageName == nullptr || config.v1.typeName == nullptr
      || std::string(kPackageName) != config.v1.packageName || std::string(kOpName) != config.v1.typeName
      || config.v1.numOfParams != 0 || config.v1.numOfInputs != 2 || config.v1.numOfOutputs != 1) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageLogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t level) {
  if (callback == nullptr || level < QNN_LOG_LEVEL_ERROR) { return QNN_LOG_ERROR_INVALID_ARGUMENT; }
  log_callback = callback;
  max_log_level = level;
  log_initialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageLogSetLevel(QnnLog_Level_t level) {
  if (level < QNN_LOG_LEVEL_ERROR) { return QNN_LOG_ERROR_INVALID_ARGUMENT; }
  max_log_level = level;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageLogTerminate() {
  log_callback = nullptr;
  max_log_level = static_cast<QnnLog_Level_t>(0);
  log_initialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t softmaxSimdPackageTerminate() {
  if (!package_initialized) { return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED; }
  global_infrastructure = nullptr;
  package_initialized = false;
  return QNN_SUCCESS;
}

extern "C" Qnn_ErrorHandle_t LLaMASoftmaxSimdPackageInterfaceProvider(QnnOpPackage_Interface_t* interface) {
  if (interface == nullptr) { return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT; }
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = softmaxSimdPackageInit;
  interface->v1_4.terminate = softmaxSimdPackageTerminate;
  interface->v1_4.getInfo = softmaxSimdPackageGetInfo;
  interface->v1_4.validateOpConfig = softmaxSimdPackageValidateOpConfig;
  interface->v1_4.createOpImpl = nullptr;
  interface->v1_4.freeOpImpl = nullptr;
  interface->v1_4.logInitialize = softmaxSimdPackageLogInitialize;
  interface->v1_4.logSetLevel = softmaxSimdPackageLogSetLevel;
  interface->v1_4.logTerminate = softmaxSimdPackageLogTerminate;
  return QNN_SUCCESS;
}
