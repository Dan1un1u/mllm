#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <chrono>

#include "QnnLog.h"
#include "HTP/QnnHtpProfile.h"

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/op/QNNCastTypeOp.hpp"
#include "mllm/backends/qnn/op/QNNElewiseOp.hpp"
#include "mllm/backends/qnn/op/QNNEmbeddingOp.hpp"
#include "mllm/backends/qnn/op/QNNGraphOp.hpp"
#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "mllm/backends/qnn/op/QNNParamOp.hpp"
#include "mllm/backends/qnn/op/QNNRMSNormOp.hpp"
#include "mllm/backends/qnn/op/QNNSiLUOp.hpp"
#include "mllm/backends/qnn/op/QNNTransposeOp.hpp"
#include "mllm/backends/qnn/op/QNNViewOp.hpp"
#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

namespace {

uint64_t monotonicTimeUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string getEnvString(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? fallback : std::string(value);
}

uint64_t getEnvUint64(const char* name, uint64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return fallback; }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  return end != value && *end == '\0' ? parsed : fallback;
}

bool getEnvBool(const char* name, bool fallback) {
  std::string value = getEnvString(name, fallback ? "1" : "0");
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  if (value == "1" || value == "true" || value == "yes" || value == "on") { return true; }
  if (value == "0" || value == "false" || value == "no" || value == "off") { return false; }
  return fallback;
}

ProfilingLevel getProfilingLevelFromEnv() {
  std::string value = getEnvString("MLLM_QNN_PROFILE_LEVEL", "linting");
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  if (value == "off") { return ProfilingLevel::OFF; }
  if (value == "basic") { return ProfilingLevel::BASIC; }
  if (value == "detailed") { return ProfilingLevel::DETAILED; }
  if (value == "linting") { return ProfilingLevel::LINTING; }
  if (value == "optrace") { return ProfilingLevel::OPTRACE; }
  MLLM_WARN("Unknown MLLM_QNN_PROFILE_LEVEL='{}'; using linting", value);
  return ProfilingLevel::LINTING;
}

const char* profilingLevelName(ProfilingLevel level) {
  switch (level) {
    case ProfilingLevel::OFF: return "off";
    case ProfilingLevel::BASIC: return "basic";
    case ProfilingLevel::DETAILED: return "detailed";
    case ProfilingLevel::LINTING: return "linting";
    case ProfilingLevel::OPTRACE: return "optrace";
    default: return "invalid";
  }
}

const char* profileUnitName(QnnProfile_EventUnit_t unit) {
  switch (unit) {
    case QNN_PROFILE_EVENTUNIT_MICROSEC: return "us";
    case QNN_PROFILE_EVENTUNIT_BYTES: return "bytes";
    case QNN_PROFILE_EVENTUNIT_CYCLES: return "cycles";
    case QNN_PROFILE_EVENTUNIT_COUNT: return "count";
    case QNN_PROFILE_EVENTUNIT_OBJECT: return "object";
    case QNN_PROFILE_EVENTUNIT_NONE: return "none";
    default: return "backend";
  }
}

const char* profileTypeName(QnnProfile_EventType_t type) {
  switch (type) {
    case QNN_PROFILE_EVENTTYPE_INIT: return "init";
    case QNN_PROFILE_EVENTTYPE_FINALIZE: return "finalize";
    case QNN_PROFILE_EVENTTYPE_EXECUTE: return "execute";
    case QNN_PROFILE_EVENTTYPE_NODE: return "node";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT: return "queue_wait";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS: return "preprocess";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE: return "device";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS: return "postprocess";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT: return "node_wait";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_OVERLAP: return "node_overlap";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_OVERLAP: return "node_wait_overlap";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_RESOURCEMASK: return "node_resources";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_CRITICAL_BG_OP_ID: return "critical_bg_op";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_BG_OP_ID: return "wait_bg_op";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_CRITICAL_ACCEL_TIME_CYCLE: return "critical_path";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_NUMBER_OF_HVX_THREADS: return "hvx_threads";
    default: return type >= QNN_PROFILE_EVENTTYPE_BACKEND ? "backend" : "unknown";
  }
}

std::string htpResourceMaskName(uint64_t mask) {
  std::string resources;
  const auto append = [&](const char* resource) {
    if (!resources.empty()) { resources += ','; }
    resources += resource;
  };
  if ((mask & 0x1U) != 0) { append("HVX"); }
  if ((mask & 0x2U) != 0) { append("HMX"); }
  if ((mask & 0x4U) != 0) { append("DMA"); }
  return resources.empty() ? "NONE" : resources;
}

}  // namespace

QNNBackend::QNNBackend() : Backend(kQNN, createQNNAllocator()) {
  // register ops
  regOpFactory<QNNAddOpFactory, QNNMulOpFactory, QNNGraphBeginOpFactory, QNNGraphEndOpFactory, QNNLinearOpFactory,
               QNNViewOpFactory, QNNRMSNormOpFactory, QNNTransposeOpFactory, QNNX2XOpFactory, QNNCastTypeOpFactory,
               QNNParamOpFactory, QNNSiLUOpFactory, QNNEmbeddingOpFactory>();

  QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_ERROR;  // default QNN log level
  profilingLevel_ = getProfilingLevelFromEnv();
  profilingWarmup_ = getEnvUint64("MLLM_QNN_PROFILE_WARMUP", 0);
  profilingEvery_ = std::max<uint64_t>(1, getEnvUint64("MLLM_QNN_PROFILE_EVERY", 1));
  profilingMaxCaptures_ = getEnvUint64("MLLM_QNN_PROFILE_MAX_CAPTURES", 1);
  if (ProfilingLevel::OPTRACE == profilingLevel_ &&
      (profilingWarmup_ != 0 || profilingEvery_ != 1 || profilingMaxCaptures_ != 1)) {
    MLLM_WARN("HTP Optrace must be attached on a graph's first execution and supports one isolated payload per "
              "process; forcing warmup=0, every=1, max captures=1");
    profilingWarmup_ = 0;
    profilingEvery_ = 1;
    profilingMaxCaptures_ = 1;
  }
  profilingFinalize_ = getEnvBool("MLLM_QNN_PROFILE_FINALIZE", false);
  profilingSerializationEnabled_ = getEnvBool("MLLM_QNN_PROFILE_SERIALIZE", true);
  profilingDirectory_ = getEnvString("MLLM_QNN_PROFILE_DIR", "/data/local/tmp");
  profilingGraphFilter_ = getEnvString("MLLM_QNN_PROFILE_GRAPH", "");
  profilingDetailPath_ = profilingDirectory_ + "/qnn_detail_profile.txt";
  profilingMacroPath_ = profilingDirectory_ + "/qnn_macro_profile.csv";
  profilingSerializedPath_ = profilingDirectory_ + "/qnn-profiling-data.log";
  debug_ = false;  // when set true, NATIVE tensor will be regared as APP_READ tensor

  // Load QNN libraries and hold handles for lifecycle management
  auto [qnnSuccess, qnnHandle] = loadQNNSymbol();
  if (!qnnSuccess) { MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to load QNN symbols"); }
  qnnHtpLibHandle_ = qnnHandle;
  MLLM_INFO("QNN symbols loaded successfully");

  auto [sysSuccess, sysHandle] = loadQNNSystemSymbol();
  if (!sysSuccess) { MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to load QNN System symbols"); }
  qnnSystemLibHandle_ = sysHandle;
  MLLM_INFO("QNN System symbols loaded successfully");

  runtime_ = QNNRuntime::create(profilingLevel_, qnnLogLevel);
  if (!runtime_) {
    MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to create QNN Runtime");
  } else {
    MLLM_INFO("QNN Runtime created successfully");
  }

  // check QNN capability, detect QNN features for future use
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS != runtime_->qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    MLLM_ERROR("Unable to get build Id from the backend.");
  }
  backendBuildId_ = backendBuildId == nullptr ? "" : backendBuildId;
  MLLM_INFO("QNN Backend Build Id: {}", backendBuildId_);
  profilingExtendedEventsSupported_ = runtime_->qnnInterface.profileGetExtendedEventData != nullptr &&
                                      runtime_->qnnInterface.propertyHasCapability(
                                          QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT) == QNN_PROPERTY_SUPPORTED;
  if (ProfilingLevel::OFF != profilingLevel_) {
    std::ofstream(profilingDetailPath_, std::ios::trunc)
        << "# level=" << profilingLevelName(profilingLevel_) << " warmup=" << profilingWarmup_
        << " every=" << profilingEvery_ << " max_captures_per_graph=" << profilingMaxCaptures_
        << " extended_events=" << profilingExtendedEventsSupported_ << "\n";
    std::ofstream(profilingMacroPath_, std::ios::trunc)
        << "graph,execution,profiled,captured,graph_execute_us\n";
    std::ofstream(profilingDirectory_ + "/qnn_e2e_profile.csv", std::ios::trunc)
        << "phase,graph,chunk,module_execute_us\n";
    initializeProfilingSerialization();
    MLLM_INFO("QNN profiling: level={}, warmup={}, every={}, max captures/graph={}, graph filter={}, output={}",
              profilingLevelName(profilingLevel_), profilingWarmup_, profilingEvery_, profilingMaxCaptures_,
              profilingGraphFilter_.empty() ? "<all>" : profilingGraphFilter_, profilingDirectory_);
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports tensor sparsity");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports dynamic dimensions");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports early termination");
  }

  // set performance parameters for better performance on HTP
  perf_ = QNNPerf::create(&runtime_->qnnInterface);
  perf_->setPowerConfigBurst();
  perf_->setRpcLatencyAndPolling();
  MLLM_INFO("QNN Perf created successfully");
}

QNNBackend::~QNNBackend() {
  // Cleanup order is critical - we hold all QNN library handles to control unload order:
  // 1. Allocator shutdown (memDeRegister + rpcmem_free) - needs QNN alive
  // 2. Clear models - tensor destructors try to free but allocator is shut down
  // 3. Perf cleanup - needs QNN HTP infrastructure alive
  // 4. Runtime cleanup - frees QNN backend/device handles
  // 5. Allocator reset - dlcloses libcdsprpc.so (held by allocator)
  // 6. Close QNN libraries - libQnnSystem.so first, then libQnnHtp.so

  // 1. Properly shutdown allocator while QNN is still alive
  //    This calls memDeRegister and rpcmem_free safely
  if (allocator_) {
    auto* qnnAllocator = dynamic_cast<QNNAllocator*>(allocator_.get());
    if (qnnAllocator) { qnnAllocator->shutdown(); }
  }

  // 2. Clear models - tensor destructors will call free() but they're now no-ops
  qnnModels_.clear();
  qnnModelIndexMap_.clear();

  // 3. Cleanup perf while QNN HTP infrastructure is still alive
  if (perf_) { perf_->shutdown(); }
  perf_.reset();

  // 4. Cleanup runtime - frees QNN backend/device handles
  if (profilingSerializationHandle_ != nullptr &&
      runtime_->qnnSystemInterface.systemProfileFreeSerializationTarget != nullptr) {
    runtime_->qnnSystemInterface.systemProfileFreeSerializationTarget(profilingSerializationHandle_);
    profilingSerializationHandle_ = nullptr;
  }
  runtime_->qnnInterface.contextFree(context_, nullptr);
  context_ = nullptr;
  runtime_.reset();

  // 5. Reset allocator - will dlclose libcdsprpc.so since shutdown() was already called
  allocator_.reset();

  // 6. Close QNN libraries in reverse order of dependency
  if (qnnSystemLibHandle_) {
    dlclose(qnnSystemLibHandle_);
    qnnSystemLibHandle_ = nullptr;
  }
  if (qnnHtpLibHandle_) {
    dlclose(qnnHtpLibHandle_);
    qnnHtpLibHandle_ = nullptr;
  }
}

QNNPerf::QNNPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface) {
  assert(qnnInterface != nullptr);
  qnnInterface_ = qnnInterface;

  QnnDevice_Infrastructure_t deviceInfra = nullptr;
  CALL_QNN(qnnInterface_->deviceGetInfrastructure(&deviceInfra));
  QnnHtpDevice_Infrastructure_t* htpInfra = static_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfra);
  perfInfra_ = htpInfra->perfInfra;

  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  CALL_QNN(perfInfra_.createPowerConfigId(deviceId, coreId, &powerConfigId_));

  powerConfigBurst_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = powerConfigId_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 0,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 40,    // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 0,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          },
  };

  powerConfigBalanced_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = powerConfigId_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 1,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 1000,  // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
          },
  };
}

void QNNPerf::shutdown() {
  if (isShutdown_) return;
  isShutdown_ = true;
  CALL_QNN(perfInfra_.destroyPowerConfigId(powerConfigId_));
}

QNNPerf::~QNNPerf() {
  // If shutdown() was already called, skip cleanup
  // This prevents crashes during program exit when QNN HTP infrastructure might be destroyed
  if (!isShutdown_) { shutdown(); }
}

void QNNPerf::setRpcLatencyAndPolling() {
  // set RPC Control Latency
  QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
  rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpcControlLatency.rpcControlLatencyConfig = 100;  // use rpc control latency recommended 100 us, refer hexagon sdk
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs1[] = {&rpcControlLatency, nullptr};

  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs1));  // set RPC latency config on power config ID created

  // set RPC Polling
  QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
  rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpcPollingTime.rpcPollingTimeConfig = 9999;  // use rpc polling time recommended 0-10000 us
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, nullptr};

  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs2));  // set RPC polling config on power config ID created
}

void QNNPerf::setPowerConfigBurst() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfigBurst_, nullptr};
  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs));
}

void QNNPerf::setPowerConfigBalanced() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfigBalanced_, nullptr};
  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs));
}

QNNRuntime::~QNNRuntime() {
  // Free Profile
  if (profileHandle != nullptr) { CALL_QNN(qnnInterface.profileFree(profileHandle)); }

  // Free Device
  CALL_QNN(qnnInterface.deviceFree(deviceHandle));

  // Free Backend
  CALL_QNN(qnnInterface.backendFree(backendHandle));

  // Free Log
  CALL_QNN(qnnInterface.logFree(logHandle));
}

QNNRuntime* QNNRuntime::initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel) {
  // Create Interface
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  {
    QnnInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    if (QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
      MLLM_ERROR("Failed to get QNN interface providers.");
      return nullptr;
    }
    if (interfaceProviders == nullptr) {
      MLLM_ERROR("Failed to get interface providers: null interface providers received.");
      return nullptr;
    }
    if (numProviders == 0) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major
          /*&& QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor*/) {
        foundValidInterface = true;
        qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!foundValidInterface) {
      MLLM_ERROR("Failed to find a valid QNN interface provider.");
      return nullptr;
    }
  }

  // Create Log
  Qnn_LogHandle_t logHandle = nullptr;
  {
    QnnLog_Callback_t logCallback = __mllmQnnLoggerCallback;
    if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, qnnLogLevel, &logHandle)) != QNN_SUCCESS)
        || (logHandle == nullptr)) {
      MLLM_ERROR("Failed to initialize logging in the backend.");
      return nullptr;
    } else {
      MLLM_INFO("Logging initialized successfully");
    }
  }

  // Create Backend
  Qnn_BackendHandle_t backendHandle = nullptr;
  {
    const QnnBackend_Config_t** backendConfig = nullptr;
    if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS)
        || (backendHandle == nullptr)) {
      MLLM_ERROR("Failed to create the backend.");
      return nullptr;
    } else {
      MLLM_INFO("Backend created successfully");
    }
  }

  // Create Device
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  {
    // Check whether the device API is supported.
    if (nullptr != qnnInterface.deviceCreate) {
      auto status = qnnInterface.deviceCreate(logHandle, nullptr, &deviceHandle);
      if (QNN_SUCCESS != status) {
        MLLM_ERROR("Failed to create device, error: {}", (int)status);
        return nullptr;
      }
      MLLM_INFO("Device created successfully");
    }
  }

  // Initialize Profiling
  Qnn_ProfileHandle_t profileHandle = nullptr;
  {
    if (ProfilingLevel::OFF != profilingLevel) {
      MLLM_INFO("Profiling turned on; level = {}", (int)profilingLevel);
      if (ProfilingLevel::BASIC == profilingLevel) {
        MLLM_INFO("Basic profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_BASIC, &profileHandle)) {
          MLLM_WARN("Unable to create profile handle in the backend.");
          return nullptr;
        }
      } else if (ProfilingLevel::DETAILED == profilingLevel) {
        MLLM_INFO("Detailed profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_DETAILED, &profileHandle)) {
          MLLM_ERROR("Unable to create profile handle in the backend.");
          return nullptr;
        }
      } else if (ProfilingLevel::LINTING == profilingLevel) {
        MLLM_INFO("HTP linting profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR !=
            qnnInterface.profileCreate(backendHandle, QNN_HTP_PROFILE_LEVEL_LINTING, &profileHandle)) {
          MLLM_ERROR("Unable to create HTP linting profile handle in the backend.");
          return nullptr;
        }
      } else if (ProfilingLevel::OPTRACE == profilingLevel) {
        MLLM_INFO("HTP Optrace requested. Creating a detailed Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR !=
            qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_DETAILED, &profileHandle)) {
          MLLM_ERROR("Unable to create detailed profile handle for HTP Optrace.");
          return nullptr;
        }
        if (qnnInterface.profileSetConfig == nullptr ||
            qnnInterface.propertyHasCapability(QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG) != QNN_PROPERTY_SUPPORTED) {
          MLLM_ERROR("The loaded QNN HTP backend does not support Optrace profile configuration.");
          qnnInterface.profileFree(profileHandle);
          return nullptr;
        }
        QnnProfile_Config_t optraceConfig = QNN_PROFILE_CONFIG_INIT;
        optraceConfig.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
        optraceConfig.enableOptrace = 1;
        const QnnProfile_Config_t* configs[] = {&optraceConfig, nullptr};
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileSetConfig(profileHandle, configs)) {
          MLLM_ERROR("Failed to enable HTP Optrace on the QNN profile handle.");
          qnnInterface.profileFree(profileHandle);
          return nullptr;
        }
      }
    }
  }

  // Register Custom OpPackages
  {
    struct OpPackageInfo {
      std::string path;
      std::string interfaceProvider;
      std::string target;
    };

    std::vector<OpPackageInfo> opPackages = {};

    for (const auto& pkg : opPackages) {
      if (!qnnInterface.backendRegisterOpPackage) {
        MLLM_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
        return nullptr;
      }
      if (QNN_BACKEND_NO_ERROR
          != qnnInterface.backendRegisterOpPackage(backendHandle, pkg.path.c_str(), pkg.interfaceProvider.c_str(),
                                                   pkg.target.c_str())) {
        MLLM_ERROR("Could not register Op Package: {} and interface provider: {}", pkg.path.c_str(),
                   pkg.interfaceProvider.c_str());
        return nullptr;
      }
      MLLM_INFO("Registered Op Package: {} and interface provider: {}", pkg.path.c_str(), pkg.interfaceProvider.c_str());
    }
  }

  // Create QNN System Interface
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
  {
    QnnSystemInterface_t** systemInterfaceProviders{nullptr};
    uint32_t numProviders{0};
    if (QNN_SUCCESS
        != QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&systemInterfaceProviders, &numProviders)) {
      MLLM_ERROR("Failed to get system interface providers.");
      return nullptr;
    } else {
      MLLM_INFO("System interface providers found: {}", numProviders);
    }
    if (0 == numProviders) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidSystemInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[pIdx]->systemApiVersion.major
          && QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
        qnnSystemInterface = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
        foundValidSystemInterface = true;
        break;
      } else {
        // Print system interface provider and self version
        MLLM_WARN("System interface provider: {} version: {}", systemInterfaceProviders[pIdx]->systemApiVersion.major,
                  systemInterfaceProviders[pIdx]->systemApiVersion.minor);
        MLLM_WARN("Self version: {} {}", QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
        MLLM_WARN("Unable to find a valid system interface.");
      }
    }
    if (!foundValidSystemInterface) {
      MLLM_ERROR("Unable to find a valid system interface.");
      return nullptr;
    }
  }

  return new QNNRuntime(qnnInterface, qnnSystemInterface, logHandle, backendHandle, deviceHandle, profileHandle);
}

bool QNNRuntime::createContext(Qnn_ContextHandle_t& context, QnnContext_Config_t** contextConfig) {
  if (QNN_CONTEXT_NO_ERROR
      != qnnInterface.contextCreate(backendHandle, deviceHandle, (const QnnContext_Config_t**)&contextConfig, &context)) {
    MLLM_ERROR("Could not create context");
    return false;
  }
  return true;
}

bool QNNRuntime::retrieveContext(const std::string& contextBinaryPath, Qnn_ContextHandle_t& context,
                                 std::vector<std::shared_ptr<QNNModel>>& qnnModels, QnnContext_Config_t** contextConfig) {
  // Read the binary from qnn_context.bin and get the size in byte
  std::ifstream file(contextBinaryPath, std::ios::binary | std::ios::ate);
  if (!file.is_open() || !file.good()) {
    MLLM_ERROR("Could not open context binary file: {}", contextBinaryPath);
    return false;
  } else {
    MLLM_INFO("Context binary file opened successfully: {}", contextBinaryPath);
  }
  std::streamsize size = file.tellg();
  MLLM_INFO("Context binary file size: {} MB", size / 1024 / 1024);
  file.seekg(0, std::ios::beg);

  auto binaryBuffer = std::make_unique<uint8_t[]>(size);

  file.read(reinterpret_cast<char*>(binaryBuffer.get()), size);
  file.close();

  // inspect binary info
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (!qnnSystemInterface.systemContextCreate) {
    MLLM_ERROR("systemContextCreate is nullptr.");
    return false;
  }
  if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    MLLM_ERROR("Could not create system handle.");
    return false;
  } else {
    MLLM_INFO("System context created successfully");
  }

  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};

  if (QNN_SUCCESS
      != qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle, static_cast<void*>(binaryBuffer.get()), size, &binaryInfo,
                                                       &binaryInfoSize)) {
    MLLM_ERROR("Failed to get context binary info");
    return false;
  } else {
    MLLM_INFO("Context binary info retrieved successfully");
  }

  // Extract graph metadata to create QNNModels instead of GraphInfo_t
  GraphInfo_t** tmpGraphsInfo = nullptr;
  uint32_t graphNum;
  // fill GraphInfo_t based on binary info - temporarily needed for tensor extraction
  if (!copyMetadataToGraphsInfo(binaryInfo, tmpGraphsInfo, graphNum)) {
    MLLM_ERROR("Failed to copy metadata.");
    return false;
  }
  if (QNN_SUCCESS != qnnSystemInterface.systemContextFree(sysCtxHandle)) {
    MLLM_ERROR("Could not free system context.");
    return false;
  } else {
    MLLM_INFO("System context freed successfully");
  }
  sysCtxHandle = nullptr;

  // Create context from binary
  Qnn_ContextBinarySize_t writtenSize = 0;
  if (QNN_CONTEXT_NO_ERROR
      != qnnInterface.contextCreateFromBinary(backendHandle, deviceHandle, (const QnnContext_Config_t**)contextConfig,
                                              binaryBuffer.get(), size, &context, profileHandle)) {
    MLLM_ERROR("Could not create context from binary. Mostly due to binary's qnn version mismatch with backend's qnn version.");
    return false;
  } else {
    MLLM_INFO("Context created from binary successfully");
  }

  // Create QNNModels for each graph and initialize from context
  qnnModels.clear();
  qnnModels.reserve(graphNum);

  for (uint32_t i = 0; i < graphNum; ++i) {
    GraphInfo_t* graphInfo = tmpGraphsInfo[i];

    // Retrieve the graph handle
    Qnn_GraphHandle_t graph = nullptr;
    if (QNN_SUCCESS != qnnInterface.graphRetrieve(context, graphInfo->graphName, &graph)) {
      MLLM_ERROR("Unable to retrieve graph handle for graph: {}", graphInfo->graphName);
      return false;
    }

    // Create QNNModel and initialize from context
    auto qnnModel = std::make_shared<QNNModel>(qnnInterface, backendHandle);
    ModelError_t err =
        qnnModel->initializeFromContext(context, graphInfo->graphName, graph, graphInfo->inputTensors,
                                        graphInfo->numInputTensors, graphInfo->outputTensors, graphInfo->numOutputTensors);

    if (err != MODEL_NO_ERROR) {
      MLLM_ERROR("Failed to initialize QNNModel from context for graph: {} with error: {}", graphInfo->graphName,
                 static_cast<int>(err));
      return false;
    }

    qnnModels.push_back(qnnModel);
    MLLM_INFO("Successfully created QNNModel for graph: {}", graphInfo->graphName);
  }

  // Clean up temporary GraphInfo_t structures
  for (uint32_t i = 0; i < graphNum; ++i) {
    if (tmpGraphsInfo[i]) {
      if (tmpGraphsInfo[i]->graphName) { free(tmpGraphsInfo[i]->graphName); }
      freeQnnTensors(tmpGraphsInfo[i]->inputTensors, tmpGraphsInfo[i]->numInputTensors);
      freeQnnTensors(tmpGraphsInfo[i]->outputTensors, tmpGraphsInfo[i]->numOutputTensors);
    }
  }
  if (graphNum > 0 && tmpGraphsInfo[0]) { free(tmpGraphsInfo[0]); }
  if (tmpGraphsInfo) { free(tmpGraphsInfo); }

  MLLM_INFO("QNN context retrieved from qnn_context.bin with {} QNNModels(QnnGraphs)", graphNum);
  return true;
}

bool QNNBackend::createContext() {
  if (!runtime_->createContext(context_, nullptr)) { return false; }
  // init QNN Allocator
  static_pointer_cast<QNNAllocator>(allocator_)->setQNNPointer(runtime_->qnnInterface, context_);
  return true;
}

bool QNNBackend::loadContext(const std::string& contextPath) {
  if (!runtime_->retrieveContext(contextPath, context_, qnnModels_, nullptr)) { return false; }
  // fill qnnModelIndexMap_ info according to qnnModels_
  for (size_t i = 0; i < qnnModels_.size(); i++) {
    auto graphName = qnnModels_[i]->getQnnGraphName();
    qnnModelIndexMap_.insert(std::make_pair(graphName, i));
  }
  // init QNN Allocator
  static_pointer_cast<QNNAllocator>(allocator_)->setQNNPointer(runtime_->qnnInterface, context_);
  return true;
}

void QNNBackend::saveContext(const std::string& contextPath) {
  uint64_t binarySize, writtenSize;

  runtime_->qnnInterface.contextGetBinarySize(context_, &binarySize);

  std::unique_ptr<uint8_t[]> binaryBuffer(new uint8_t[binarySize]);

  runtime_->qnnInterface.contextGetBinary(context_, reinterpret_cast<void*>(binaryBuffer.get()), binarySize, &writtenSize);

  if (binarySize < writtenSize) {
    MLLM_ERROR("QNN context binary size mismatch. Written {}  bytes, expected {} bytes.", writtenSize, binarySize);
  }
  std::ofstream file(contextPath, std::ios::binary);
  file.write(reinterpret_cast<char*>(binaryBuffer.get()), writtenSize);
  file.close();

  MLLM_INFO("QNN context saved to {} written {} bytes.", contextPath, writtenSize);
}

std::shared_ptr<QNNModel> QNNBackend::createQnnGraph(const std::string& graphName) {
  // If the graph already exists, return the existing model
  if (qnnModelIndexMap_.find(graphName) != qnnModelIndexMap_.end()) {
    currentQnnModelIndex_ = qnnModelIndexMap_[graphName];
    return qnnModels_[currentQnnModelIndex_];
  }

  // Create a new QNNModel
  currentQnnModelIndex_ = static_cast<int>(qnnModels_.size());
  qnnModelIndexMap_.insert(std::make_pair(graphName, currentQnnModelIndex_));

  auto qnnModel = std::make_shared<QNNModel>(runtime_->qnnInterface, runtime_->backendHandle);
  qnnModels_.push_back(qnnModel);

  // Initialize QNN graph info with basic configs
  const QnnGraph_Config_t* graphConfigList[] = {nullptr};

  ModelError_t err = MODEL_NO_ERROR;
  if ((err = qnnModel->initialize(context_, graphName.c_str(), debug_, 1, graphConfigList)) != MODEL_NO_ERROR) {
    MLLM_ERROR("QNN graph initialization failed for graph: {} with error code: {}", graphName, static_cast<int>(err));
    qnnModels_.pop_back();
    qnnModelIndexMap_.erase(graphName);
    return nullptr;
  }

  return qnnModel;
}

void QNNBackend::graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                              const std::vector<std::string>& inputTensorNames,
                              const std::vector<std::string>& outputTensorNames,
                              const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                              const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                              const std::string& packageName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding node", graphName);
    return;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) { return; }

  // Add node to the model
  ModelError_t err = qnnModel->addNode(QNN_OPCONFIG_VERSION_1, nodeName, packageName, nodeType, tensorParams, scalarParams,
                                       inputTensorNames, outputTensorNames);

  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add node {} of type {} to graph {}: error code {}\n", nodeName, nodeType, graphName,
               static_cast<int>(err));
  }
}

bool QNNBackend::graphFinalize(const std::string& graphName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for finalization", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_INFO("Graph {} is loaded from cache, skipping finalization", graphName);
    return true;
  }

  // Graph finalize
  if (MODEL_NO_ERROR != qnnModel->finalizeGraph(runtime_->profileHandle, nullptr)) {
    MLLM_ERROR("Failed to finalize graph: {}", graphName);
    return false;
  }

  qnnModel->freeCachedTensors();

  // Extract profiling info if enabled
  if (ProfilingLevel::OFF != profilingLevel_ && profilingFinalize_) {
    extractBackendProfilingInfo(runtime_->profileHandle, graphName, "finalize", 0, 0, 0);
  }

  return true;
}

void QNNBackend::graphExecute(const std::string& graphName, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for execution", graphName);
    return;
  }
  auto model = qnnModels_[it->second];

  // Validate input size matches expected input count
  if (inputs.size() != model->getGraphInputTensorWrappers().size()) {
    MLLM_ERROR("Input size mismatch: expected {}, got {} for graph '{}'", model->getGraphInputTensorWrappers().size(),
               inputs.size(), graphName);
    return;
  }

  std::vector<Qnn_Tensor_t> qnn_inputs;
  std::vector<Qnn_Tensor_t> qnn_outputs;
  // Prepare QNN inputs
  for (int i = 0; i < model->getGraphInputTensorWrappers().size(); i++) {
    auto wrapper = model->getGraphInputTensorWrappers()[i];
    auto& wrapper_tensor = wrapper->getDataContainer();
    const auto& runtime_input = inputs[i];

    // Validate input tensors
    if (runtime_input.isNil()) {
      MLLM_ERROR("Input tensor {} is nil for graph '{}'", i, graphName);
      return;
    }

    // Case of executing retrieved graph created by AOT
    // input wrapper is empty, set wrapper's dataContainer(mllm::Tensor)
    if (!wrapper->isAlloc()) { wrapper->__setDataContainer(runtime_input); }

    // Allocate and register the wrapper tensor with QNN allocator
    // QNNAllocator will handle registered memory descriptor when needed
    wrapper->alloc();
    qnn_inputs.push_back(*(wrapper->getNativeTensor()));
  }
  // Prepare QNN outputs
  for (int j = 0; j < model->getGraphOutputTensorWrappers().size(); j++) {
    auto wrapper = model->getGraphOutputTensorWrappers()[j];
    auto& wrapper_tensor = wrapper->getDataContainer();
    const auto& runtime_output = outputs[j];

    // Validate output tensors
    if (runtime_output.isNil()) {
      MLLM_ERROR("Output tensor {} is nil for graph '{}'", j, graphName);
      return;
    }

    // output wrapper is empty, set wrapper's dataContainer(mllm::Tensor)
    if (!wrapper->isAlloc()) { wrapper->__setDataContainer(runtime_output); }

    // alloc and register qnn tensor
    wrapper->alloc();  // QNNAllocator will handle registered memory descriptor
    qnn_outputs.push_back(*(wrapper->getNativeTensor()));
  }
  
//=========================================================================================================
  uint64_t executionIndex = 0;
  bool profileEnabled = false;
  const bool captureProfile = shouldCaptureProfile(graphName, executionIndex, profileEnabled);
  const uint64_t startTimeUs = monotonicTimeUs();
  CALL_QNN(runtime_->qnnInterface.graphExecute(model->getQnnGraph(), qnn_inputs.data(), qnn_inputs.size(), qnn_outputs.data(),
                                               qnn_outputs.size(), profileEnabled ? runtime_->profileHandle : nullptr, nullptr));
  const uint64_t stopTimeUs = monotonicTimeUs();
  const uint64_t duration = stopTimeUs - startTimeUs;
  
  {
    std::ofstream macroFile(profilingMacroPath_, std::ios::app);
    if (macroFile.is_open()) {
      macroFile << graphName << ',' << executionIndex << ',' << profileEnabled << ',' << captureProfile << ',' << duration
                << '\n';
    }
  }
//=========================================================================================================  
  if (captureProfile) {
    extractBackendProfilingInfo(runtime_->profileHandle, graphName, "execute", executionIndex, startTimeUs, stopTimeUs);
  }
}

bool QNNBackend::addTensor(const std::string& graphName, const std::string& tensorName, Qnn_TensorType_t type,
                           const Tensor& tensor, Qnn_QuantizeParams_t quantize) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addTensor(tensorName, type, tensor, quantize);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

bool QNNBackend::addStaticTensor(const std::string& graphName, const std::string& tensorName, const Tensor& tensor,
                                 Qnn_QuantizeParams_t quantize) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding static tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add static tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addStaticTensor(tensorName, tensor, quantize);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add static tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

std::shared_ptr<QNNTensorWrapper> QNNBackend::getTensorWrapper(const std::string& graphName, const std::string& tensorName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for getting tensor wrapper", graphName);
    return nullptr;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  return qnnModel->getTensorWrapper(tensorName);
}

bool QNNBackend::shouldCaptureProfile(const std::string& graphName, uint64_t& executionIndex, bool& profileEnabled) {
  auto& state = profilingCaptureStates_[graphName];
  executionIndex = ++state.executions;
  profileEnabled = false;
  if (ProfilingLevel::OFF == profilingLevel_) { return false; }
  if (!profilingGraphFilter_.empty() && graphName != profilingGraphFilter_) { return false; }

  if (ProfilingLevel::OPTRACE == profilingLevel_) {
    // The HTP Optrace payload is updated inside a retained root event rather than
    // appended as a new root. Keep one execution per process so the serialized log
    // always has exactly one payload for exactly one schematic. HTP also requires
    // the handle to be attached on the graph's first execution, so Optrace does
    // not support an unprofiled warmup in the same process.
    if (profilingGraphFilter_.empty()) {
      for (const auto& [capturedGraph, capturedState] : profilingCaptureStates_) {
        if (capturedGraph != graphName && capturedState.captures != 0) { return false; }
      }
    }
    if (state.captures != 0) { return false; }
    profileEnabled = true;
    ++state.captures;
    return true;
  }

  // HTP linting cannot be attached to a graph after that graph has already run without
  // a profile handle. Keep profiling continuously enabled from the first execution
  // through the final requested capture. Warmup and interval executions are profiled
  // but intentionally not serialized.
  const bool captureLimitReached = profilingMaxCaptures_ != 0 && state.captures >= profilingMaxCaptures_;
  if (captureLimitReached) { return false; }
  profileEnabled = true;

  if (executionIndex <= profilingWarmup_) { return false; }
  if ((executionIndex - profilingWarmup_ - 1) % profilingEvery_ != 0) { return false; }
  ++state.captures;
  return true;
}

bool QNNBackend::initializeProfilingSerialization() {
  if (!profilingSerializationEnabled_) { return false; }
  const auto& system = runtime_->qnnSystemInterface;
  if (system.systemProfileCreateSerializationTarget == nullptr ||
      system.systemProfileSerializeEventData == nullptr ||
      system.systemProfileFreeSerializationTarget == nullptr) {
    MLLM_WARN("QNN System profile serialization is unavailable; keeping text profiling only.");
    return false;
  }

  std::ofstream(profilingSerializedPath_, std::ios::binary | std::ios::trunc).close();
  QnnSystemProfile_SerializationFileHeader_t header{"mllm", "1", backendBuildId_.c_str()};
  QnnSystemProfile_SerializationTargetFile_t file{"qnn-profiling-data.log", profilingDirectory_.c_str()};
  QnnSystemProfile_SerializationTarget_t target{};
  target.type = QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE;
  target.file = file;
  QnnSystemProfile_SerializationTargetConfig_t config{};
  config.type = QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER;
  config.serializationHeader = header;
  const auto status = system.systemProfileCreateSerializationTarget(target, &config, 1,
                                                                     &profilingSerializationHandle_);
  if (status != QNN_SYSTEM_PROFILE_NO_ERROR) {
    MLLM_WARN("Failed to create QNN profile serialization target: {}", static_cast<int>(status));
    profilingSerializationHandle_ = nullptr;
    return false;
  }
  return true;
}

void QNNBackend::extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle, const std::string& graphName,
                                             const char* phase, uint64_t invocation, uint64_t startTimeUs,
                                             uint64_t stopTimeUs) {
  if (profileHandle == nullptr) { return; }

  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    return;
  }
  std::ofstream detailFile(profilingDetailPath_, std::ios::app);
  if (!detailFile.is_open()) return;

  detailFile << "BEGIN_PROFILE|phase=" << phase << "|graph=" << graphName << "|invocation=" << invocation
             << "|host_start_us=" << startTimeUs << "|host_stop_us=" << stopTimeUs
             << "|host_duration_us=" << (stopTimeUs >= startTimeUs ? stopTimeUs - startTimeUs : 0)
             << "|root_events=" << numEvents << '\n';

  uint64_t eventCount = 0;
  std::function<void(QnnProfile_EventId_t, uint32_t)> dumpEvent;
  dumpEvent = [&](QnnProfile_EventId_t eventId, uint32_t depth) {
    if (depth > 64) {
      detailFile << "EVENT_ERROR|depth=" << depth << "|reason=max_depth\n";
      return;
    }
    QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
    if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetEventData(eventId, &eventData)) { return; }

    uint64_t timestampUs = 0;
    if (profilingExtendedEventsSupported_) {
      QnnProfile_ExtendedEventData_t extended = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
      if (QNN_PROFILE_NO_ERROR == runtime_->qnnInterface.profileGetExtendedEventData(eventId, &extended) &&
          extended.version == QNN_PROFILE_DATA_VERSION_1) {
        timestampUs = extended.v1.timestamp;
      }
    }

    ++eventCount;
    detailFile << "EVENT|depth=" << depth << "|type=" << profileTypeName(eventData.type)
               << "|type_id=" << eventData.type << "|unit="
               << (eventData.type == QNN_HTP_PROFILE_EVENTTYPE_NODE_RESOURCEMASK ? "mask"
                                                                                 : profileUnitName(eventData.unit))
               << "|unit_id=" << eventData.unit << "|value=" << eventData.value
               << (eventData.type == QNN_HTP_PROFILE_EVENTTYPE_NODE_RESOURCEMASK
                       ? "|resources=" + htpResourceMaskName(eventData.value)
                       : "")
               << "|timestamp_us=" << timestampUs << "|identifier="
               << (eventData.identifier ? eventData.identifier : "") << '\n';

    const QnnProfile_EventId_t* children = nullptr;
    uint32_t numChildren = 0;
    if (QNN_PROFILE_NO_ERROR == runtime_->qnnInterface.profileGetSubEvents(eventId, &children, &numChildren)) {
      for (uint32_t i = 0; i < numChildren; ++i) { dumpEvent(children[i], depth + 1); }
    }
  };

  for (uint32_t i = 0; i < numEvents; ++i) {
    dumpEvent(profileEvents[i], 0);
  }
  detailFile << "END_PROFILE|phase=" << phase << "|graph=" << graphName << "|invocation=" << invocation
             << "|events=" << eventCount << "\n\n";

  if (profilingSerializationHandle_ == nullptr) { return; }

  std::list<std::vector<QnnSystemProfile_ProfileEventV1_t>> childStorage;
  std::function<bool(QnnProfile_EventId_t, QnnSystemProfile_ProfileEventV1_t&)> buildEvent;
  buildEvent = [&](QnnProfile_EventId_t eventId, QnnSystemProfile_ProfileEventV1_t& output) {
    output = QNN_SYSTEM_PROFILE_EVENT_V1_INIT;
    QnnProfile_EventData_t data = QNN_PROFILE_EVENT_DATA_INIT;
    if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetEventData(eventId, &data)) { return false; }
    bool usedExtendedData = false;
    if (profilingExtendedEventsSupported_ &&
        (data.unit == QNN_PROFILE_EVENTUNIT_OBJECT || data.type == QNN_PROFILE_EVENTTYPE_TRACE)) {
      QnnProfile_ExtendedEventData_t extended = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
      if (QNN_PROFILE_NO_ERROR == runtime_->qnnInterface.profileGetExtendedEventData(eventId, &extended)) {
        output.type = QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA;
        output.extendedEventData = extended;
        usedExtendedData = true;
      }
    }
    if (!usedExtendedData) {
      output.type = QNN_SYSTEM_PROFILE_EVENT_DATA;
      output.eventData = data;
    }

    const QnnProfile_EventId_t* children = nullptr;
    uint32_t numChildren = 0;
    if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetSubEvents(eventId, &children, &numChildren) ||
        numChildren == 0) {
      return true;
    }
    std::vector<QnnSystemProfile_ProfileEventV1_t> childEvents;
    childEvents.reserve(numChildren);
    for (uint32_t i = 0; i < numChildren; ++i) {
      QnnSystemProfile_ProfileEventV1_t child = QNN_SYSTEM_PROFILE_EVENT_V1_INIT;
      if (buildEvent(children[i], child)) { childEvents.push_back(child); }
    }
    childStorage.push_back(std::move(childEvents));
    output.profileSubEventData = childStorage.back().data();
    output.numSubEvents = childStorage.back().size();
    return true;
  };

  std::vector<QnnSystemProfile_ProfileEventV1_t> serializedEvents;
  serializedEvents.reserve(numEvents);
  for (uint32_t i = 0; i < numEvents; ++i) {
    QnnSystemProfile_ProfileEventV1_t event = QNN_SYSTEM_PROFILE_EVENT_V1_INIT;
    if (buildEvent(profileEvents[i], event)) { serializedEvents.push_back(event); }
  }

  QnnSystemProfile_ProfileData_t profileData = QNN_SYSTEM_PROFILE_DATA_INIT;
  profileData.version = QNN_SYSTEM_PROFILE_DATA_VERSION_1;
  profileData.v1.header.methodType = std::strcmp(phase, "execute") == 0
                                         ? QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE
                                         : QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE;
  profileData.v1.header.startTime = startTimeUs;
  profileData.v1.header.stopTime = stopTimeUs;
  profileData.v1.header.graphName = graphName.c_str();
  profileData.v1.profilingEvents = serializedEvents.data();
  profileData.v1.numProfilingEvents = serializedEvents.size();
  const QnnSystemProfile_ProfileData_t* dataPtr = &profileData;
  if (QNN_SUCCESS != runtime_->qnnSystemInterface.systemProfileSerializeEventData(
                         profilingSerializationHandle_, &dataPtr, 1)) {
    MLLM_WARN("Failed to serialize QNN profiling data for graph {} invocation {}", graphName, invocation);
  }
}

}  // namespace mllm::qnn
