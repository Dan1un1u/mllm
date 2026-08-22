# AOT targets run on x86
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-qwen3-aot-c compile.cpp)
  target_link_libraries(mllm-qwen3-aot-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

  # G32 variant.  The regular Qwen3 target intentionally remains G16 for
  # backwards compatibility; the compile definition selects the physical
  # LPBQ Conv2D op in modeling_qwen_qnn_aot.hpp.
  add_executable(mllm-qwen3-aot-g32-c compile.cpp)
  target_compile_definitions(mllm-qwen3-aot-g32-c PRIVATE MLLM_QWEN3_QNN_AOT_G32=1)
  target_link_libraries(mllm-qwen3-aot-g32-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-g32-c PRIVATE ${MLLM_INCLUDE_DIR})

  # SHA (Single Head Attention) version - MHA to SHA optimization
  add_executable(mllm-qwen3-aot-sha-c compile_sha.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-sha-g32-c compile_sha.cpp)
  target_compile_definitions(mllm-qwen3-aot-sha-g32-c PRIVATE MLLM_QWEN3_QNN_AOT_G32=1)
  target_link_libraries(mllm-qwen3-aot-sha-g32-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-g32-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-native-u8-masked-softmax-c compile_native_u8_masked_softmax.cpp)
  target_compile_definitions(mllm-qwen3-native-u8-masked-softmax-c PRIVATE MLLM_QWEN3_QNN_AOT_G32=1)
  target_link_libraries(mllm-qwen3-native-u8-masked-softmax-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-native-u8-masked-softmax-c PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-qwen3-aot-runner aot_run.cpp)
target_link_libraries(mllm-qwen3-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-native-u8-masked-softmax-runner run_native_u8_masked_softmax.cpp)
target_link_libraries(mllm-qwen3-native-u8-masked-softmax-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-native-u8-masked-softmax-runner PRIVATE ${MLLM_INCLUDE_DIR})
