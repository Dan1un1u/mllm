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

  # Explicit W4A16G32 control.  It retains UInt16 activations while selecting
  # the same G32 LPBQ weight geometry used by the A8 experiment.
  add_executable(mllm-qwen3-aot-sha-a16-g32-c compile_sha.cpp)
  target_compile_definitions(mllm-qwen3-aot-sha-a16-g32-c PRIVATE MLLM_QWEN3_QNN_AOT_A16_G32=1)
  target_link_libraries(mllm-qwen3-aot-sha-a16-g32-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-a16-g32-c PRIVATE ${MLLM_INCLUDE_DIR})

  # Strict LPBQ projection diagnostic: identical W4G32 payload/layout, with
  # only the asymmetric activation contract switched between A16 and A8.
  add_executable(mllm-qwen3-lpbq-a16-a8-projection-c compile_lpbq_a16_a8_projection.cpp)
  target_link_libraries(mllm-qwen3-lpbq-a16-a8-projection-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-lpbq-a16-a8-projection-c PRIVATE ${MLLM_INCLUDE_DIR})

  # Real layer-14 single-head K/V W4G32/A8 projection used to isolate
  # finalize scheduling for the 2048 -> 128 attention microkernels.
  add_executable(mllm-qwen3-kv-head-projection-c compile_kv_head_projection.cpp)
  target_link_libraries(mllm-qwen3-kv-head-projection-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-kv-head-projection-c PRIVATE ${MLLM_INCLUDE_DIR})

  # Fixed-P19 K/V projection packing gate.  The split control executes eight
  # real 2048 -> 128 LPBQ projections; the candidate executes one
  # 2048 -> 1024 LPBQ projection and exposes the same eight head tensors.
  add_executable(mllm-qwen3-kv-head-packing-c compile_kv_head_packing.cpp)
  target_link_libraries(mllm-qwen3-kv-head-packing-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-kv-head-packing-c PRIVATE ${MLLM_INCLUDE_DIR})

  # First-chunk prefill diagnostic.  It compares the accepted 1024-column
  # attention core with a 32-column compact form while retaining native QNN
  # MatMul/Softmax and the real layer-14 A8 qparams.
  add_executable(mllm-qwen3-compact-prefill-attention-c compile_compact_prefill_attention.cpp)
  target_compile_definitions(mllm-qwen3-compact-prefill-attention-c PRIVATE MLLM_QWEN3_QNN_AOT_G32=1)
  target_link_libraries(mllm-qwen3-compact-prefill-attention-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-compact-prefill-attention-c PRIVATE ${MLLM_INCLUDE_DIR})

  # End-to-end first-prefill validation after the attention-core gate.  The
  # compiler emits either the 1024-column full control or the cache-free
  # 32-column compact graph from the same accepted RMSNorm-A8 model.
  add_executable(mllm-qwen3-compact-initial-prefill-c compile_compact_initial_prefill.cpp)
  target_compile_definitions(mllm-qwen3-compact-initial-prefill-c PRIVATE MLLM_QWEN3_QNN_AOT_G32=1)
  target_link_libraries(mllm-qwen3-compact-initial-prefill-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-compact-initial-prefill-c PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-qwen3-aot-runner aot_run.cpp)
target_link_libraries(mllm-qwen3-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-lpbq-a16-a8-projection-runner run_lpbq_a16_a8_projection.cpp)
target_link_libraries(mllm-qwen3-lpbq-a16-a8-projection-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-lpbq-a16-a8-projection-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-kv-head-packing-runner run_kv_head_packing.cpp)
target_link_libraries(mllm-qwen3-kv-head-packing-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-kv-head-packing-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-compact-prefill-attention-runner run_compact_prefill_attention.cpp)
target_link_libraries(mllm-qwen3-compact-prefill-attention-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-compact-prefill-attention-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-compact-initial-prefill-runner run_compact_initial_prefill.cpp)
target_link_libraries(mllm-qwen3-compact-initial-prefill-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-compact-initial-prefill-runner PRIVATE ${MLLM_INCLUDE_DIR})
