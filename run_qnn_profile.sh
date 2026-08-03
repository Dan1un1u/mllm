#!/usr/bin/env bash

set -euo pipefail

PROMPT="请深入分析并撰写一篇关于人工智能基础设施演进的技术综述。内容需要涵盖从经典的冯·诺依曼架构到现代端侧 NPU/TPU 异构计算的转变。特别需要详细阐述 Transformer 架构中 KV Cache 机制对于推理延迟的影响。"
PROFILE_LEVEL="${MLLM_QNN_PROFILE_LEVEL:-linting}"
PROFILE_WARMUP="${MLLM_QNN_PROFILE_WARMUP:-0}"
PROFILE_EVERY="${MLLM_QNN_PROFILE_EVERY:-1}"
PROFILE_MAX_CAPTURES="${MLLM_QNN_PROFILE_MAX_CAPTURES:-1}"
PROFILE_GRAPH="${MLLM_QNN_PROFILE_GRAPH:-}"
RESULT_DIR="${MLLM_QNN_RESULT_DIR:-/home/daniuniu/llm_exp/results/qnn_profile_$(date +%Y%m%d_%H%M%S)}"
if [[ "${PROFILE_LEVEL}" == "optrace" && -z "${PROFILE_GRAPH}" ]]; then
    PROFILE_GRAPH="model.0.s32"
fi

echo "===== Build ====="
python3 task.py tasks/build_android_qnn.yaml

echo "===== Push ====="
adb push build-android-arm64-v8a-qnn/bin/*.so /data/local/tmp
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner /data/local/tmp

echo "===== Run ====="

adb shell <<EOF
cd /data/local/tmp
rm -f qnn_detail_profile.txt qnn_macro_profile.csv qnn_e2e_profile.csv qnn-profiling-data.log

export LD_LIBRARY_PATH=.
export MLLM_QNN_PROFILE_LEVEL="${PROFILE_LEVEL}"
export MLLM_QNN_PROFILE_WARMUP="${PROFILE_WARMUP}"
export MLLM_QNN_PROFILE_EVERY="${PROFILE_EVERY}"
export MLLM_QNN_PROFILE_MAX_CAPTURES="${PROFILE_MAX_CAPTURES}"
export MLLM_QNN_PROFILE_GRAPH="${PROFILE_GRAPH}"
export MLLM_QNN_PROFILE_SERIALIZE=1

echo "${PROMPT}" | \
./mllm-qwen3-aot-runner \
    -m qwen3-1.7B-lpbq-sha.bin \
    -t qwen3-tokenizer.json \
    -c config_1.7B.json \
    --ar_len 32
EOF

echo "===== Pull Profile ====="
mkdir -p "${RESULT_DIR}"

adb pull \
    /data/local/tmp/qnn_detail_profile.txt \
    "${RESULT_DIR}/"
adb pull \
    /data/local/tmp/qnn_macro_profile.csv \
    "${RESULT_DIR}/"
adb pull \
    /data/local/tmp/qnn_e2e_profile.csv \
    "${RESULT_DIR}/"
adb pull \
    /data/local/tmp/qnn-profiling-data.log \
    "${RESULT_DIR}/"

if [[ -n "${QAIRT_SDK_ROOT:-}" ]]; then
    if [[ "${PROFILE_LEVEL}" == "optrace" ]]; then
        if [[ -z "${MLLM_QNN_OPTRACE_SCHEMATIC:-}" ]]; then
            echo "Optrace captured. Set MLLM_QNN_OPTRACE_SCHEMATIC to generate a Chrome Trace."
        else
            "${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer" \
                --reader "${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so" \
                --input_log "${RESULT_DIR}/qnn-profiling-data.log" \
                --schematic "${MLLM_QNN_OPTRACE_SCHEMATIC}" \
                --output "${RESULT_DIR}/qnn_optrace.json"
            echo "Generated ${RESULT_DIR}/qnn_optrace.json"
            if [[ -f "${RESULT_DIR}/qnn_optrace_htp.json" ]]; then
                python3 scripts/qnn_optrace_summary.py \
                    "${RESULT_DIR}/qnn_optrace.json" \
                    --htp-json "${RESULT_DIR}/qnn_optrace_htp.json" \
                    --output "${RESULT_DIR}/qnn_optrace_operators.csv" \
                    --type-summary-output "${RESULT_DIR}/qnn_optrace_logical_types.csv"
                echo "Generated ${RESULT_DIR}/qnn_optrace_operators.csv"
                QHAS_JSON="${RESULT_DIR}/qnn_optrace_qnn_htp_analysis_summary.json"
                if [[ -f "${QHAS_JSON}" ]]; then
                    python3 scripts/qnn_optrace_qwen3_structure.py \
                        "${RESULT_DIR}/qnn_optrace.json" \
                        --htp-json "${RESULT_DIR}/qnn_optrace_htp.json" \
                        --qhas-json "${QHAS_JSON}" \
                        --output-prefix "${RESULT_DIR}/qnn_optrace"
                    echo "Generated ${RESULT_DIR}/qnn_optrace-qwen3-structure.html"
                    QUANT_MANIFEST_ARGS=()
                    QUANT_MANIFEST="${MLLM_QNN_QUANT_MANIFEST:-}"
                    if [[ -z "${QUANT_MANIFEST}" && -n "${MLLM_QNN_OPTRACE_SCHEMATIC:-}" ]]; then
                        QUANT_MANIFEST="${MLLM_QNN_OPTRACE_SCHEMATIC%_schematic.bin}_quant_manifest.json"
                    fi
                    if [[ -n "${QUANT_MANIFEST}" && -f "${QUANT_MANIFEST}" ]]; then
                        QUANT_MANIFEST_ARGS=(--quant-manifest "${QUANT_MANIFEST}")
                    fi
                    python3 scripts/qnn_optrace_quantization.py \
                        "${RESULT_DIR}/qnn_optrace.json" \
                        --qhas-json "${QHAS_JSON}" \
                        "${QUANT_MANIFEST_ARGS[@]}" \
                        --output-prefix "${RESULT_DIR}/qnn_optrace"
                    echo "Generated ${RESULT_DIR}/qnn_optrace-quantization.html"
                fi
            fi
        fi
    else
        "${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer" \
            --reader "${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so" \
            --input_log "${RESULT_DIR}/qnn-profiling-data.log" \
            --output "${RESULT_DIR}/qnn_profile.csv" \
            >/dev/null
        echo "Generated ${RESULT_DIR}/qnn_profile.csv"
    fi
fi

echo "Finished. Results: ${RESULT_DIR}"
