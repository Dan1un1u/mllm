#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
artifact_root="${LPBQ_MLP_ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_lpbq_mlp/20260818_micrograph_layer14}"
source_model="${LPBQ_MLP_CONV_MODEL:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm}"
fc_model="${LPBQ_MLP_FC_MODEL:-${artifact_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8-layer14-fc.mllm}"
matmul_model="${LPBQ_MLP_MATMUL_MODEL:-${artifact_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8-layer14-matmul.mllm}"
compiler="${LPBQ_MLP_COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-lpbq-mlp-micrograph-c}"
compiler_bin="$(dirname "${compiler}")"
aot_config="${LPBQ_MLP_AOT_CONFIG:-${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

read -r -a layouts <<< "${LPBQ_MLP_LAYOUTS:-conv matmul}"
read -r -a projections <<< "${LPBQ_MLP_PROJECTIONS:-gate_proj up_proj down_proj}"
read -r -a sequences <<< "${LPBQ_MLP_SEQUENCES:-1 32}"

[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
for library in libMllmRT.so libMllmCPUBackend.so libMllmQNNBackend.so; do
  [[ -f "${compiler_bin}/${library}" ]] || {
    echo "compiler-local runtime missing: ${compiler_bin}/${library}" >&2
    exit 2
  }
done
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QAIRT HTP library missing: ${qnn_lib}" >&2; exit 2; }

export LD_LIBRARY_PATH="${compiler_bin}:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1

for layout in "${layouts[@]}"; do
  case "${layout}" in
    conv) model="${source_model}" ;;
    fc) model="${fc_model}" ;;
    matmul) model="${matmul_model}" ;;
    *) echo "unsupported layout: ${layout}" >&2; exit 2 ;;
  esac
  [[ -f "${model}" ]] || { echo "model missing for ${layout}: ${model}" >&2; exit 2; }

  for projection in "${projections[@]}"; do
    for seq in "${sequences[@]}"; do
      case_name="${layout}_${projection}_s${seq}"
      case_dir="${artifact_root}/contexts/${case_name}"
      context="${case_dir}/${case_name}.bin"
      manifest_dir="${case_dir}/manifests"
      schematic_dir="${case_dir}/schematics"
      manifest="${manifest_dir}/model.0.s${seq}_quant_manifest.json"
      schematic="${schematic_dir}/model.0.s${seq}_schematic.bin"
      if [[ -s "${context}" && -s "${manifest}" && -s "${schematic}" ]]; then
        echo "SKIP complete ${case_name}"
        continue
      fi

      mkdir -p "${manifest_dir}" "${schematic_dir}"
      export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${manifest_dir}"
      export MLLM_QNN_AOT_OPTRACE_DIR="${schematic_dir}"
      echo "BUILD ${case_name}"
      (
        cd "${case_dir}"
        stdbuf -oL -eL "${compiler}" \
          -m "${model}" \
          -aot_cfg "${aot_config}" \
          -qnn_env "${qnn_lib}/" \
          -o "${context}" \
          --layout "${layout}" \
          --projection "${projection}" \
          --seq "${seq}"
      ) > "${case_dir}/compile.log" 2>&1
      [[ -s "${context}" ]] || { echo "context missing for ${case_name}" >&2; exit 1; }
      [[ -s "${manifest}" ]] || { echo "manifest missing for ${case_name}" >&2; exit 1; }
      [[ -s "${schematic}" ]] || { echo "schematic missing for ${case_name}" >&2; exit 1; }
      echo "PASS finalize ${case_name}"
    done
  done
done

echo "LPBQ MLP micrograph contexts complete: ${artifact_root}/contexts"
