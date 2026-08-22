#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/native_u8_masked_softmax_qairt249/20260822}"
source_root="${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529"
source_model="${SOURCE_MODEL:-${source_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm}"
compiler_247="${COMPILER_247:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-native-u8-masked-softmax-c}"
compiler_249="${COMPILER_249:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-native-u8-masked-softmax-c}"
config="${AOT_CONFIG:-${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json}"
python="${PYTHON:-python3}"

read -r -a versions <<< "${SOFTMAX_QAIRT_VERSIONS:-247 249}"
read -r -a sequences <<< "${SOFTMAX_SEQUENCES:-1 32}"
read -r -a p_points <<< "${SOFTMAX_P_POINTS:-default 0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23}"

sdk_root() {
  case "$1" in
    247) printf '%s\n' "${models_root}/qualcomm-sdk/qairt/2.47.0.260601" ;;
    249) printf '%s\n' "${models_root}/qualcomm-sdk/qairt/2.49.0.260730" ;;
    *) echo "unsupported QAIRT key: $1" >&2; return 2 ;;
  esac
}

compiler_path() {
  case "$1" in
    247) printf '%s\n' "${compiler_247}" ;;
    249) printf '%s\n' "${compiler_249}" ;;
    *) echo "unsupported QAIRT key: $1" >&2; return 2 ;;
  esac
}

[[ -x "${compiler_247}" ]] || { echo "compiler missing: ${compiler_247}" >&2; exit 2; }
[[ -x "${compiler_249}" ]] || { echo "compiler missing: ${compiler_249}" >&2; exit 2; }
[[ -s "${source_model}" ]] || { echo "source model missing: ${source_model}" >&2; exit 2; }
mkdir -p "${artifact_root}" "${repo_root}/tmp"

if [[ ! -s "${artifact_root}/fixture_report.json" ]]; then
  stage="$(mktemp -d "${repo_root}/tmp/native_u8_softmax_fixture.XXXXXX")"
  trap 'rm -rf "${stage}"' ERR INT TERM
  "${python}" "${repo_root}/scripts/qnn_native_u8_masked_softmax_fixture.py" \
    --manifest-s1 "${source_root}/manifests/model.0.s1_quant_manifest.json" \
    --manifest-s32 "${source_root}/manifests/model.0.s32_quant_manifest.json" \
    --source-model "${source_model}" \
    --output-dir "${stage}" \
    --report "${stage}/fixture_report.json" \
    >"${stage}/fixture_generation.log"
  cp -f "${stage}"/*.raw "${stage}/fixture_report.json" \
    "${stage}/fixture_generation.log" "${artifact_root}/"
  rm -rf "${stage}"
  trap - ERR INT TERM
fi

mkdir -p "${artifact_root}/contexts" "${artifact_root}/rejected"
index_tmp="${artifact_root}/contexts/index.tsv.tmp.$$"
printf 'sdk\tseq\tp\ttag\tcontext\tmanifest\tschematic\n' >"${index_tmp}"

for version in "${versions[@]}"; do
  sdk="$(sdk_root "${version}")"
  compiler="$(compiler_path "${version}")"
  compiler_bin="$(dirname "${compiler}")"
  qnn_lib="${sdk}/lib/x86_64-linux-clang"
  [[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "missing SDK library: ${qnn_lib}" >&2; exit 2; }
  for seq in "${sequences[@]}"; do
    for p in "${p_points[@]}"; do
      tag="qairt${version}_s${seq}_p${p}"
      case_dir="${artifact_root}/contexts/${tag}"
      context="${case_dir}/${tag}.bin"
      # Standalone traces use the fixed internal graph symbol model.0.s1;
      # the case tag and manifest tensor dimensions carry the real sequence.
      manifest="${case_dir}/manifests/model.0.s1_quant_manifest.json"
      schematic="${case_dir}/schematics/model.0.s1_schematic.bin"
      if [[ -s "${context}" && -s "${manifest}" && -s "${schematic}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${version}" "${seq}" "${p}" "${tag}" "${context}" "${manifest}" "${schematic}" \
          >>"${index_tmp}"
        continue
      fi
      [[ ! -e "${case_dir}" ]] || { echo "refusing partial case: ${case_dir}" >&2; exit 2; }
      stage="$(mktemp -d "${repo_root}/tmp/native_u8_softmax_compile.XXXXXX")"
      mkdir -p "${stage}/manifests" "${stage}/schematics"
      env_args=(
        "LD_LIBRARY_PATH=${compiler_bin}:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        "MLLM_QNN_AOT_OPTRACE=1"
        "MLLM_QNN_AOT_QUANT_MANIFEST_DIR=${stage}/manifests"
        "MLLM_QNN_AOT_OPTRACE_DIR=${stage}/schematics"
      )
      if [[ "${p}" != default ]]; then env_args+=("MLLM_QNN_AOT_FINALIZE_P=${p}"); fi
      set +e
      (
        cd "${stage}"
        env "${env_args[@]}" "${compiler}" \
          -m "${source_model}" \
          -aot_cfg "${config}" \
          -qnn_env "${qnn_lib}/" \
          -o "${stage}/${tag}.bin" \
          --seq "${seq}"
      ) >"${stage}/compile.log" 2>&1
      status=$?
      set -e
      if (( status != 0 )) || [[ ! -s "${stage}/${tag}.bin" ]]; then
        rejected="${artifact_root}/rejected/${tag}"
        mkdir -p "${rejected}"
        cp -f "${stage}/compile.log" "${rejected}/"
        printf '%s\n' "${status}" >"${rejected}/exit_code.txt"
        rm -rf "${stage}"
        if [[ "${p}" == default ]]; then
          echo "default compile failed: ${tag}" >&2
          exit 1
        fi
        echo "REJECT ${tag}"
        continue
      fi
      if [[ "${p}" != default ]]; then
        grep -q "P = ${p}" "${stage}/compile.log" || {
          echo "forced P evidence missing: ${tag}" >&2
          exit 1
        }
      fi
      for output in \
        "${stage}/manifests/model.0.s1_quant_manifest.json" \
        "${stage}/schematics/model.0.s1_schematic.bin"; do
        [[ -s "${output}" ]] || { echo "missing compiler output: ${output}" >&2; exit 1; }
      done
      sha256sum "${stage}/${tag}.bin" \
        "${stage}/manifests/model.0.s1_quant_manifest.json" \
        "${stage}/schematics/model.0.s1_schematic.bin" \
        >"${stage}/artifacts.sha256"
      publish="${case_dir}.tmp.$$"
      mkdir -p "${publish}"
      cp -a "${stage}/." "${publish}/"
      mv "${publish}" "${case_dir}"
      rm -rf "${stage}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${version}" "${seq}" "${p}" "${tag}" "${context}" "${manifest}" "${schematic}" \
        >>"${index_tmp}"
      echo "PASS ${tag}"
    done
  done
done

mv "${index_tmp}" "${artifact_root}/contexts/index.tsv"
echo "Native-U8 masked-softmax contexts complete: ${artifact_root}/contexts"
