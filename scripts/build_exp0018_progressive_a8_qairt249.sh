#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce EXP-0018's selected software candidate and compile it with the
# isolated QAIRT 2.49 P19 toolchain.  Small-file staging stays on WSL ext4;
# only final model artifacts and evidence are atomically published to D:.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
[[ "${run_id}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
    echo "RUN_ID must match YYYYMMDD_HHMMSS" >&2
    exit 2
}

models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results}"
source_model="${SOURCE_MODEL:-${models_root}/Qwen3-origin}"
corpus="${CALIBRATION_CORPUS:-${models_root}/qwen3_sm8750_v79/g32/exp0018_deployment_calibration/calibration/qwen3_wikitext_103_v1_train_128x512.jsonl}"
qparams_report="${ACTIVATION_QPARAMS_REPORT:-${results_root}/exp0018_deployment_calibration_20260825/progressive_block_calibration_after_l0_l1_mlp_v5/report.json}"
venv="${EXP0018_VENV:-/home/daniuniu/llm_exp_work/venvs/exp0018}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-aot-sha-g32-c}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/exp0018_progressive_calibration/${run_id}}"
publish_root="${PUBLISH_ROOT:-${models_root}/qwen3_sm8750_v79/g32/exp0018_progressive_calibration/${run_id}}"

case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/exp0018_progressive_calibration/*) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
case "${publish_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/exp0018_progressive_calibration/*) ;;
  *) echo "refusing unexpected PUBLISH_ROOT: ${publish_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || exit 2
[[ -d "${source_model}" && -s "${corpus}" && -s "${qparams_report}" ]] || exit 2
[[ -x "${venv}/bin/python" && -x "${compiler}" ]] || exit 2
[[ -f "${qairt_root}/lib/x86_64-linux-clang/libQnnHtp.so" ]] || exit 2
[[ ! -e "${work_root}" && ! -e "${publish_root}" ]] || exit 2
[[ -z "$(git -C "${repo_root}" status --porcelain=v1)" ]] || {
    echo "formal build requires a clean source worktree" >&2
    exit 2
}

model="${work_root}/qwen3-1.7B-w4a8g32-exp0018.mllm"
context="${work_root}/qnn/qwen3-1.7B-w4a8g32-exp0018-qairt249-p19.bin"
manifest_dir="${work_root}/qnn/manifests"
schematic_dir="${work_root}/qnn/schematics"
evidence_dir="${work_root}/evidence"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
mkdir -p "${manifest_dir}" "${schematic_dir}" "${evidence_dir}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
"${venv}/bin/python" -m pymllm.mobile.backends.qualcomm.transformers.qwen3.train \
    --model_path "${source_model}" \
    --max_length 512 \
    --num_samples 128 \
    --calibration_corpus "${corpus}" \
    --activation_bits 8 \
    --linear_block_size 32 \
    --activation_qparams_report "${qparams_report}" \
    --infer_max_new_tokens 16 \
    --output_dir "${work_root}/intermediate" \
    --output_mllm "${model}" \
    >"${evidence_dir}/model-build.log" 2>&1

python3 "${repo_root}/scripts/verify_mllm_v2.py" "${model}" \
    --output-json "${evidence_dir}/mllm-v2-audit.json"

export LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${manifest_dir}"
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_OPTRACE_DIR="${schematic_dir}"
export MLLM_QNN_AOT_FINALIZE_P=19
(
    cd "${work_root}/qnn"
    "${compiler}" \
        -m "${model}" \
        -c "${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json" \
        -aot_cfg "${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json" \
        -qnn_env "${qnn_lib}/" \
        -o "${context}"
) >"${evidence_dir}/context-build.log" 2>&1

[[ "$(grep -c "init graph option: P = 19" "${evidence_dir}/context-build.log")" -eq 2 ]] || exit 1
for graph in s1 s32; do
    [[ -s "${manifest_dir}/model.0.${graph}_quant_manifest.json" ]] || exit 1
    [[ -s "${schematic_dir}/model.0.${graph}_schematic.bin" ]] || exit 1
done
[[ -s "${model}" && -s "${context}" ]] || exit 1
python3 "${repo_root}/scripts/qnn_w4a8_manifest_audit.py" \
    "${manifest_dir}/model.0.s1_quant_manifest.json" \
    "${manifest_dir}/model.0.s32_quant_manifest.json" \
    --rmsnorm-u8 \
    --output "${evidence_dir}/manifest-audit.json"

{
    printf 'experiment=EXP-0018\n'
    printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
    printf 'qairt_release=2.49.0.260730\nfinalize_P=19\n'
    printf 'qparams_report=%s\n' "${qparams_report}"
    printf 'qparams_sha256=%s\n' "$(sha256sum "${qparams_report}" | awk '{print $1}')"
    printf 'corpus_sha256=%s\n' "$(sha256sum "${corpus}" | awk '{print $1}')"
} >"${evidence_dir}/provenance.env"

publish="${publish_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/exp0018_progressive_calibration/*.tmp.*) ;;
  *) exit 2 ;;
esac
mkdir -p "${publish}/manifests" "${publish}/schematics" "${publish}/evidence"
trap 'rm -rf "${publish}"' ERR INT TERM
cp "${model}" "${publish}/$(basename "${model}")"
cp "${context}" "${publish}/$(basename "${context}")"
cp "${manifest_dir}"/*.json "${publish}/manifests/"
cp "${schematic_dir}"/*.bin "${publish}/schematics/"
cp "${evidence_dir}"/* "${publish}/evidence/"
cp "${qparams_report}" "${publish}/evidence/activation-qparams-report.json"
(
    cd "${publish}"
    sha256sum ./*.mllm ./*.bin manifests/* schematics/* evidence/* >artifacts.sha256
)
mv "${publish}" "${publish_root}"
trap - ERR INT TERM
echo "EXP-0018 QAIRT 2.49 P19 artifacts published: ${publish_root}"
