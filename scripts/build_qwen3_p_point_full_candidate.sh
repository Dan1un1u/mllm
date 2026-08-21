#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-c}"
kind="${1:?usage: build_qwen3_p_point_full_candidate.sh a8|a16 P_S1|default P_S32|default TAG}"
p_s1="${2:?missing s1 P point}"
p_s32="${3:?missing s32 P point}"
tag="${4:?missing candidate tag}"
publish_root="${PUBLISH_ROOT:-${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/${tag}}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_full_20260822/${tag}}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

case "${kind}" in
  a8)
    model="${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
    baseline_manifest_dir="${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/manifests"
    ;;
  a16)
    model="${models_root}/qwen3_sm8750_v79/g32/w4a16/source_g32_export/qwen3_1.7b_g32.mllm"
    baseline_manifest_dir="${models_root}/qwen3_sm8750_v79/g32/w4a16/schematics"
    compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-a16-c}"
    ;;
  *) echo "unsupported kind: ${kind}" >&2; exit 2 ;;
esac
compiler_bin="$(dirname "${compiler}")"
valid_point() {
  [[ "$1" == default || "$1" =~ ^(0|1|2|3|4|5|6|8|13|15|16|17|19|20|21|22|23)$ ]]
}
valid_point "${p_s1}" || { echo "invalid s1 P point: ${p_s1}" >&2; exit 2; }
valid_point "${p_s32}" || { echo "invalid s32 P point: ${p_s32}" >&2; exit 2; }
[[ "${tag}" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe tag: ${tag}" >&2; exit 2; }
case "${publish_root}" in /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/*) ;;
  *) echo "refusing unexpected PUBLISH_ROOT: ${publish_root}" >&2; exit 2 ;;
esac
case "${work_root}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_full_20260822/*) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == 2.47.0.260601 ]] || exit 2
[[ -x "${compiler}" && -s "${model}" && -f "${qnn_lib}/libQnnHtp.so" ]] || exit 2
[[ ! -e "${publish_root}" ]] || { echo "candidate already published: ${publish_root}" >&2; exit 2; }
[[ ! -e "${work_root}" ]] || { echo "candidate work root exists: ${work_root}" >&2; exit 2; }

mkdir -p "${work_root}/manifests"
# An archived A16 compiler may come from an isolated worktree.  Its matching
# Mllm shared libraries must win dynamic resolution; mixing the executable
# with the current A8 backend silently changes the quantization contract.
export LD_LIBRARY_PATH="${compiler_bin}:${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${work_root}/manifests"
unset MLLM_QNN_AOT_OPTRACE MLLM_QNN_AOT_OPTRACE_DIR MLLM_QNN_AOT_FINALIZE_P
if [[ "${p_s1}" != default ]]; then export MLLM_QNN_AOT_FINALIZE_P_S1="${p_s1}"; else unset MLLM_QNN_AOT_FINALIZE_P_S1; fi
if [[ "${p_s32}" != default ]]; then export MLLM_QNN_AOT_FINALIZE_P_S32="${p_s32}"; else unset MLLM_QNN_AOT_FINALIZE_P_S32; fi

if [[ "${kind}" == a16 ]]; then
  model_config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32_a16.json"
  aot_config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32_a16.json"
else
  model_config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
  aot_config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json"
fi

(
  cd "${work_root}"
  "${compiler}" \
    -m "${model}" \
    -c "${model_config}" \
    -aot_cfg "${aot_config}" \
    -qnn_env "${qnn_lib}/" \
    -o context.bin
) >"${work_root}/compile.log" 2>&1

for graph in s1 s32; do
  [[ -s "${work_root}/manifests/model.0.${graph}_quant_manifest.json" ]] || exit 1
  python3 "${repo_root}/scripts/qnn_quant_manifest_equivalent.py" \
    "${baseline_manifest_dir}/model.0.${graph}_quant_manifest.json" \
    "${work_root}/manifests/model.0.${graph}_quant_manifest.json" \
    >"${work_root}/manifests/model.0.${graph}_canonical.sha256"
done
[[ -s "${work_root}/context.bin" ]] || exit 1
for graph in s1 s32; do
  if [[ "${graph}" == s1 ]]; then point="${p_s1}"; else point="${p_s32}"; fi
  if [[ "${point}" == default ]]; then
    if grep -q "Graph model.0.${graph} with init graph option: P =" "${work_root}/compile.log"; then
      echo "unexpected explicit P on ${graph}" >&2
      exit 1
    fi
  else
    grep -q "Graph model.0.${graph} with init graph option: P = ${point}" "${work_root}/compile.log" || {
      echo "compile log does not prove ${graph}=P${point}" >&2
      exit 1
    }
  fi
done

publish="${publish_root}.tmp.$$"
case "${publish}" in /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/*.tmp.*) ;;
  *) exit 2 ;;
esac
mkdir -p "${publish}/manifests"
trap 'rm -rf "${publish}"' ERR INT TERM
cp "${work_root}/context.bin" "${publish}/context.bin"
cp "${work_root}/manifests/"*.json "${work_root}/manifests/"*.sha256 "${publish}/manifests/"
cp "${work_root}/compile.log" "${publish}/compile.log"
{
  printf 'kind=%s\nfinalize_O=3\nfinalize_P_s1=%s\nfinalize_P_s32=%s\n' \
    "${kind}" "${p_s1}" "${p_s32}"
  printf 'source_model=%s\nsource_model_sha256=%s\n' "${model}" "$(sha256sum "${model}" | awk '{print $1}')"
  printf 'git_commit=%s\ngit_branch=%s\nqairt_release=2.47.0.260601\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
  printf 'logical_quant_manifests=canonical-equivalent to archived %s baseline\n' "${kind}"
} >"${publish}/provenance.env"
(
  cd "${publish}"
  sha256sum context.bin manifests/*.json manifests/*.sha256 provenance.env >artifacts.sha256
)
mv "${publish}" "${publish_root}"
resolved_work="$(realpath -e "${work_root}")"
case "${resolved_work}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_full_20260822/*) ;;
  *) exit 2 ;;
esac
rm -rf "${resolved_work}"
trap - ERR INT TERM
echo "full P-point candidate published: ${publish_root}"
