# Qwen3 native-U8 RMSNorm baseline migration to QAIRT 2.49

Date: 2026-08-23
Target: PJZ110 / SM8750 / HTP V79
Source branch: `codex/w4a8g32-rmsnorm-u8-qairt249`
Artifact source commit: `a8a3dcd26f184f11bad4840650e9666cebbcc43b`
QAIRT: `2.49.0.260730` / QNN API `2.38.0`

## Decision

The native-U8 RMSNorm W4A8G32 baseline is successfully migrated to the
isolated QAIRT 2.49 environment. The migrated baseline uses graph-finalize
`O=3, P=19` for both s1 and s32. The QAIRT-default context is retained only as
an audit/control artifact because it is slower in both workloads.

No model or quantization recipe changed during the migration. The accepted
QAIRT 2.47 `.mllm` source, W4G32 weights, asymmetric-U8 activation encodings,
RMSNorm U8 contract, graph shapes, tokenizer, prompt, runner protocol, and
accuracy suite remain fixed. QAIRT 2.47 was not replaced or modified.

## Isolated build contract

The 2.49 SDK is pinned at:

```text
D:\llm_exp\models\qualcomm-sdk\qairt\2.49.0.260730
```

The source model is the accepted native-U8 RMSNorm model:

```text
D:\llm_exp\models\qwen3_sm8750_v79\g32\w4a8_rmsnorm_u8\20260813_211529\qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm
SHA-256 de6c789a684f3276d16b9673f13f4c69aa9eb4a68ab8b9fca2a76be3fbb75a3e
```

Host compiler, Android runner, and the LLaMA custom-op package were rebuilt
against 2.49 in distinct build/work directories. The result runner reports:

```text
QNN Backend Build Id: v2.49.0.260730134355
```

The build and profiling entry points are:

```bash
scripts/build_qwen3_w4a8g32_rmsnorm_u8_qairt249_context.sh default
scripts/build_qwen3_w4a8g32_rmsnorm_u8_qairt249_context.sh p19

PROFILE_WORK_ROOT=/home/daniuniu/llm_exp_work/profile/<unique-name> \
  ./run_qwen3_sm8750_v79_g32_rmsnorm_u8_qairt249_profile.sh p19
```

Trace viewer small-file work is performed on the WSL ext4 filesystem. Only
the published context binaries and final evidence are archived on the D drive.

## Logical and numerical equivalence

The QAIRT 2.47 reference, QAIRT 2.49 default context, and QAIRT 2.49 P19
context have byte-identical quantization manifests:

| Graph | Manifest SHA-256 |
|---|---|
| s1 | `87bf57c5849180fcd77b09cf06b65aa003862f9603ee03570a71023c8e70dd5c` |
| s32 | `306bf8c07b9f67572611117bf0c7a39e8905ced9eae85f1aa73bde89206bceb8` |

The 100-question sanity suite remains 0/100, which is the already accepted
state of this experimental A8 baseline and is not an accuracy gate. More
importantly for migration control, all three per-question CSV files are
byte-identical:

```text
SHA-256 e3904e8750692efd579f94a5d10369f1340efb2232007093cf54f3d2d18536cd
```

Thus changing QAIRT and finalization scheduling did not change any generated
answer in the deterministic suite.

## Hardware acceptance

Fresh-process s1 and s32 Optrace both pass the existing true-W4A8 contract:

| Evidence per graph | Result |
|---|---:|
| Target W4G32 operations observed | 1009 / 1009 |
| Native U8 RMSNorm operations observed | 729 / 729 |
| Physical U16 target operations | 0 |
| Explicit RMSNorm A8/A16 bridges | 0 |
| Added CPU fallback | 0 |

The default and P19 acceptance JSON files are byte-identical. The P19
published context is:

```text
D:\llm_exp\models\qwen3_sm8750_v79\g32\w4a8_rmsnorm_u8_qairt249\20260823\p19\qwen3-1.7B-w4a8g32-rmsnorm-u8-qairt249-p19.bin
SHA-256 df67e7f1410cc58a348f7212176628afdc97465c06fb6231236cddb581a3e7c6
```

## Performance

The profiling-off throughput values are medians of three fresh runner
processes. There is no speed threshold; the table records the measured
migration outcome.

| Context | Prefill | Decode after first token |
|---|---:|---:|
| QAIRT 2.47 RMSNorm-U8 P19 reference | 797.735 tok/s | 41.682 tok/s |
| QAIRT 2.49 RMSNorm-U8 default | 752.126 tok/s | 41.496 tok/s |
| QAIRT 2.49 RMSNorm-U8 P19 | 783.838 tok/s | 45.313 tok/s |
| 2.49 P19 vs 2.47 P19 | -1.74% | +8.71% |
| QAIRT 2.49 W4A16 default | 835.963 tok/s | 46.674 tok/s |
| 2.49 P19 A8 gap to 2.49 A16 | -6.24% | -2.92% |

P19 is selected because, within QAIRT 2.49, it improves the migrated default
by 4.22% in prefill and 9.20% in decode. This validates P19 for the migrated
baseline; it does not claim that point numbers have identical internal meaning
across QAIRT releases or that an exhaustive 2.49 P-point search was performed.

## Published evidence

- selected P19 result:
  `D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_qairt249_p19_20260823_132031`
- default control result:
  `D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_qairt249_default_20260823_130146`
- primary 2.47 reference:
  `D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_p19_20260821_231318`
- 2.49 W4A16 context reference:
  `D:\llm_exp\results\qwen3_sm8750_v79_w4a16_qairt249_default_20260822_223744`

The canonical report to inspect is
`qwen3-sm8750-v79-g32-e2e-critical-path.html` in the selected P19 result.
