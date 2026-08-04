# P1 extended static-scale training results (2026-08-04)

## Scope

This run keeps the deployment constraints unchanged:

- no rotation;
- LPBQ W4 G32 weights with fixed int4 codes;
- static affine A8 input scales with integer zero-points taken from the
  selected calibration map;
- SiLU, RMSNorm, softmax, KV, residual, and lm_head remain BF16/A16;
- one decoder block has an autograd graph at a time, and its exported scale
  files replace the prefix before the next block is calibrated.

The trainer now uses a two-stage curriculum per block:

1. W4A16 LPBQ-only optimization for 200 steps (`lr=0.003`);
2. joint static A8 scale and LPBQ scale optimization for 400 steps
   (`lr=0.005`).

Both stages use 30-step warmup, cosine decay to 10% of the stage learning
rate, gradient clipping at 1.0, and a deployment round-trip evaluation every
50 steps. Candidate selection is performed after rounding `scale1` to UInt4,
rebuilding the decoded LPBQ weight, and applying the fixed A8 zero-point.

## Source-aligned rerun (authoritative for VM)

The earlier extended runs regenerated INT4 codes from the BF16 teacher inside
the trainer. They remain useful software experiments, but they are **not** the
deployment artifact for the VM. The authoritative rerun uses two explicit
sources:

- BF16 teacher: `/home/daniuniu/llm_exp/models/Qwen3-origin`;
- fixed G32 INT4 code and initial `scale1/scale2`:
  `/mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors`.

The trainer now decodes the base file's HWIO carrier to logical OI signed INT4
codes, keeps those codes fixed, and learns only the LPBQ scales/A8 scales. The
aligned manifest records both `codes_sha256` (logical OI) and
`packed_codes_sha256` (HWIO low-nibble carrier). All `196/196` projection hashes
match the G32 base checkpoint; the base file SHA-256 is
`6caa0d36ef6846ad2ab138335204699d9abd44c7dd30df25a5846a6b3e5e6bae`.

## Held-out software-oracle results

| variant | A16 Linear inputs | logits cosine | top-1 agreement | logits NMSE | aggregate block NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| W4A16 reference | 196/196 | 0.916866 | 1.000000 | 0.159378 | 0.016657 |
| previous prefix mixed | 46/196 | 0.895313 | 0.833333 | 0.207467 | 0.006163 |
| previous prefix all-risk | 62/196 | 0.889310 | 1.000000 | 0.221470 | 0.005967 |
| **extended prefix mixed** | **46/196** | **0.903255** | **0.833333** | **0.204042** | **0.005287** |
| **extended prefix all-risk** | **62/196** | **0.918487** | **1.000000** | **0.169250** | **0.005310** |
| **source-aligned prefix all-risk** | **62/196** | **0.915216** | **1.000000** | **0.171894** | **see manifest** |

The extended mixed map improves cosine by `+0.007942` over the previous mixed
run. The extended all-risk map improves cosine by `+0.029177`, preserves the
W4A16 held-out top-1 agreement, and is `+0.001621` above the recorded W4A16
cosine reference in this software oracle. This is a candidate result, not a
native-QNN accuracy claim.

The source-aligned all-risk result is the one to take to the VM: logits cosine
`0.915216`, top-1 agreement `1.000000`, with fixed codes from the G32 base. The
earlier `0.918487` result used teacher-regenerated codes and must not be used for
QNN checkpoint construction.

## Artifacts

- mixed manifest and 28 exported scale files:
  `artifacts/p1/streaming-full-robust-prefix-mapzp/`
- all-risk manifest and 28 exported scale files:
  `artifacts/p1/streaming-full-robust-prefix-allrisk-mapzp/`
- source-aligned all-risk manifest and 28 exported scale files (VM input):
  `artifacts/p1/aligned-full-allrisk-base-g32-mapzp/`
- one-layer dry-run:
  `artifacts/p1/streaming-dryrun-enhanced/`

Each full run reports `complete=true`, contains 28 per-layer rows, and exports
28 `layerNN-lpbq-scales.safetensors` files. The GPU run completed without OOM.

## Interpretation and next gate

The result shows that the previous 100-step/fixed-learning-rate setup was not
the static-scale ceiling. Deployment-aware curriculum and candidate selection
recover most of the earlier cosine gap without rotation. For VM construction,
the source-aligned 62/196 all-risk map is the candidate: it preserves top-1
parity and its 196 code hashes match the authoritative G32 base.

The result is still not a QNN deployment result: this branch does not yet have
the native `kQNN_LPBQ_w4a8o8_G32` backend. Before calling this a device GO,
generate scheme-specific G32 checkpoints, finalize V79 contexts with QAIRT
2.47.0.260601, run the VM AOT/ADB wrappers, and compare the native oracle and
optrace. A separate fixed-`zp=128` run is also required if the final QNN graph
cannot retain the per-tensor integer zero-points from the calibration map.
