# P1 mixed W4A8+A16 full-model gate

This experiment composes the existing static mixed map with the learned
layer-2 and layer-27 A8/LPBQ scales. The map is tensor/layer static:

- 46/196 Linear inputs are A16 and 150/196 are A8 in the selected map.
- Full-layer A16 overrides are layers `0,5,9,17`.
- MLP-only A16 fallbacks are layers `1,2,7,27`.
- Existing tensor-local sensitive `down_proj` fallbacks remain A16.
- SiLU, RMSNorm, softmax, KV, residual and lm_head stay BF16/A16 in this
  Python oracle; no dynamic bit-width branch or channel-level mask is used.

The evaluator also reports aggregate block-output metrics and robust logits
metrics: NMSE, relative absolute error with an epsilon floor, and absolute
error P99.9.

## Held-out results

| variant | A16 Linear inputs | logits cosine | top-1 | logits NMSE | logits relative error | logits abs P99.9 | block NMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| W4A16 upper bound | 196 | 0.916866 | 1.000000 | 0.159378 | 1.694682 | 7.4375 | 0.016657 |
| selected mixed map | 46 | 0.855164 | 0.833333 | 0.270244 | 2.192270 | 10.0889 | 0.018852 |
| selected mixed + learned layer 2/27 | 46 | 0.873683 | 0.833333 | 0.237473 | 2.023929 | 11.1250 | 0.013936 |
| all eight risk layers full A16 | 62 | 0.894788 | 1.000000 | 0.200269 | 1.963159 | 8.2656 | 0.018538 |
| all-risk + learned layer 2/27 | 62 | 0.889973 | 0.833333 | 0.212566 | 2.137260 | 8.8750 | 0.013578 |

Artifacts:

- `artifacts/p1/full-model-eval-mixed-base-metrics.json`
- `artifacts/p1/full-model-eval-mixed-learned.json`
- `artifacts/p1/full-model-eval-all-risk-base-metrics.json`
- `artifacts/p1/full-model-eval-all-risk-learned.json`

## GO decision

The selected learned mixed map improves final logits cosine from `0.855164`
to `0.873683` and reduces aggregate block NMSE from `0.018852` to `0.013936`,
but it does not reach the GO cosine target `0.95`. The all-risk A16 fallback
restores top-1 agreement to `100%` but still reaches only `0.894788`; even the
W4A16 upper bound is `0.916866`. Therefore this route is not GO under the
current G32 LPBQ weight oracle.

The gap between excellent single-block learned scores and the composed full
model (layer 27 reaches `0.157154` block NMSE in the composed run) shows that
calibrating each block against BF16 teacher inputs is not composition-safe.
The next trainer must stream the already-quantized prefix when collecting the
next layer's calibration data, while retaining A16 for the selected
down_proj/nonlinear paths. No QNN AOT or ADB run has been made for these new
learned-scale artifacts yet.
