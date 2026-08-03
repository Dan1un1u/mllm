# P1 prefix-aware full-model streaming training

The trainer `scripts/qwen3_p1_streaming_train.py` runs the model layer by
layer. For layer `L`, the prefix `[0, L)` is already replaced by fixed
exported LPBQ/A8 wrappers; only layer `L` has an autograd graph. After the
update, its UInt4 `scale1`, FP32 `scale2`, and fixed A8 parameters are saved
and installed before collecting layer `L+1`.

The non-linear paths remain BF16/A16 and no dynamic bit-width or channel mask
is introduced.

## Full-model results

| variant | A16 Linear inputs | target mode | logits cosine | top-1 | logits NMSE | block NMSE | layer-27 block NMSE |
|---|---:|---|---:|---:|---:|---:|---:|
| W4A16 reference | 196 | n/a | 0.916866 | 1.000000 | 0.159378 | 0.016657 | n/a |
| old mixed map | 46 | n/a | 0.855164 | 0.833333 | 0.270244 | 0.018852 | n/a |
| streaming mixed map | 46 | prefix | 0.895313 | 0.833333 | 0.207467 | 0.006163 | 0.077292 |
| streaming all-risk map | 62 | prefix | 0.889310 | 1.000000 | 0.221470 | 0.005967 | 0.071695 |
| streaming all-risk map | 62 | teacher | 0.882484 | 0.666667 | 0.232572 | 0.004936 | 0.058266 |

Artifacts:

- `artifacts/p1/streaming-full-mapzp/streaming-train.json`
- `artifacts/p1/streaming-full-allrisk-mapzp/streaming-train.json`
- `artifacts/p1/streaming-full-allrisk-teacher/streaming-train.json`

Each directory contains one safetensors LPBQ scale file per layer, so the
trained result is not a JSON-only training artifact.

## Interpretation

Prefix-aware streaming is materially better than independently trained
teacher-input blocks: the selected 46/196-A16 map improves logits cosine from
`0.855164` to `0.895313`, while aggregate block NMSE falls from `0.018852` to
`0.006163`. The more conservative 62/196-A16 map preserves held-out top-1
agreement at `100%`, matching W4A16, but its cosine is `0.889310` versus
`0.916866`.

Teacher-target streaming lowers block NMSE further but hurts final logits;
for this model the prefix-target objective is the better composition choice.

Under the relaxed GO criterion of no task/top-1 drop, the 62/196-A16 prefix
variant is a provisional candidate. It is not cosine-parity GO: the remaining
`0.027556` absolute cosine gap to W4A16 is measurable. QNN AOT and ADB have
not yet been run for these per-layer safetensors.
