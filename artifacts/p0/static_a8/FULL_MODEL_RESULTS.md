# P0 composed full-model gate

`qwen3_p0_full_eval.py` applies the same G32 LPBQ decode and static A8 fake
quantization to all 28 decoder layers, then evaluates the fixed six-prompt
held-out split. Metrics are final last-token logits against the BF16 teacher.

| variant | A16 tensors | A8 tensors | logits cosine | top-1 agreement |
|---|---:|---:|---:|---:|
| W4A16 weight upper bound | 196 | 0 | 0.916866 | 1.000000 |
| tensor-local map | 11 | 185 | 0.692846 | 0.500000 |
| selected mixed policy | 46 | 150 | 0.855164 | 0.833333 |
| all eight risk layers full A16 | 62 | 134 | 0.894788 | 1.000000 |

Artifacts:

- `full-model-eval-mixed-policy.json`
- `full-model-eval-tensor-local.json`
- `full-model-eval-all-risk-full.json`

The mixed policy substantially improves over the tensor-local map, and making
all eight block-risk layers full A16 restores top-1 agreement on this split.
However, even the W4A16 weight-only upper bound is `0.916866`, below the P0 GO
target of `0.95`. Static A8 scale optimization alone therefore cannot satisfy
that target under the current G32 LPBQ weight oracle. Further A16 fallback is
not a substitute for improving the weight-scale contract; the next experiment
should learn activation clipping/scale together with a weight-scale objective
and report against this explicit `0.916866` ceiling.

No QNN AOT or ADB run was performed for these Python oracle maps.
