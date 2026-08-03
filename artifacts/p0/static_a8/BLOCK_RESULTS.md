# P0 block-output gate (representative layers)

This gate runs the BF16 Qwen3 teacher on the six held-out prompts, then
re-runs one decoder layer at a time with all seven Linear weights decoded as
G32 LPBQ.  `w4a16` keeps those layer inputs in A16; `selected_mixed` applies
the tensor-local sensitivity-map choices, including its A16 fallbacks.  All
other layers stay BF16, so this is not yet a full-model quantized result.

| layer | W4A16 block NMSE | selected mixed block NMSE | W4A16 logits cosine | mixed logits cosine | mixed top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.032412 | 0.041240 | 0.992153 | 0.983323 | 83.3% |
| 13 | 0.000018 | 0.000024 | 0.996557 | 0.996585 | 100% |
| 27 | 0.081759 | **0.452274** | 0.993461 | 0.983495 | 100% |

The layer-27 result is the key gate failure: locally selected A8 scales can
compose badly through attention/MLP nonlinearities even when every individual
Linear passed the NMSE screening threshold.  Therefore the next P0 step is
block-level scale selection (or block-output distillation) with Max-Min/A16
fallback, not exporting the tensor-local map directly to QNN.

