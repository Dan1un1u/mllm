# Full 28-layer block gate

This is a one-layer-at-a-time gate: for each layer, only its seven Linear
weights are replaced by G32 LPBQ and the tensor-local A8/A16 map is applied.
All other layers remain BF16.  It therefore measures composition risk without
requiring a full-model autograd graph.

## Risk rows

| layer | mixed block NMSE | mixed logits cosine | mixed top-1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.041240 | 0.983323 | 83.3% |
| 1 | 0.011375 | 0.996120 | 100% |
| 2 | 0.021405 | **0.768184** | 66.7% |
| 27 | **0.452274** | 0.983495 | 100% |

The block-NMSE screening threshold of 0.01 flags layers 0, 1, 2 and 27.
Logits are more sensitive: layers 0, 2, 5, 7, 9, 17 and 27 fall below 0.99
cosine even when their block NMSE is tiny.  This is why both block output and
held-out logits must be retained as P0 gates.

Layer 2 MLP-A16 ablation reduces block NMSE to 0.000766 and logits cosine to
0.984312 (near its W4A16-only baseline cosine 0.985949).  Layer 27 MLP-A16
ablation reduces block NMSE to 0.081883 versus 0.452274 for the all-A8 MLP;
the gate branch is the dominant individual fallback.  These are still
single-layer ablations, not a full-model accuracy claim.

