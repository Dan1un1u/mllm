# P0 mixed-precision fallback gate

The policy was generated from the complete 28-layer tensor-local map and the
one-layer block gate.  A layer is considered risky when either selected mixed
block-output NMSE is above `0.01` or held-out last-token logits cosine is below
`0.99`.  Risky layers fall back to A16 for the MLP `gate_proj`, `up_proj`, and
`down_proj` inputs; attention remains A8.  Existing tensor-local A16 choices
are preserved.

## Map

- Risky layers: `0, 1, 2, 5, 7, 9, 17, 27` (8 of 28).
- A16 tensors: 46 of 196; A8 tensors: 150 of 196.
- Weight contract remains unrotated LPBQ W4 G32.
- The map is offline-only and introduces no runtime rotation/operator.
- Full-layer A16 overrides: `0, 5, 9, 17` (attention ablation); the other
  risky layers use MLP-only fallback.

Artifact: `mixed-precision-map.json`.

## Block gate

The table compares the previous tensor-local mixed map with the new block-risk
fallback.  Values are block-output NMSE / held-out logits cosine.

| layer | tensor-local map | mixed fallback | A16 fallback |
|---:|---:|---:|---|
| 0 | 0.041240 / 0.983323 | 0.032412 / 0.992153 | full layer |
| 1 | 0.011375 / 0.996120 | 0.009284 / 0.993984 | MLP |
| 2 | 0.021405 / 0.768184 | 0.000766 / 0.984312 | MLP |
| 5 | 0.000004 / 0.980177 | 0.000002 / 0.990391 | full layer |
| 7 | 0.000005 / 0.989772 | 0.000005 / 0.990792 | MLP |
| 9 | 0.000008 / 0.989161 | 0.000006 / 0.993788 | full layer |
| 17 | 0.000174 / 0.989705 | 0.000123 / 0.995138 | full layer |
| 27 | 0.452274 / 0.983495 | 0.081883 / 0.993668 | MLP |

The layer-2 failure is essentially removed by the MLP fallback, and layer 27
improves from `0.452274` to `0.081883` block NMSE while logits cosine rises to
`0.993668`.  The layer-0 attention ablation reaches its W4A16 baseline
(`0.032412` / `0.992153`).  The remaining layer-2 logits cosine (`0.984312`)
is already close to that layer's W4A16 ceiling (`0.985949`), indicating a
weight-side limit rather than an activation-A8-only problem.  This is not yet
a full-model accuracy claim.

The one-layer full result is in `block-eval-mixed-policy-all.json`.  The
composed full-model gate is summarized in `FULL_MODEL_RESULTS.md`: the current
46/196-A16 map reaches `0.855164` logits cosine / `0.833333` top-1, while
promoting all eight risky layers to full A16 reaches `0.894788` / `1.0`.
QNN AOT and ADB have not been run for these Python maps yet; they remain the
deployment validation gate after the scale contract is frozen.
