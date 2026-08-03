# Full P0 static-A8 sensitivity map

Run date: 2026-08-03.  This map uses all 28 Qwen3 decoder layers and all
seven Linear inputs (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`).  The fixed prompt split has 127 train rows and 192
held-out rows.  Every row uses real unrotated LPBQ W4 G32 weights; the
reference is W4A16 with the same decoded weight.

The screening rule is `best held-out output NMSE > 0.02 => A16`.  This is a
conservative tensor-level heuristic, not a final block/model accuracy gate.

## Summary

- 196 tensors evaluated.
- 185 pass the A8 screening heuristic; 11 recommend A16.
- All 11 A16 recommendations are `down_proj` inputs.
- Best strategy counts: learnable P99.9 = 96, P99.99 = 64, Max-Min = 36.
- Mean best NMSE by projection: q 0.001588, k 0.001472, v 0.003135,
  o 0.001876, gate 0.001569, up 0.003206, down 0.021614.

## A16 screening rows

| layer | tensor | best strategy | held-out NMSE |
| ---: | --- | --- | ---: |
| 0 | down_proj | Max-Min | 0.023197 |
| 1 | down_proj | Max-Min | 0.042535 |
| 3 | down_proj | Max-Min | 0.037809 |
| 4 | down_proj | learnable P99.9 | 0.039167 |
| 7 | down_proj | Max-Min | 0.068067 |
| 9 | down_proj | Max-Min | 0.049088 |
| 10 | down_proj | Max-Min | 0.081883 |
| 11 | down_proj | Max-Min | 0.043675 |
| 14 | down_proj | Max-Min | 0.021500 |
| 17 | down_proj | learnable P99.9 | 0.021933 |
| 22 | down_proj | Max-Min | 0.021588 |

The full per-tensor candidate metrics, fixed zero-points, clipping ranges,
and saturation fractions are in `sensitivity-map.json`.  These results do
not yet justify changing the whole-model bit-width map: the next gate is a
block-output reconstruction test using the selected scale per tensor, then a
held-out logits comparison.  QNN AOT and runtime profiling remain deferred.

