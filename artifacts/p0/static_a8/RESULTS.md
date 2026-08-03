# P0 static A8 first matrix

Run date: 2026-08-03.  The matrix used the WSL CUDA environment with
`torch 2.11.0+cu128` on an RTX 5070 Ti Laptop.  Fit rows are the fixed train
shards (127 rows) and metrics are computed only on the held-out shards (192
rows).  Every row uses the real Qwen3-origin weight, unrotated LPBQ W4 G32,
and compares W4A8 against the same G32-decoded weight with an A16 input.

| slice | Max-Min NMSE / cosine | P99.9 NMSE | P99.99 NMSE | learnable P99.9 NMSE / cosine |
| --- | ---: | ---: | ---: | ---: |
| layer 0 `o_proj` | 0.00133856 / 0.99933124 | 0.00805278 | 0.00119164 | **0.00113793 / 0.99943137** |
| layer 0 `down_proj` | **0.02319706 / 0.98841596** | 0.19159436 | 0.04687382 | 0.02383364 / 0.98804766 |
| layer 13 `o_proj` | **0.00427611 / 0.99786901** | 0.01085416 | 0.00337932 | 0.00559323 / 0.99719971 |
| layer 13 `down_proj` | 0.00772299 / 0.99615210 | 0.03472515 | 0.00671725 | **0.00497015 / 0.99751198** |
| layer 27 `o_proj` | 0.00154053 / 0.99923021 | 0.00950868 | 0.00249415 | **0.00148508 / 0.99925733** |
| layer 27 `down_proj` | **0.00398520 / 0.99804121** | 0.19148427 | 0.01452133 | 0.00499833 / 0.99782932 |

## Interpretation

- Static A8 is recoverable without rotation on some slices, but there is no
  globally best clipping rule yet.
- P99.9 is unsafe for the observed `down_proj` distributions.  Its held-out
  output NMSE reaches about 0.19 on layers 0 and 27.
- Learnable clipping improves layer 0/27 `o_proj` and layer 13 `down_proj`,
  but loses to Max-Min on layer 13 `o_proj` and on layers 0/27 `down_proj`.
- Therefore P0 has not passed a whole-model gate.  The next software step is
  a layer/tensor sensitivity map and a conservative learned-scale selection
  rule (with Max-Min fallback), not rotation or an AOT W4A8 kernel.

The JSON files beside this report contain the fixed zero-point, effective
clip range, saturation fraction, activation NMSE/cosine, and output
NMSE/cosine.  The output reference is W4A16 with the same decoded G32 weight;
weight quantization error is reported separately and is not folded into the
A8 comparison.

