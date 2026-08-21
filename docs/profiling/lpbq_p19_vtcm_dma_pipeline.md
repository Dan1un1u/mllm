# LPBQ P19 VTCM/DMA pipeline experiment

Date: 2026-08-21
Target: SM8750 / HTP V79
QAIRT: 2.47.0.260601
Branch: `codex/w4a8g32-lpbq-p-point-search`

## Outcome

The experiment gate passes. QAIRT finalize search point P19 improves the
accepted native-U8 RMSNorm W4A8 model without changing its graph-level
quantization contract or numerical output:

- manifests are byte-identical to the accepted RMSNorm-A8 model;
- the 100-case output CSV is byte-identical;
- s1/s32 Optrace acceptance passes with 1009/1009 W4G32 targets, 729/729
  physical U8 RMSNorm operations, zero physical U16 target operations, and
  zero RMSNorm bridges;
- profiling uses the same QAIRT 2.47 runtime, runner, prompt, three
  fresh-process E2E rounds, and fresh-process s1/s32 Optrace protocol as the
  accepted RMSNorm-A8 and archived A16 baselines.

Relative to accepted RMSNorm-A8, P19 improves the median full-model result by
1.20% for prefill throughput and 9.86% for decode throughput. It does not close
the A16 gap: P19 remains 7.24% slower in prefill and 8.37% slower in decode.

The physical explanation is now more precise than "weight DMA is slower".
Most of the regressing `weights_to_vtcm` cycles are dependency waits rather
than payload-transfer work. P19 lowers the full-graph VTCM high-water mark and
removes most weight-DMA wait from gate/up/lm_head. The remaining critical path
shifts toward W4 expansion, bias DMA, and synchronization. P19 therefore
repairs overlap; it does not reduce the compressed W4 payload or replace the
LPBQ/HMX algorithm.

## Controlled implementation

The default code path is unchanged. Setting
`MLLM_QNN_AOT_FINALIZE_P=<point>` adds QAIRT's `FINALIZE_CONFIG` key `P` to an
otherwise unchanged O=3 AOT graph. QAIRT 2.47 accepts these points:

`0,1,2,3,4,5,6,8,13,15,16,17,19,20,21,22,23`

The compiler log proves that both full-model graphs consumed P19:

```text
Prepare: Graph model.0.s32 with init graph option: P = 19
Prepare: Graph model.0.s1 with init graph option: P = 19
```

P-point selection is a graph-finalize scheduling choice. It does not alter
weights, activation encodings, calibration parameters, model structure, or
runtime code.

## Search and focused micrographs

All 17 QAIRT 2.47 P points were first screened on byte-identical gate/up s1
W4G32/A8 projection artifacts. P2, P15, and P19 were then compared with the
default compiler choice over ten fresh processes; each process used 50 warmup
and 1000 measured graph executions. P19 was the only common, stable winner.

| Projection | Default A8 median | P19 median | Paired median delta | Wins/ties |
|---|---:|---:|---:|---:|
| gate s1 | 189.5 us | 184.5 us | -4.5 us | 8/1 |
| up s1 | 189.0 us | 183.5 us | -7.0 us | 8/1 |
| lm_head s1 | 4115.5 us | 3806.0 us | -292.3 us | 10/0 |

Device outputs are byte-exact between the default and P19 contexts.

The micrograph VTCM and Optrace evidence identifies the same mechanism:

| Projection | Default A8 peak VTCM | P19 peak VTCM | Default weight-wait dominant | P19 weight-wait dominant |
|---|---:|---:|---:|---:|
| gate s1 | 3,467,264 B | 2,707,456 B | 136,375 cycles | 120,378 cycles |
| up s1 | 3,467,264 B | 2,707,456 B | 127,880 cycles | 104,220 cycles |
| lm_head s1 | 5,758,976 B | 3,526,656 B | 4,051,905 cycles | 3,056,052 cycles |

Actual compressed-weight DMA work remains close; the large cost classified as
`weights_to_vtcm` is primarily an empty-shape DMA-wait event. P19 changes the
checkpoint/runlist arrangement and reduces how long HMX waits for the next
expanded tile.

## Full-model performance

| Model | Prefill median | Decode median | Prefill vs RMS-A8 | Decode vs RMS-A8 |
|---|---:|---:|---:|---:|
| archived W4A16 | 859.964 tok/s | 45.490 tok/s | +9.10% | +19.89% |
| accepted RMSNorm-A8 | 788.252 tok/s | 37.942 tok/s | reference | reference |
| RMSNorm-A8 + P19 | 797.735 tok/s | 41.682 tok/s | +1.20% | +9.86% |

The profiling-off P19 medians are 77,720 us for 62 prefill tokens and
1,511,434 us for 63 decode tokens. The three P19 rounds span 795.20--799.19
prefill tok/s and 41.44--42.01 decode tok/s.

One fresh-process Optrace per graph gives the same direction:

| Graph | A16 execute | RMS-A8 execute | P19 execute | P19 vs RMS-A8 |
|---|---:|---:|---:|---:|
| s1 | 18,999 us | 21,829 us | 21,100 us | -3.34% |
| s32 | 21,210 us | 23,002 us | 22,282 us | -3.13% |

The full-graph VTCM high-water mark falls from 6,844,416 to 6,395,904 bytes
for s1 (-6.55%) and from 7,208,960 to 6,434,816 bytes for s32 (-10.74%). Both
P19 values are also below the corresponding A16 trace.

## Projection critical-path attribution

The table reports the stage `num_dominant_path_cycles_htp_0` delta from
accepted RMSNorm-A8 to P19.

| Projection | s1 RMS-A8 -> P19 | s1 delta | s32 RMS-A8 -> P19 | s32 delta |
|---|---:|---:|---:|---:|
| gate | 8,118,157 -> 7,297,615 | -10.11% | 7,294,136 -> 7,196,659 | -1.34% |
| up | 8,483,252 -> 7,791,882 | -8.15% | 7,194,463 -> 7,498,788 | +4.23% |
| down | 6,414,454 -> 6,442,529 | +0.44% | 8,054,951 -> 5,880,215 | -27.00% |
| lm_head | 8,590,443 -> 7,413,859 | -13.70% | 7,556,439 -> 7,096,195 | -6.09% |

P19 is therefore not a uniform per-projection acceleration. Its major wins are
decode gate/up/lm_head and prefill down/lm_head. Prefill up regresses slightly,
which explains why the full prefill gain is much smaller than the decode gain.

### Weight DMA wait

P19 sharply reduces the dominant contribution of
`q::ConvLayer.opt.weights_to_vtcm` DMA-wait events:

| Projection | s1 reduction | s32 reduction |
|---|---:|---:|
| gate | -80.2% | -80.7% |
| up | -83.4% | -78.7% |
| down | -50.3% | -72.6% |
| lm_head | -90.5% | -32.1% |

Compressed weight DRAM bytes are unchanged. This is a scheduling/overlap win,
not a bandwidth reduction.

### Where the bottleneck moves

The saved weight-wait time is not free. In s1, W4 expansion work changes by
only about -3% to +6%, but its exposed dominant-path cycles increase. Gate/up
and lm_head also acquire more bias-DMA wait and more synchronization work.
For lm_head s1, HMX work rises from 1.505M to 1.847M cycles while the whole
stage still improves by 13.70%; the removed 3.58M cycles of dominant
weight-wait outweigh the extra exposed compute.

This is the expected signature of a better pipeline schedule:

```text
default A8: DMA wait -------- expand/HMX gaps -------- DMA wait
P19:        DMA -> expand -> HMX -> DMA -> expand -> HMX
                         ^ more useful work is now visible
```

The next limiter is no longer the compressed-weight transfer itself. It is
the balance among expansion, zero-bias materialization/DMA, checkpoints, and
HMX tile consumption.

## Interpretation and next experiment

P19 is a useful optimization candidate, especially for decode, but it should
not yet replace the accepted RMSNorm-A8 baseline silently. It is a compiler
schedule derivative with a clear positive result and a remaining graph-shape
tradeoff.

The next highest-value experiment is per-graph P selection:

1. retain P19 for s1, where decode E2E and gate/up/lm_head all improve;
2. screen valid P points on s32 gate/up/down/lm_head micrographs;
3. only compile full s32 candidates that do not regress up and preserve the
   P19 down/lm_head gains;
4. add separate `s1` and `s32` P controls only after that screen, keeping the
   current global environment variable as the reproducible P19 experiment.

If a better s32 point is not found, the next kernel-adjacent target is the
newly exposed bias-DMA/checkpoint overhead, not W4 decoding. The earlier
explicit-zero-bias experiment did not improve the default path, so any retry
must be tied to P19's changed schedule and verified by Optrace rather than
assuming that removing a logical bias removes the physical bias tile.

## Artifacts

- P-point contexts and compiler evidence:
  `D:\llm_exp\models\qwen3_sm8750_v79\g32\lpbq_p_point_search\20260821`
- P19 full-model context and contract:
  `D:\llm_exp\models\qwen3_sm8750_v79\g32\w4a8_rmsnorm_u8_p19\20260821`
- P-point focused gate/up results:
  `D:\llm_exp\results\qwen3_sm8750_v79_lpbq_p_point_focus_20260821`
- P19 lm_head results:
  `D:\llm_exp\results\qwen3_sm8750_v79_lpbq_p19_lm_head_20260821`
- P19 full-model results and canonical report:
  `D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_p19_20260821_231318`

