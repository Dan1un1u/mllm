# Qwen3 K/V head-packing micrograph

## Decision

K/V head packing is mathematically valid and satisfies the literal non-regression gate, but it does not provide a robust speedup on QAIRT 2.47 / SM8750. Do not migrate the full model yet.

The fair experiment reports only 0.5%–1.6% lower profiling-off medians. Each case wins only five or six of ten paired fresh-process rounds, and the Optrace direction is mixed. This is device/run variation rather than sufficient evidence that packing removes the A8 bottleneck.

## Experiment contract

- Source: accepted native-U8 RMSNorm W4A8G32 artifact, layer 14.
- Target: SM8750 / V79, QAIRT 2.47.0.260601, graph-finalize `P=19`.
- Control: eight independent `2048 -> 128` K or V LPBQ Conv2d projections.
- Candidate: one `2048 -> 1024` LPBQ Conv2d projection, sliced back to eight `[1,S,128]` outputs.
- Activations: identical asymmetric per-tensor U8 input and output encodings.
- Weights: byte-equivalent signed W4 `[-7,7]`, G32 LPBQ carrier and scales.
- Boundary: both variants expose eight graph outputs with identical shapes and qparams.
- Shapes: `S=1` and `S=32` for both K and V.
- Timing: ten alternating fresh-process rounds; each uses 50 warmups and 1000 measured executions with QNN profiling disabled.
- Trace: one Optrace capture per case using the same contexts and device.

This boundary matters. An earlier diagnostic exposed eight graph outputs for the control but only one for packed. That version appeared 25%–31% faster because it removed seven output bindings/slices; its internal projection timeline was not faster. Those numbers are rejected as a head-packing result.

## Correctness

All four fair comparisons are byte-exact between the per-head and packed device outputs and repeat byte-exactly on a second execution.

| Projection | Shape | Device equality | Host sampled reference |
| --- | ---: | --- | --- |
| K | S=1 | exact | max U8 code delta 1; 98.44% exact |
| K | S=32 | exact | max U8 code delta 1; 96.48% exact |
| V | S=1 | exact | max U8 code delta 1; 98.44% exact |
| V | S=32 | exact | max U8 code delta 1; 98.05% exact |

The one-code host discrepancy is consistent with the host reference's floating-point reconstruction versus the deployed HTP LPBQ arithmetic. It is not a difference between the two graph expressions.

## Profiling-off latency

| Projection | Shape | Per-head median | Packed median | Nominal speedup | Packed wins / 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K | S=1 | 96.5 us | 95.0 us | 1.58% | 5 (2 ties) |
| K | S=32 | 93.0 us | 92.5 us | 0.54% | 6 |
| V | S=1 | 96.0 us | 94.5 us | 1.59% | 6 (2 ties) |
| V | S=32 | 93.0 us | 92.0 us | 1.09% | 6 (1 tie) |

All aggregate medians narrowly favor packed, so the literal “not slower” criterion passes. The effect is nevertheless below a defensible optimization signal: the absolute differences are only 0.5–1.5 us, the paired direction is weak, and no case wins consistently across fresh processes.

## Optrace diagnosis

Percentages below are packed relative to per-head; negative is lower.

| Case | Timeline | Projection work | Projection dominant | Weight-DMA wait work | Weight-DMA wait dominant | W4-expand dominant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K S=1 | -17.3% | -8.7% | -23.9% | -22.9% | -21.7% | -30.0% |
| K S=32 | +15.7% | -2.8% | +8.2% | -53.5% | +11.3% | +87.2% |
| V S=1 | +15.1% | +5.6% | +19.0% | +14.2% | +45.5% | +18.3% |
| V S=32 | -4.7% | -14.3% | -11.2% | -58.0% | -27.9% | -22.9% |

Packing does not change the physical amount of model data:

- Weight DRAM read remains 1,114,112 bytes in every case.
- W4 expansion still has 96 physical instances.
- HMX still has 16 physical instances and essentially the same work.
- K/V packed and per-head have the same W4 payload, scale data, activation qparams, and output interface.

What changes is QAIRT's schedule: the same DMA, W4 expansion, and HMX work lands on the dominant path differently. K/S=1 and V/S=32 happen to overlap better in the captured run, while K/S=32 and V/S=1 become worse. Thus head granularity is not the controlling variable; compiler tiling/scheduling specialization is.

## Conclusion and next boundary

The experiment falsifies the simple hypothesis that eight logical K/V heads are themselves causing the A8 regression through extra weight traffic or extra W4 expansion. QAIRT lowers both forms to nearly identical physical work and instance counts. Packing only perturbs scheduling and does not produce a repeatable end-to-end gain.

The result therefore stops at the micrograph boundary. A full-model implementation would add packed-to-head slicing around K RMSNorm/RoPE and the V cache path without an established projection benefit. A future K/V experiment should target a larger fused region that remains packed through downstream consumers, or directly control compiler tiling/double-buffer scheduling; it should not merely replace eight projection nodes with one.

Machine-readable evidence is archived at:

- Models and contexts: `/mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/kv_head_packing/20260822_split8`
- Results and decoded Optrace: `/mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_packing_split8_20260822`
- Unified summary: `/mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_packing_split8_20260822/summary.json`
