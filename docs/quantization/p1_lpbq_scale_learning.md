# P1 first-stage LPBQ scale learning

This is the first P1 extension beyond static A8 scale optimization. It is a
layer-by-layer/block streaming prototype; it never keeps a 28-layer autograd
graph.

## Fixed deployment choices

- G32 LPBQ int4 weight codes are fixed after the initial quantization.
- `scale1` is learned through a rounded straight-through estimator and is
  exported as UInt4 in `[1,16]`.
- `scale2` is learned as a positive per-output-channel FP32 scale and exported
  as the LPBQ level-2 scale.
- A8 zero-point is never a trainable tensor. `--fixed-zero-point 0` or
  `--fixed-zero-point 128` overrides the map; only A8 scale/clipping is
  optimized. `--fixed-zero-point map` preserves the calibrated integer
  zero-point for comparison.

The implementation is in `pymllm/quantization/static_a8.py`:
`LearnableLPBQScale` keeps codes frozen, uses deploy-shaped rounded `scale1`
in the forward pass, and exposes `export()` for the final LPBQ scales.

## Block trainer

```bash
python scripts/qwen3_p0_block_optimize.py \
  --sensitivity-map artifacts/p0/static_a8/mixed-precision-map.json \
  --layers 27 \
  --steps 100 \
  --lr 0.01 \
  --learn-weight-scale \
  --fixed-zero-point 128 \
  --output-json artifacts/p1/layer27-lpbq-a8-scale-zp128.json
```

The loss remains block-output NMSE plus cosine on captured training examples;
the teacher capture is `no_grad`, and only one decoder block is wrapped and
optimized at a time. After optimization, the trainer reinstalls the exported
UInt4/FP32 LPBQ scales and reports `exported_deploy_block_scales` separately.

## Initial result: layers 2 and 27

| layer | fixed A8 zero-point | initial block NMSE | learned block NMSE | exported deploy NMSE | exported cosine |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 0.000905 | 0.000096 | 0.000096 | 1.000000 |
| 27 | 128 | 0.081913 | 0.002119 | 0.002122 | 0.998919 |
| 27 | 0 | 0.091666 | 0.005951 | 0.005956 | 0.996993 |
| 27 (all-A8 map) | 128 | 0.691248 | 0.153541 | 0.154696 | 0.941242 |

The learned LPBQ scale removes most of the layer-27 activation-quantization
error in this block-only experiment. The near-identical learned/exported
scores show that the improvement is not coming from a continuous
training-only weight scale. This is not yet a full-model result: scales must
be collected and trained layer-by-layer for all 28 layers, then composed and
checked against the deployment/QNN contract.

The all-A8 row is the strict activation-A8 test. It improves substantially
from LPBQ scale learning but remains far worse than the mixed-map row, showing
that fixed-zp A8 clipping/activation error is still a separate problem.

Stage Two (static calibration for output activation, SiLU, KV, softmax,
residual and lm_head) is not implemented by this prototype; those paths remain
outside the trainer.
