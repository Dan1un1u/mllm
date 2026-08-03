# Selective-A16 block ablation

The tensor-local map was tested with one layer modified at a time.  Each row
still uses G32 LPBQ weights; only the listed activation inputs are forced to
A16.

| layer | selected mixed | force `down_proj` A16 | force `up_proj` A16 | force `gate_proj` A16 | force gate+up A16 | force all MLP A16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 block NMSE | 0.452274 | 0.449146 | 0.446132 | **0.093598** | **0.089648** | 0.081883 |

For comparison, layer 27 W4A16 block NMSE is 0.081759.  Making attention
q/k/v/o A16 while leaving the MLP A8 makes the result worse (0.477435), so
the dominant failure is the gate/up → SiLU → down nonlinear path, especially
the gate branch.  Layer 0 shows the same direction but a smaller effect:
selected mixed is 0.041240 and forcing `gate_proj` A16 gives 0.040433 versus
W4A16 0.032412.

This supports a conservative P0 fallback policy: do not infer deployable
precision from Linear-local NMSE alone; sensitive MLP gate/up tensors need a
block-level gate, and `gate_proj` A16 is the first fallback to test.  No QNN
W4A8 export should be attempted from the current tensor-local map.

