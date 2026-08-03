# P0 static A8 scale study

P0 keeps the deployment contract deliberately narrow:

- no rotation;
- unrotated LPBQ W4 G32 weights;
- static per-tensor affine A8 for Linear inputs;
- unsigned quantization range `[0, 255]`;
- the integer zero-point is frozen after calibration;
- only the scale / effective clipping range is optimized;
- SiLU, RMSNorm, softmax and KV remain on their existing A16/W8A8 paths.

The pure PyTorch oracle is `pymllm/quantization/static_a8.py`.  It implements
Max-Min, mean/3-sigma, percentile clipping, and a learnable scale initialized
from percentile clipping.  `scripts/qwen3_p0_static_a8.py` loads one input
activation key from BF16 safetensor shards and one real Qwen3 weight tensor,
then reports W4A16 versus each W4A8 candidate.  The reference output is the
same frozen G32-decoded weight with an unquantized input, so the experiment
isolates activation A8 instead of hiding weight error in the target.

Run it in the GPU WSL environment, for example:

```bash
python scripts/qwen3_p0_static_a8.py \
  --inputs /home/daniuniu/llm_exp/calibration/qwen3-p1-layers-0-13-27-seed17-s96 \
  --input-key layer_00.o_proj_input \
  --weight /home/daniuniu/llm_exp/models/Qwen3-origin \
  --weight-key model.layers.0.self_attn.o_proj.weight \
  --output-json /home/daniuniu/llm_exp/p0/layer00-o-proj.json
```

The output records the G32 scale1/scale2 shapes, effective clipping range,
saturation fraction, activation NMSE/cosine, output NMSE/cosine, and the
learnable optimizer's initial/final loss.  It is intentionally a layer-level
oracle; whole-model calibration and QNN AOT are separate gates.

For the complete 28-layer map, first collect all seven Linear inputs:

```bash
python scripts/qwen3_collect_p0_calibration.py \
  --model /home/daniuniu/llm_exp/models/Qwen3-origin \
  --prompt-tsv scripts/qwen3_sm8750_v79_accuracy.tsv \
  --output-dir /home/daniuniu/llm_exp/calibration/qwen3-p0-all-layers-seed17-s96

DEVICE=cuda bash scripts/run_qwen3_p0_sensitivity_map.sh
```

The sensitivity map is tensor-local only.  Before exporting a mixed-precision
map, run `scripts/qwen3_p0_block_eval.py` and require block-output/held-out
logits checks; local best scales can compose badly through the SiLU gate/up
path.

For QNN commands on Windows, dot-source
`scripts/use_qairt_2_47_0_260601.ps1` first.  This selects
`D:\llm_exp\models\qualcomm-sdk\qairt\2.47.0.260601`; the P0 Python oracle
does not import QNN or `MllmFFIExtension.so`.

## Mixed-precision fallback

The deployable fallback map is generated from the tensor-local sensitivity
map plus the one-layer block gate:

```bash
# The current P0 ablation promotes layers 0, 5, 9 and 17 to full A16.
python scripts/qwen3_p0_make_mixed_precision_map.py \
  --full-a16-layers 0,5,9,17
python scripts/qwen3_p0_block_eval.py \
  --sensitivity-map artifacts/p0/static_a8/mixed-precision-map.json \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --output-json artifacts/p0/static_a8/block-eval-mixed-policy-all.json
```

The conservative policy falls back to A16 for the MLP gate/up/down inputs in
layers whose selected mixed block NMSE is above `0.01` or whose held-out
last-token logits cosine is below `0.99`.  Attention q/k/v/o remains A8 unless
the tensor-local screen or an explicit ablation promotes a layer.  The current
ablation promotes layers `0,5,9,17` to full A16.  This is an offline precision
map: it does not add a runtime rotation or any new operator.

For the composed full-model gate, use `scripts/qwen3_p0_full_eval.py`.  It
evaluates all 28 layers together on the same held-out prompts and reports
final last-token logits against the BF16 teacher.  This catches error
accumulation that a one-layer block gate cannot see.
