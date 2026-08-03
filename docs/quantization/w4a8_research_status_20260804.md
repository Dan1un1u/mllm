# MLLM QNN W4A8 研究状态（2026-08-04）

## 当前范围

当前工作树为 `codex/quant-npu-w4a8-roadmap`，目标是 Qwen3-1.7B 在
SM8750/V79 上的 G32 LPBQ 权重与静态 A8 activation 研究。原始模型位于
`D:\llm_exp\models\Qwen3-origin`。GPU 训练在 WSL，QNN AOT、Android 构建和
ADB 真机验证回到 Linux VMware VM。

## 已完成成果

### P0：静态 A8 scale optimization

- 无 rotation；权重保持 LPBQ W4 G32。
- A8 为 per-tensor affine QDQ，zero-point 固定为整数常量；训练只优化
  scale/clipping，不保留运行时计算。
- 已实现 Max-Min、Mean/3σ、percentile clipping、learnable clipping 的
  fake-quant 与评估工具。
- 已实现按 projection/layer 的 mixed-precision map，敏感 down_proj、完整
  attention/MLP 风险层可静态回退 A16。
- P0 产物位于 `artifacts/p0/static_a8/`，包括 sensitivity map、mixed map、
  all-risk map、逐 block/full-model 汇总。

### P1：prefix-aware streaming trainer

`scripts/qwen3_p1_streaming_train.py` 按 layer streaming：前缀已经替换为固定
量化 wrapper，只有当前 block 保留 autograd 图，训练结束立即导出并冻结该层。
因此不会把 28 层完整 autograd 图放入 12 GB GPU。

最终 held-out 软件 oracle：

| variant | A16 Linear inputs | logits cosine | top-1 agreement | logits NMSE | block NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| W4A16 reference | 196 | 0.916866 | 1.000000 | 0.159378 | 0.016657 |
| prefix streaming mixed | 46 | 0.895313 | 0.833333 | 0.207467 | 0.006163 |
| prefix streaming all-risk | 62 | 0.889310 | 1.000000 | 0.221470 | 0.005967 |

解释：all-risk 的 top-1 与 W4A16 reference 一致，但 logits cosine 仍有可测差距；
mixed 的 cosine 较高但 held-out top-1 有下降。按当前放宽后的 GO 规则，all-risk
是更稳妥的候选，但两者都还不是 cosine-parity GO。

最终 scale 产物：

- `artifacts/p1/streaming-full-mapzp/`
- `artifacts/p1/streaming-full-allrisk-mapzp/`

每个目录包含 28 个 layer 的 `scale1`/`scale2` safetensors 和
`streaming-train.json`。scale1 是 UInt4 carrier，scale2 是 per-output-channel
FP32；weight int4 code 固定不变。

## 部署状态与边界

- 公开 QNN LPBQ G32 contract、V79 finalize、HMX/LPBQ optrace 和真机 oracle
  equality 已完成，G32 W4A16 路线可部署。
- 当前源码仍没有 `kQNN_LPBQ_w4a8o8_G32` backend；不能只修改 JSON 把现有
  `QNN_LPBQ_w4a16o16_G32` 变成真实 W4A8。P0/P1 的 W4A8 结果目前是纯
  PyTorch fake-quant/software oracle，不是 QNN native W4A8 结果。
- 两个 prefix-streaming 一键 profiler 已添加：

  - `run_qwen3_sm8750_v79_prefix_streaming_mixed_profile.sh`
  - `run_qwen3_sm8750_v79_prefix_streaming_allrisk_profile.sh`

  它们复用 `run_qwen3_sm8750_v79_g32_profile.sh`，记录 scale 文件 hash、训练
  manifest、precision map、native context SHA 和离线指标；默认拒绝旧的
  actaware/selective context，避免误标为本次 prefix-streaming 方案。

## 回到 VM 后的下一步

1. 在 VM checkout 本分支，确认 QAIRT SDK 为
   `D:\llm_exp\models\qualcomm-sdk\qairt\2.47.0.260601` 对应的 Linux 环境。
2. 从原始 Qwen3 checkpoint 和上述 28 个 scale 文件生成两套完整 G32
   checkpoint；校验每层 codes SHA、scale1/scale2 shape 和 LPBQ decode 差分。
3. 用 `mllm-qwen3-aot-sha-g32-c` 生成两套 native V79 context 和 s1/s32
   schematics。context 必须使用方案独立文件名和 SHA，不能复用旧 baseline。
4. 在 VM 运行两个 wrapper；如果通过 WSL 调用 Windows ADB，设置
   `ADB_BIN=adb.exe`，结果写入 `D:\llm_exp\results`。
5. 只有在 QNN W4A8 backend 实现并完成单算子 contract gate 后，才把配置切到
   真正 W4A8；随后比较 W4A16、固定 Hadamard、prefix mixed、prefix all-risk、
   W4A8 mixed 的 held-out logits/top-1、E2E throughput、HMX/HVX optrace。
6. P3 的 W4A8 backend 需要新增物理 dtype/QuantSpec、A8 QDQ、mixed layer
   fallback 和 QNN AOT MIR；在此之前不要把当前 fake-quant 数字写成真机 W4A8
   精度结论。

## 验证记录

- static A8 单测：`pymllm/tests/test_static_a8.py`
- 训练/评估脚本均为逐层或逐 block streaming。
- shell profiler 已通过 `bash -n`；当前 VM 还需要生成方案对应的 native
  context/schematics 才能开始真实机 profiling。
