# W4A8O8 对称 KV 量化：主机 GPU 重量化交接方案

更新时间：2026-08-05
当前分支：`codex/quant-npu-w4a8-roadmap`
项目路径：`/home/daniuniu/llm_exp/mllm`

当前工作树包含此前的 QNN/AOT、profiling 脚本和实验产物修改，均未在本次
文档整理中覆盖或提交；开始 GPU 重训前请保留这些修改，并将新的 artifact
输出到独立目录。

## 1. 目的与当前决策

下一步在宿主机/WSL 的 RTX 5070 Ti 上重新训练 A8 scale，并生成一个能与
SM8750/V79 QNN 硬件口径对齐的候选 artifact。第一轮不做全模型“盲目改成
对称”，而采用选择性方案：

> **`W4A8O8_VSYM_G32`：固定 W4 LPBQ G32 code，只重新训练对称 A8 scale；先让
> V projection → V cache → attention×V 这条边界完全对齐。**

这样可以直接验证当前最明确的硬件问题，同时不破坏已经验证过的 Q/K/O/MLP
原生投影 lowering 和 mixed/all-risk A16 fallback。若 V-only 对称方案仍无法
达到 W4A16 速度，再转向 fused cache append 或 K-RoPE+quant，而不是立即扩大
到全模型对称 A8。

当前文档是交接与实施计划；本次尚未把全模型 symmetric recipe 写入 QNN AOT，
因此旧的 `mapzp` 训练产物不能直接当作最终的 symmetric artifact。

## 2. 当前工作状态

### 2.1 软件训练结果

已在 WSL RTX 5070 Ti 12 GB 上完成 28 层 prefix-aware streaming 训练。P1
训练使用两阶段 curriculum：

1. Stage 1：W4A16，仅训练 LPBQ G32 `scale1/scale2`，200 steps，`lr=0.003`；
2. Stage 2：联合训练静态 A8 scale/clipping 与 LPBQ scale，400 steps，`lr=0.005`。

两阶段均使用 30-step warmup、cosine decay、gradient clip 1.0，并在部署态
round-trip loss 上选择 scale。已有 source-aligned 全模型候选：

| 候选 | A16 fallback | held-out logits cosine | top-1 agreement | 说明 |
| --- | ---: | ---: | ---: | --- |
| W4A16 reference | 196/196 | 0.916866 | 1.000000 | 软件参考 |
| source-aligned prefix all-risk | 62/196 | 0.915216 | 1.000000 | 当前更稳妥的 mixed 候选 |
| source-aligned prefix mixed | 46/196 | 0.917246 | 0.666667 | 精度仍需谨慎 |
| source-aligned full W4A8 | 0/196 | 0.104340 | 0.000000 | negative control |

这些数字是 PyTorch fake-quant/software oracle，不能直接等价为 native QNN
W4A8 精度。当前 native QNN source-aligned 方案的设备 accuracy sanity 仍为
`0/100`，生成文本为乱码；速度和精度必须分开分析。

### 2.2 W4 G32 权重来源

第一轮 symmetric 重训必须保留已经核对过的 W4 code，不要重新从 BF16
teacher 生成 code：

```text
base checkpoint:
/mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors

base SHA-256:
6caa0d36ef6846ad2ab138335204699d9abd44c7dd30df25a5846a6b3e5e6bae
```

该 base 的 logical OI code 与 packed HWIO code 已对 196/196 projection
核对一致。新的训练首先只改变 A8 clipping/scale，以及必要时 LPBQ `scale1`
和 `scale2`；UInt4 code 初始阶段保持不变。

现有 source-aligned 目录（可作对照，不是新 symmetric 输出）：

```text
artifacts/p1/aligned-full-allrisk-base-g32-mapzp/
artifacts/p1/aligned-full-mixed-base-g32-mapzp/
```

它们使用 `fixed_zero_point=map`，zero-point 是固定整数但不是统一 128，不能
通过修改 JSON 直接伪装成 symmetric QNN artifact。

### 2.3 真机 native A8O8/KV 现状

已经打通并在 SM8750/V79 真机跑通了 direct-symmetric KV consumer 原型：

```text
context:
/tmp/qwen3-source-aligned-aot/kv-sym-direct-full-both/

context SHA-256:
7af1841a97918c9652d6c90ddcd76410f2e66d3c0031236c5ceafc53bb4f5c7d

profile result:
/tmp/qwen3_sm8750_v79_direct_results/qwen3_sm8750_v79_g32_20260805_005733/
```

基准（3 次运行中位数）如下。W4A16 reference 为跨日期结果，主要用于量级
参考；其余三项是同一套 source-aligned native 流程的对照。

| 方案 | Prefill token/s | Decode token/s | Graph S1 | Graph S32 |
| --- | ---: | ---: | ---: | ---: |
| W4A16 reference | 787.712 | 41.629 | 19.868 ms | 23.680 ms |
| source-aligned symmetric KV | 663.883 | 38.824 | 22.118 ms | 28.867 ms |
| affine V cache | 682.383 | 39.987 | 21.973 ms | 29.224 ms |
| **direct symmetric V boundary** | **691.578** | **39.968** | **21.881 ms** | **28.981 ms** |

direct 方案是当前 native A8 KV 变体中最快的：相对 affine V cache，prefill
约快 1.35%，decode 基本持平；相对 source-aligned symmetric KV，prefill 约快
4.17%，decode 约快 2.95%。但它仍比 W4A16 慢约 12.2% prefill、4.0% decode。

主要 stage：

| stage | S1 | S32 |
| --- | ---: | ---: |
| `kv_cache_update` | 13,987,242 cycles / 1.426 ms | 73,021,171 cycles / 6.181 ms |
| `attention_value` | 3,716,499 cycles / 0.680 ms | 4,385,680 cycles / 0.609 ms |

direct 方案消除了当前 V projection 到 cache 的一段 affine→symmetric 转换，
但 KV 路径仍有大量 `Dequantize/Quantize/Convert`。因此重新量化 V 可以去掉
边界转换，预计是增量优化；它不会自动消除 K 的 post-RoPE QDQ、cache append、
slice/concat 等公共开销。

## 3. 必须遵守的硬件量化合同

### 3.1 `zp=128` 不是可随意修改的约定

当前 KV QDQ 路径在
`examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp:151-171` 只接受
`kUInt8PerTensorSym`，并断言附带 zero-point 为 `128`，随后附加常量 128。
这是 signed int8 数值以 UInt8 存储时的编码约定：

```text
signed code: [-128, 127]
stored UInt8: signed code + 128
```

`mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/LLaMALinear.cpp`
中的 U8 输入/输出也明确执行减/加 128。故当前 native symmetric KV 的软件
合同是 **UInt8 storage + zp=128**，不是 affine map 的任意 zero-point。

QNN recipe 侧在
`mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp:80-104` 为
`kUInt8PerTensorSym` 创建 `SymPerTensor`；而 native projection 当前仍在约
`LLMQuantRecipePass.cpp:343-353` 生成 `AsymPerTensor(0,255,UInt8)`。
所以不能只把训练产物中的整数改成 128 就宣称已经是 native symmetric A8O8；
还需要同步修改 recipe、Conv2D visitor、QuantSpec 和 AOT MIR gate。

### 3.2 当前推荐的选择性 recipe

第一轮使用以下 contract：

| 路径 | 推荐 dtype/合同 |
| --- | --- |
| W4 权重 | LPBQ G32，固定 UInt4 code，硬件 tile/HWIO packed layout |
| Q/K/O、MLP、LM head native projection | 保留当前已经验证的 affine `UInt8PerTensorAsy` |
| V projection output | 改为 `UInt8PerTensorSym`，`zp=128` |
| V cache | `UInt8PerTensorSym`，直接复用 V projection scale |
| K cache | post-RoPE `UInt8PerTensorSym`，`zp=128`；K 仍保留必要的 A16/RoPE 边界 |
| attention query/非线性/RMSNorm/RoPE/SiLU/softmax/residual | 保持现有 A16 |
| 敏感层 | 继续使用 W4A16/O16 fallback（prefix mixed/all-risk） |

关键约束是：

```text
S_v_projection == S_v_cache == S_v_attention_input
```

V 当前 block 只量化一次，不再通过 `.to(kUInt8PerTensorSym)` 做第二次独立
requant。K 暂时接受一处量化边界，避免把尚未证明的 K-RoPE fusion 混入第一轮。

### 3.3 为什么不先全模型改 symmetric

现在只有 attention×V 的 symmetric-U8 kernel 路径得到过真机正向验证。Q/K/O/
MLP/LM-head 全部切到 Sym U8 需要重新验证 HMX kernel 分类、tile layout、
QNN graph finalize 和端到端速度；如果某些 native projection 退回 HVX 或增加
QDQ，可能比当前 affine 路径更慢。故先做 V-only 对齐，得到明确增益后再扩大范围。

## 4. 主机 GPU 重量化方法

### 4.1 对称 fake-quant 定义

新 trainer 应显式使用 `sym128` 模式，而不是仅传旧的
`--fixed-zero-point 128`。建议定义为：

```text
q_signed = clamp(round(x / scale), -128, 127)
q_uint8  = q_signed + 128
x_hat    = (q_uint8 - 128) * scale
scale    = alpha / 127
```

其中 `alpha` 是每个 activation tensor 的可学习 clipping 半径；`scale > 0`
并由训练约束保持。对于 V projection，训练和导出时强制三者使用同一个
scale identity：projection output、cache、attention input。

建议新增清晰的 recipe/manifest 标记，例如：

```text
a8_recipe: sym128_vsym
storage_dtype: UInt8
zero_point: 128
kv_dtype: UInt8PerTensorSym
v_scale_tied: true
```

旧脚本的 `--fixed-zero-point 128` 目前只能固定整数 zero-point，不能单独完成
上述 symmetric clipping、V scale tying 或 QNN native dtype lowering；在 trainer
模式补齐前，不要把它当成可部署的最终命令。

### 4.2 训练 curriculum

沿用已经验证过的 prefix-aware streaming 结构，避免 28 层 autograd 图进入 12 GB
显存：

1. 固定 W4 code，载入 base 的 LPBQ `scale1/scale2`；
2. Stage 1 先在 W4A16 条件下稳定 LPBQ scale；
3. Stage 2 打开 symmetric A8 clipping/scale，联合微调 LPBQ scale；
4. 每隔若干 step 使用真实 UInt4 carrier 做 deployment round-trip loss；
5. 当前 block 完成后立刻导出并冻结，最后用 held-out calibration 做全模型 no-grad。

第一轮建议仍采用 200/400 steps、`0.003/0.005` learning rate、30-step warmup、
cosine decay、gradient clip 1.0，便于与 P1 软件 oracle 对照。不要在第一轮同时
重新学习 rotation 或重新搜索 W4 code，否则无法判断收益来自 scale 还是 code。

### 4.3 期望的主机 GPU 命令接口

以下是补齐 `sym128_vsym` trainer mode 后的目标命令。当前代码尚未保证该参数
已经实现，运行前需要先完成第 5 节的 trainer/manifest 改动：

```bash
python scripts/qwen3_p1_streaming_train.py \
  --model /home/daniuniu/llm_exp/models/Qwen3-origin \
  --base-quant-checkpoint /mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors \
  --calibration-manifest /home/daniuniu/llm_exp/calibration/qwen3-p0-all-layers-seed17-s96/manifest.json \
  --sensitivity-map artifacts/p0/static_a8/mixed-precision-map.json \
  --a8-recipe sym128_vsym \
  --fixed-zero-point 128 \
  --stage1-steps 200 --steps 400 \
  --stage1-lr 0.003 --lr 0.005 \
  --warmup-steps 30 --deployment-eval-every 50 \
  --selection-split held_out \
  --target-mode teacher \
  --device cuda \
  --output-dir artifacts/p1/sym128-vsym-mixed/
```

若第一轮要优先保证质量，先用 all-risk map 跑通，再跑 mixed map：

```text
artifacts/p1/sym128-vsym-allrisk/
artifacts/p1/sym128-vsym-mixed/
```

训练完成后，使用现有 merge 工具把 learned `scale1/scale2` 合并回固定-code
base；不要用 BF16 teacher 直接替换权重：

```bash
python scripts/merge_qwen3_lpbq_scales_from_base.py \
  --base-checkpoint /mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors \
  --scale-dir artifacts/p1/sym128-vsym-mixed \
  --training-manifest artifacts/p1/sym128-vsym-mixed/streaming-train.json \
  --output artifacts/p1/sym128-vsym-mixed/qwen3_w4a8o8_vsym_g32.safetensors
```

all-risk 只需把 `--scale-dir` 和 manifest 换成 all-risk 目录。合并后必须重新
生成 packed HWIO/G32 artifact，不得沿用旧 context 的 SHA。

## 5. 重新量化前需要补齐的代码工作

### P0：trainer 与 artifact contract

- 在 `scripts/qwen3_p1_streaming_train.py` 增加显式 `sym128_vsym` recipe；
- fake-quant 使用对称 clipping，而不是沿用 affine map 的统计公式；
- manifest 为每个 A8 tensor 记录 `dtype=UInt8`、`zero_point=128`、scale 和
  clipping 半径；
- V projection/cache/attention 三个消费者记录同一个 scale identity；
- 训练后检查所有 V scale 是否一致、是否正数、是否发生 silent affine fallback。

### P1：离线 gate

对 mixed 和 all-risk 各跑一遍：

1. 196/196 projection 的 logical OI 与 packed HWIO code hash 必须和 G32 base
   一致；
2. `zp=128`、symmetric clipping、scale round-trip 全部通过；
3. V projection/cache/attention scale identity 通过；
4. held-out block NMSE、logits cosine、top-1 与 W4A16、旧 source-aligned
   `mapzp` 对照；
5. 先做 no-grad 全模型评估，再进入 VM/QNN AOT。

建议将输出命名为：

```text
artifacts/p1/sym128-vsym-{mixed,allrisk}/
  streaming-train.json
  qwen3_w4a8o8_vsym_g32.safetensors
  layerNN-lpbq-scales.safetensors
```

## 6. VM/QNN-AOT 与真机 gate

GPU 训练通过后才回 VM 做部署验证：

1. 为 V-only symmetric recipe 生成独立 QNN context、S1/S32 MIR 与 schematic；
2. MIR 中 native V Conv2D output、V cache、Concat/past、attention×V V input
   必须都是 `SymPerTensor UInt8`；
3. native V 路径不应再出现 `.to(kUInt8PerTensorSym)` 的二次转换；
4. 用 `run_qwen3_sm8750_v79_source_aligned_mixed_kv_sym_direct_profile.sh`
   的 direct 流程做速度对照；
5. optrace 重点检查 HMX/HVX kernel、`kv_cache_update`、attention×V、
   `Dequantize/Quantize/Convert` 数量；
6. accuracy sanity 单独执行，不能用“速度变快”替代精度 gate。

如果 V-only 对称方案只带来小幅改善，下一步优先级为：

```text
fused cache append
  → K post-RoPE quant / cache update fusion
  → 减少 history slice/concat 与重复 QDQ
  → 再评估全模型 Sym U8 projection
```

## 7. 当前不要做的事情

- 不要把 `aligned-full-*-mapzp` 直接改 JSON 成 `zp=128`；这会造成软件/硬件
  contract 不一致；
- 不要把当前 native device `0/100` accuracy 解释成 GPU fake-quant 结果已经失效，
  先分别定位 QNN graph 输出/格式问题；
- 不要第一轮同时改变 W4 code、rotation、A8 scale 和 KV dtype；
- 不要把 `Unclassified` stage 无条件写成 LM head，只能标注为“很可能包含 LM
  head shard，需要结合 optrace 确认”；
- 不要在尚未完成单算子 Sym U8 HMX contract 前，把全模型 native symmetric
  projection 标记为可部署。

## 8. 交接结论

主机 GPU 下一步的最小闭环是：

```text
固定 G32 W4 code
  → 新增 sym128_vsym fake-quant/scale tying
  → 28 层 streaming GPU 训练（all-risk，再 mixed）
  → code/hash + zp/scale + held-out oracle gate
  → 合并生成独立 artifact
  → VM 中做 V-only Sym U8 QNN/AOT
  → SM8750/V79 direct profiling 与 accuracy 分离验证
```

这条路线既能利用当前已经验证的 G32/混合精度成果，也能把最需要硬件对齐的
KV 边界单独拿出来验证。最终目标仍是 W4A8O8；W4A16 只是性能和精度基线，
不是终点。
