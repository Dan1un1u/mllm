# Quant.npu 方法在 mllm 上的下一步路线

## 1. 文档目的

本文以 `/home/daniuniu/Desktop/Quant.npu.pdf` 为主要参考，结合当前 mllm 的 QNN HTP/HMX 实现、W4A16G32 全模型产物、真实设备 profiling，以及已有的 learned-R1/A8 scale 实验，确定后续向 W4A8 推进的优先级。

本文不是照搬论文 recipe，而是保留当前已经验证的 G32 LPBQ 权重 contract，吸收 Quant.npu 的静态量化优化方法。

## 2. 当前基线和已知事实

当前全模型基线为：

```text
Linear/lm_head weight : W4 LPBQ G32
主干 activation       : 静态 A16/O16 QDQ
KV cache              : W8A8 per-tensor symmetric
SiLU/RMSNorm/softmax  : 保持现有 A16/QDQ 路径
rotation              : none
backend               : QNN HTP standard LPBQ lowering + HMX
```

相关实现：

- `scripts/export_qwen3_lpbq_g32.py`
- `examples/qwen3_qnn_aot/config_1.7B_g32.json`
- `examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json`
- `examples/qwen3_qnn_aot/modeling_qwen_qnn_aot.hpp`

当前 G32 权重不能直接通过改配置变成 W4A8。`mllm/core/aops/Conv2DOp.hpp` 目前只注册了 `kQNN_LPBQ_w4a16o16_G16/G32/G64`，还没有 `kQNN_LPBQ_w4a8o8_G32`。现有 `kQNN_tensor_symm_w8a8` 是 W8A8，不是 W4A8。

已有实验给出的方向判断：

1. 固定 H64 优于当前 learned shared-R1；继续扩大 learned-R1 暂无充分收益。
2. 固定 H64 下学习静态 A8 scale 后，W4A8 block loss 约改善 10.7%，说明当前瓶颈更接近 activation clipping/scale，而非 R1 表达能力。
3. 动态 O8 会带来明显量化/转换开销，不能作为默认部署路径。
4. 现有 profiling 的主要耗时仍包括 LPBQ weight expansion/dequant、weight DMA 和 HMX MAC，因此 A8 不能保证自动获得论文中的速度收益。

## 3. Quant.npu 方法的迁移判断

| 方法 | 迁移判断 | 处理方式 |
|---|---|---|
| 全静态 activation quantization | 直接适用 | 保持 QNN compile-time scale/zero-point；禁止运行时 min/max |
| learnable activation scale | 高优先级 | 先只学习 Linear input scale/clipping，zero-point 初版固定 |
| learnable weight scale | 可选 | 在固定 G32 code 下学习 LPBQ scale；避免一开始联合学习所有参数 |
| mean/3-sigma、Max-Min 初始化 | 直接适用但需分布感知 | 旋转后测试 Mean/3-sigma，未旋转/重 outlier 测试 Max-Min 和 percentile |
| local quantization error loss | 高优先级 | 与 block-output distillation 联合，优先作用于 activation fakequant |
| gradient scaling | 高优先级 | 抑制共享 scale 的梯度随 tensor size 放大 |
| distribution-aware two-stage optimization | 高优先级 | Stage One 优化输入 activation/权重；Stage Two 静态校准其余 tensor |
| down_proj sensitivity mixed precision | 高优先级 | 采用 layer-level 静态 A8/A16 map，先提升少量敏感层 |
| R1/R2 offline fusion | 有条件适用 | 先做 fixed H64/R2-only 对照；必须在 BF16 权重上融合后重新 G32 量化和全模型 calibration |
| learned shared-R1 | 暂缓 | 当前三层 held-out 实验未优于 fixed H64，不作为第一主线 |
| R3/R4 online rotation | 不适用 | 引入在线 FP 矩阵乘法，违反纯 HTP 和低 overhead 目标 |
| 论文 per-channel W4 | 不直接适用 | 保留 QNN G32 LPBQ；只有在 QAIRT 提供 native W4A8 per-channel kernel 时单独 benchmark |
| 动态 per-token/per-vector quantization | 不适用 | 运行时 reduction、HVX conversion 和 cache 管理开销过高 |
| 论文 16 个 calibration sample | 不照搬 | 使用当前全模型、多 split、held-out calibration；论文自身也承认 calibration data 会显著影响结果 |

## 4. P0：静态 A8 scale 原型

### 4.1 目标

在不引入 learned rotation 的前提下，验证静态 A8 scale 是否可以显著降低 W4A8 精度损失。保持 G32 LPBQ 权重不变，先把量化训练问题和硬件 kernel 问题分开。

### 4.2 训练对象

Stage One 只优化：

- Linear 输入 activation 的静态 A8 scale；
- 可选的 LPBQ `scale1/scale2` 或其连续代理；
- block-output distillation 所需的 fakequant 参数。

Stage One 暂时冻结：

- Linear output activation；
- KV cache scale；
- SiLU、RMSNorm、RoPE、softmax 等非线性/中间 tensor；
- lm_head 的输出 quantizer。

这些冻结 tensor 在 Stage Two 重新做静态 calibration。非线性保持当前 A16/QDQ，不强行改为 FP16，也不强行改为 A8。

### 4.3 初始化矩阵

至少比较以下初始化：

1. Max-Min；
2. mean/3-sigma；
3. 99.9% 或 99.99% percentile clipping；
4. learnable clipping threshold 的 warm start。

当前 mllm 的 A16 主干是 asymmetric QDQ，而论文主要采用 symmetric per-tensor activation。因此不能直接复制论文公式，必须转换为 mllm 的 scale/zero-point 表示。

第一版建议固定 zero-point，只学习 scale/clipping；待 A8 硬件 contract 明确后，再评估 zero-point 是否需要学习。

### 4.4 损失函数

```text
L = λlocal * local_activation_reconstruction
  + λblock * block_output_distillation
  + λlogit * held_out_logit_distillation
```

训练早期优先使用 local activation loss，随后提高 block-output 和 logits loss。实现论文中的 gradient scaling，避免量化 scale 的梯度压过 block/R1 参数。

### 4.5 内存策略

当前 12 GB GPU 不适合一次性对 28 层建立完整 autograd 图。采用逐层/逐 block streaming：

```text
BF16 reference block output
    → 当前层 fakequant/scale optimization
    → 保存 scale/zero-point
    → no_grad 进入下一层
```

已有单层 learned-R1 约 1.11 GiB reserved 的结果说明单层训练可行，但不代表全模型联合训练可行。

## 5. P1：敏感度驱动的静态 mixed precision

### 5.1 目标

解决无法完全 A8 的 outlier tensor，优先验证论文中 down_proj 的观察是否同样适用于 Qwen3/mllm。

### 5.2 敏感度统计

对 28 层完整收集以下 tensor：

- q/k/v/o input；
- gate/up input；
- down_proj input；
- SiLU output；
- down_proj output。

计算：

- relative quantization error；
- NMSE；
- clipping ratio；
- 99.9% error；
- block-output error；
- held-out logits impact。

论文的 relative error 可作为起点，但不能单独使用，因为接近 0 的 activation 会放大比值。最终排序应结合 NMSE 和 block-output impact。

### 5.3 静态 bit-width map

先只允许整层/整 tensor 提升：

```text
普通层：down_proj input = A8
敏感层：down_proj input = A16
SiLU/residual/softmax：保持 A16
KV cache：保持 W8A8
```

测试比例：0%、10%、20%、50%、100% 的 down_proj input A16。不要第一版做逐 token 或逐 channel 动态混合，因为会引入运行时分支、layout 转换和额外 HVX 处理。

## 6. P2：旋转只做离线对照

按以下顺序实验：

```text
No rotation + learned static scale
Fixed H64 + learned static scale
R2-only + learned static scale
Fixed H64 + R2 + learned static scale
最后才考虑 learned R1
```

旋转必须作用于 BF16 原始权重：

```text
BF16 weight
  → offline R1/R2 fusion
  → fresh G32 LPBQ quantization
  → fresh activation calibration
  → QNN/AOT export
```

不能对已导出的 INT4 code 直接乘旋转矩阵。

需要额外检查：

- attention/MLP residual 的 basis 是否一致；
- q/k/v/o 和 gate/up/down 的旋转方向；
- lm_head 与 tied embedding 的一致性；
- SHA 拆 head 后 scale1/scale2 是否仍按输出通道正确切片。

R3/R4 不进入主线，因为它们需要在线 FP rotation，与当前纯 HTP 目标冲突。

## 7. P3：QNN W4A8 物理 contract

只有 P0/P1 的软件精度结果达到预期后，才启动硬件 W4A8 backend。

### 7.1 先确认 QAIRT 能力

确认以下内容是否被当前 SDK/设备支持：

- W4 LPBQ + A8 input/output；
- G32 block size；
- static per-tensor A8 scale；
- mixed A8/A16 graph；
- lm_head 的 W4A8；
- HMX physical path 是否生效。

### 7.2 mllm 需要新增的内容

如果 QAIRT 支持，需要新增：

- `Conv2DOpImplType::kQNN_LPBQ_w4a8o8_G32`；
- 对应 Linear/Conv2D mapping；
- A8 QDQ dtype 和 zero-point 处理；
- G32 W4A8 MIR/QuantSpec；
- AOT config 和 mixed-precision dtype map；
- profiling 分类，区分 A8/A16 cast 和真正的 HMX MAC。

### 7.3 单算子 gate

先验证：

```text
2048 → 2048
2048 → 6144
6144 → 2048
```

验收条件：

- graph finalize 成功；
- physical input/output 为 A8 或明确的 QNN A8 physical lowering；
- HMX 被使用；
- 没有隐式退回 FP16/A16；
- A8/A16 conversion 不吞掉理论收益；
- 与 G32 W4A16 单算子输出误差符合 fakequant 预估。

## 8. P4：全模型真机验证

最终至少比较以下版本：

1. 原始 G16 W4A16；
2. 当前 G32 W4A16；
3. G32 W4A8，无 mixed precision；
4. G32 W4A8 + down_proj layer-level A16；
5. G32 W4A8 + fixed H64/R2（如果软件 gate 通过）。

每个版本固定相同：

- prompt 数据；
- prefill/decode 长度；
- batch；
- HTP performance mode；
- 温度和 profile 开关。

同时记录：

- prefill/decode latency；
- tokens/s；
- peak memory；
- energy（如果可用）；
- LPBQ expansion/dequant；
- weight DMA/wait；
- HMX MAC；
- A8/A16 cast；
- end-to-end task accuracy；
- held-out PPL/logit divergence。

## 9. 主要风险和替代方案

### 风险 1：A8 activation accuracy 仍然不足

优先使用：

```text
learned static clipping
down_proj layer-level A16
SiLU/residual A16
fixed H64/R2 offline rotation
```

不优先使用动态 quant 或 outlier CPU side path。

### 风险 2：A8 速度没有超过 A16

原因可能不是 MAC，而是：

- LPBQ expansion 仍占主要时间；
- A8/A16 cast 过多；
- QNN 没有真正选择 W4A8 HMX kernel；
- mixed-precision graph 破坏了 fusion。

此时应先减少 cast、检查 physical dtype 和 graph fusion，而不是继续优化量化算法。

### 风险 3：G32 比 per-channel W4 精度差

不要直接放弃已经验证的 G32 contract。可单独做一个 QAIRT native per-channel W4A8 benchmark；如果没有硬件 kernel，则继续使用 G32，并只对敏感层做 A16 fallback 或更细粒度 G16 混合。

### 风险 4：SmoothQuant 不能吸收 down_proj outlier

由于 `SiLU(gate) × up → down_proj` 中间存在非线性和逐元素乘法，SmoothQuant 的 scale 不能像普通 Linear 链那样完整折叠。它可作为对照实验，但主线采用 learnable clipping + static mixed precision。

## 10. 推荐执行顺序

```text
P0. 全模型静态 A8 scale/clip 学习（无 rotation）
  ↓
P1. down_proj sensitivity map + layer-level A16 fallback
  ↓
P2. fixed H64/R2 离线旋转对照
  ↓
P3. QAIRT/QNN W4A8G32 单算子物理 gate
  ↓
P4. 全模型 W4A8 mixed-precision AOT + 真机 profiling
```

当前最重要的判断标准不是“能否复现论文的 R1”，而是：

```text
在保持 G32 LPBQ 权重和 A16 非线性 fallback 的前提下，
静态 A8 activation + 少量 A16 sensitive layer，
能否同时满足精度和端到端延迟目标。
```
