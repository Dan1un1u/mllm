# 现有 W4A16G32 量化 Recipe

现有的 `W4A16G32` 是一套“标准 QNN LPBQ + 量化 A16 激活”的全模型方案，准确地说是：

> 逻辑权重 W4、逻辑激活 A16、逻辑输出 O16、Linear 权重按 K 维 G32 分组；KV cache 例外使用 W8A8；运行时走 QNN HTP 的 LPBQ block expansion + HMX MAC。

它目前不是 FP16 路径，也不是论文中 Direct Hexagon/VLUT16 的自定义 LUT 路径。

## 1. 量化对象和覆盖范围

模型是 Qwen3-1.7B，共 28 层。

每层有 7 个 Linear：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

因此：

```text
28 × 7 + lm_head = 197 个矩阵
```

当前导出器会重新量化这 197 个权重，具体实现位于 [`export_qwen3_lpbq_g32.py`](/home/daniuniu/llm_exp/mllm/scripts/export_qwen3_lpbq_g32.py:119)。

非 Linear 参数，例如 embedding、RMSNorm 权重、RoPE 相关参数和其它常量，则直接从原来的量化 checkpoint 中保留，不在 G32 导出阶段重新处理。

虽然 Qwen3 配置里 `tie_word_embeddings=true`，当前导出器仍然会显式把 `lm_head.weight` 作为第 197 个矩阵进行 G32 LPBQ 量化；embedding 输入权重本身不走这套 Linear 导出流程。

## 2. W4 的具体含义

当前权重量化是有符号对称 INT4：

```text
q ∈ [-7, 7]
```

对每一个输出通道 `o`，沿输入通道 `K` 每 32 个元素分为一组：

```text
W[o, K] → W[o, K/32, 32]
```

对于每一组，首先计算普通的 per-group scale：

\[
s_{o,g} = \frac{\max_{k \in g}|W_{o,g,k}|}{7}
\]

然后得到 INT4 code：

\[
q_{o,g,k} = \operatorname{clip}\left(\operatorname{round}\left(\frac{W_{o,g,k}}{s_{o,g}}\right),-7,7\right)
\]

最终的近似权重是：

\[
\hat W_{o,g,k}=q_{o,g,k}\cdot s_{o,g}
\]

实际代码使用 torchao 的 `_quantize_affine`，并设置：

```text
quant_min = -7
quant_max = 7
zero_point = 0
```

## 3. LPBQ 的两级 scale

当前不是简单的每组 FP32 scale，而是 LPBQ 的两级 scale。

首先，对于每个输出通道，取所有 G32 group scale 的最大值：

\[
s^{(2)}_o = \frac{\max_g s_{o,g}}{16}
\]

然后将每一个 group scale 表示为一个较小的 level-1 scale：

\[
s^{(1)}_{o,g}=\operatorname{round}\left(\frac{s_{o,g}}{s^{(2)}_o}\right)
\]

并限制在：

```text
s1 ∈ [1, 16]
```

最终实际使用的 group scale 是：

\[
s_{o,g}\approx s^{(1)}_{o,g}\cdot s^{(2)}_o
\]

因此最终重建公式是：

\[
\hat W_{o,g,k}=q_{o,g,k}\cdot s^{(1)}_{o,g}\cdot s^{(2)}_o
\]

这种设计的好处是：

- `scale1` 只需要很小的整数表示；
- `scale2` 每个输出通道只有一个 FP32；
- 比每个 G32 group 都保存 FP32 scale 更节省 metadata；
- 符合 QNN LPBQ 的硬件接口。

这里要区分“逻辑类型”和“文件 carrier”：

- 逻辑上：权重是 Int4，`scale1` 是 UInt4；
- 当前 safetensors 中：每个 INT4 code 使用一个 INT8 carrier byte 保存；
- `scale1` 也以 UInt8 carrier 保存；
- QNN 的 QuantSpec 再把它解释为 LPBQ 的 Int4/UInt4 语义。

当前 safetensors 并不是简单地把两个 INT4 压进一个 byte。QNN 在后续转换和 AOT 阶段负责解释和处理逻辑 Int4。

## 4. 权重布局

原始 Linear 权重通常是：

```text
[O, K]
```

但 mllm 使用 Conv2D 来替代 Linear，因此导出后权重转换为 HWIO：

```text
[1, 1, K, O]
```

三个相关 tensor 的布局是：

| Tensor | 逻辑内容 | 当前布局 |
|---|---|---|
| `weight` | signed INT4 code | `[1,1,K,O]` |
| `scale1` | 每个 `(O,G32)` 一个 level-1 scale | flatten 后的 `[O,K/32]` |
| `scale2` | 每个输出通道一个 level-2 scale | `[O]` |

`scale1` 必须按照输出通道优先、group 次序排列：

```text
[o=0, g=0...G-1],
[o=1, g=0...G-1],
...
```

## 5. 不同矩阵的 G32 规模

| 矩阵 | 权重形状 `[O,K]` | 每个输出通道的 group 数 | `scale1` 元素数 |
|---|---:|---:|---:|
| q/o_proj | `[2048,2048]` | 64 | 131,072 |
| k/v_proj | `[1024,2048]` | 64 | 65,536 |
| gate/up_proj | `[6144,2048]` | 64 | 393,216 |
| down_proj | `[2048,6144]` | 192 | 393,216 |
| lm_head | `[151936,2048]` | 64 | 9,723,904 |

相比 G16，G32 会让沿 K 维的 group 数减半，因此减少 `scale1` 数量和 LPBQ group 管理开销，但每个 group 覆盖的权重更多，量化粒度更粗。

## 6. A16 的真实含义

当前 `A16` 不是 FP16。

在 mllm 图中，主干激活使用：

```cpp
kUInt16PerTensorAsy
```

也就是：

- 16-bit fixed-point activation；
- per-tensor scale；
- asymmetric zero-point；
- 通过 QDQ scale/zero-point 进行量化。

因此：

```text
A16 = 16-bit quantized activation
```

而不是：

```text
A16 = IEEE FP16
```

当前模型配置使用：

```json
"linear_impl_type": "QNN_LPBQ_w4a16o16_G32"
```

这里的 `o16` 表示 Linear 的逻辑输出也是 16-bit quantized output。

在底层 HTP lowering 中，逻辑 UInt16 不一定始终以物理 UInt16 形式执行。QNN 可能把不同算子的输入、输出转换成物理 QInt8、QUInt8 或 QUInt16，再连接到 HMX。但从 mllm/QNN 图的逻辑契约看，主干是 A16/O16。

## 7. MLP 路径

MLP 中的三个 Linear 都使用 G32：

```text
gate_proj: W4A16G32
up_proj:   W4A16G32
down_proj: W4A16G32
```

MLP 计算结构仍然是：

\[
\operatorname{MLP}(x)=
\operatorname{down}\left(
\operatorname{SiLU}(\operatorname{gate}(x))
\odot \operatorname{up}(x)
\right)
\]

代码中 SiLU 展开为：

\[
\operatorname{SiLU}(x)=x\cdot\operatorname{sigmoid}(x)
\]

gate/up 输出、sigmoid 输出、SiLU 输出以及 down 输入都通过 QDQ。非线性部分并不是自动退回 FP16，而是尽量保持原有 mllm 的 A16/QDQ 路径。

## 8. Attention 路径

Attention 中以下四个投影全部使用 G32：

```text
q_proj: W4A16G32
k_proj: W4A16G32
v_proj: W4A16G32
o_proj: W4A16G32
```

主要流程是：

```text
hidden_states
  → q/k/v G32 Linear
  → Q/K RMSNorm
  → RoPE
  → KV cache 转为 UInt8 symmetric
  → QK matmul
  → scaling/mask/softmax
  → attention-value matmul
  → o_proj G32
```

## 9. KV cache 是例外的 W8A8

KV cache 使用 per-tensor symmetric UInt8：

```text
key:   W8A8 per-tensor symmetric
value: W8A8 per-tensor symmetric
```

运行时使用 `zero_point=128`，写入 KV cache 后，后续 attention 从 UInt8 cache 读取。这样做是为了降低随序列增长的 cache 存储和带宽开销。

## 10. lm_head 也已经是 G32 W4A16

当前 lm_head 不再作为 FP16 特殊路径处理，而是：

```text
lm_head weight: W4 LPBQ G32
lm_head input:  A16 QDQ
lm_head output: A16 QDQ
```

在 AOT 配置中，`lm_head` 被列为 `op_on_qnn`，主干模型放在 `graph_on_qnn` 中。这是编译和执行边界，不代表 lm_head 的量化格式不同。

## 11. SHA 版本如何处理 G32

SHA 版本会把：

- q_proj 拆成 16 个 head；
- k_proj 拆成 8 个 KV head；
- v_proj 拆成 8 个 KV head。

但它不会重新量化权重，只是沿输出通道切片：

```text
weight: [1,1,K,O] → [1,1,K,O_head]
scale1: 按输出通道连续区间切片
scale2: 按输出通道连续区间切片
```

这要求 `scale1` 使用 `[output_channel][group]` 的 row-major 排列。G32 的分组轴仍然是原始输入通道 K，SHA 只改变输出通道分片，不改变 G32 group 边界。

## 12. QNN/HMX 的实际执行方式

编译阶段通过宏选择：

```cpp
kQNN_LPBQ_w4a16o16_G32
```

在 QNN HTP 上，逻辑算子大致被降低成：

```text
INT4 LPBQ weight
    ↓
HVX block expansion / dequant
    ↓
per-channel/int8-like physical weight
    ↓
HMX MAC
    ↓
quantized output
```

当前 profiling 可以看到 LPBQ 展开、权重搬运和 HMX MAC 等阶段。

但当前实现没有包含：

- 论文中的 VLUT16 解量化；
- 自定义 Direct Hexagon kernel；
- 显式 32×32 tile reorder；
- LUT 与自定义 HMX GEMM 的融合。

因此它是“物理 group size 与 HMX/QNN 接口契合的 G32”，还不是论文中的完整硬件感知 LUT 方案。

## 13. 当前 G32 和 G16 的核心差异

两者主干激活路径基本相同，主要区别是 Linear 权重的 group size：

```text
G16: 每 16 个输入通道一组
G32: 每 32 个输入通道一组
```

G32 带来的变化是：

- 权重 code 数量不变；
- 每个权重仍然是逻辑 INT4；
- `scale1` 数量约减半；
- scale metadata 和部分 LPBQ 管理开销降低；
- 量化粒度变粗，理论误差可能增加；
- HTP/QNN 的实际速度取决于 block expansion 和数据搬运是否受益。

例如 q_proj：

```text
G16: 2048 × 128 = 262,144 个 scale1
G32: 2048 × 64  = 131,072 个 scale1
```

## 14. 当前方案明确没有包含的内容

当前 `W4A16G32` 是无旋转方案：

```text
rotation = none
```

没有使用：

- Hadamard H64；
- learned R1；
- SmoothQuant；
- activation clipping；
- per-token O8；
- W4A8；
- 自定义 VLUT16；
- 论文中的 32×32 tile-aware quantization layout。

所以当前方案的定位是：

```text
权重侧：完整的物理 G32 LPBQ
激活侧：标准量化 A16/O16
KV cache：单独使用 W8A8
硬件侧：标准 QNN HTP/HMX，不是自定义 LUT kernel
研究价值：作为未来 W4A8 的权重和布局基线
```

向 W4A8 继续推进时，权重侧的 G32 LPBQ contract 可以复用，但必须重新处理主干 activation QDQ、Linear 输出 dtype、非线性算子 O8 可行性、KV cache contract，以及全模型 calibration 和 AOT/MIR 配置。
