# QAIRT 2.49 原生 U8 Softmax 隔离实验

## 结论

QAIRT 2.49 在相同的非对称 U8 图、shape 和 qparam 下，仍选择
`q::MaskedSoftmax_Crouton_Scratch`。该 kernel 继续完全在 VTCM 内完成，
没有 DRAM 读写；2.49 降低了它的 work/dominant-path cycles，但没有带来
可测的 Softmax micrograph profiling-off 延迟改善。放回完整模型后，prefill
吞吐反而比已验收的 QAIRT 2.47 RMSNorm-A8/P19 基线低 1.76%。因此，本实验
不支持“仅升级 QAIRT 即可解决 A8 Softmax 退化”的假设。

QAIRT 2.49 的全模 decode 吞吐提高了 8.94%，但 decode 几乎不受 Softmax
影响。Optrace 表明这一收益主要来自 LPBQ 权重展开、weights-to-VTCM 和其它
全图调度变化，不能记为 Softmax 优化收益。

## 单变量合同

- 基线源码：`codex/w4a8g32-native-u8-rmsnorm`，提交 `987a7156`。
- 目标：SM8750/V79；激活保持 asymmetric U8；不改为 S8。
- 只使用 `qti.aisw::Softmax`，不启用历史 custom Softmax。
- QAIRT 2.47 保持在 `2.47.0.260601`；2.49 解压到独立的
  `/mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.49.0.260730`。
- host compiler、Android runner 和运行时库按 SDK 分别构建、加载，不混用
  2.47/2.49 的 QNN API ABI。
- micrograph 复刻 layer 14 的 16-head masked-Softmax 片段：
  `ReduceMin -> Add(-20) -> Equal(mask, 0) -> Where -> Softmax`。
- shape 分别为 `[1,1,1,1024]` 和 `[1,1,32,1024]`；所有输入、Softmax
  边界和输出均沿用已验收 manifest 的 asymmetric-U8 qparam。
- P 点对两个 SDK、两个 shape 独立搜索：
  `default,0,1,2,3,4,5,6,8,13,15,16,17,19,20,21,22,23`。
- stage 1 每例 20 次 warmup + 300 次测量；各组 top-3 再做 7 个新进程，
  每进程 50 次 warmup + 1000 次测量。

QAIRT 2.49 release notes 的 2.48/2.49 小节没有声明 HTP U8 Softmax 性能修复。
实验仍观察到相同 kernel 名称下的物理 cycles 变化，说明未公开的实现或调度
调整确实存在，但 `_Scratch` 名称本身不能作为 fallback/劣质实现的证据。

## Micrograph 结果

| SDK / shape | 最优 P | profiling-off 中位数 | work cycles | dominant cycles | DRAM R/W | VTCM R/W |
|---|---:|---:|---:|---:|---:|---:|
| 2.47 / s1 | 1 | 123 us | 110,879 | 19,256 | 0 / 0 | 2,129,920 / 1,310,720 |
| 2.49 / s1 | 6 | 123 us | 91,765 (-17.24%) | 15,014 (-22.03%) | 0 / 0 | 2,129,920 / 1,310,720 |
| 2.47 / s32 | 19 | 163 us | 193,012 | 33,966 | 0 / 0 | 8,519,680 / 4,456,448 |
| 2.49 / s32 | 19 | 163 us | 177,013 (-8.29%) | 31,253 (-7.99%) | 0 / 0 | 8,519,680 / 4,456,448 |

四个 winner 均为 16 个 `q::MaskedSoftmax_Crouton_Scratch`。2.49 没有改变
VTCM 流量，也没有引入 DRAM 往返。单次 Optrace graph timeline 在 s1 下降
3.65%，在 s32 反而上升 1.07%；这与稳定的 profiling-off 结果“123/163 us
不变”一致，说明 kernel 内部节省被图边界或调度开销抵消。

两个 SDK 的输出逐字节相同：s1 共 16,384 bytes、s32 共 524,288 bytes，
两者的 equal fraction 都是 1.0。重复运行也逐字节一致。host 浮点模拟仅作为
sanity，不设置精度门槛。

## 完整模型复核

完整图必须使用与基线相同的 `mllm-qwen3-aot-sha-g32-c`。通用
`mllm-qwen3-aot-c` 不是同一模型表达，会保留 RoPE 前的 U16 张量，不能用于
本实验对照。正确入口生成的 2.49 s1/s32 quant manifest 与 2.47 基线分别
具有完全相同的 SHA256，证明逻辑图、tensor shape、dtype 和 qparam 未改变。

prefill track 使用 micrograph 重新搜索得到的 s32 winner P19。P 是整个 context
的 finalize 配置，因此同一完整 context 中的 s1 也使用 P19；这里没有把 s1
micrograph 的 P6 偷换进全模。

| 指标 | QAIRT 2.47 / P19 | QAIRT 2.49 / P19 | 变化 |
|---|---:|---:|---:|
| profiling-off prefill | 797.735 tok/s | 783.659 tok/s | -1.76% |
| profiling-off decode | 41.682 tok/s | 45.410 tok/s | +8.94% |
| full s32 Softmax work | 11,708,607 cycles | 11,661,786 cycles | -0.40% |
| full s32 Softmax dominant | 2,657,277 cycles | 2,588,962 cycles | -2.57% |
| full s1 Softmax work | 1,711,373 cycles | 1,651,372 cycles | -3.51% |
| full s1 Softmax dominant | 397,485 cycles | 379,517 cycles | -4.52% |

完整图中仍有 448 个 Softmax，全部落到
`q::MaskedSoftmax_Crouton_Scratch`，kernel DRAM 仍为 0，VTCM 流量与 2.47
完全相同。s32 Softmax 的 dominant-path 节省只有 68,315 cycles，远不足以
覆盖 profiling-off prefill 增加的约 1.4 ms。

最低限度数值检查也没有发现 SDK 迁移分歧：2.49 的 100 个 sanity case 每例
前 8 个 token 都是 2.47/P19 结果的精确前缀。两者本身都不满足精度要求，但
本实验原定不设置精度 gate；这个检查只说明升级没有改变已存在的数值行为。

## 产物

- micrograph context 与 fixture：
  `/mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/native_u8_masked_softmax_qairt249/20260822`
- micrograph profiling：
  `/mnt/d/llm_exp/results/qwen3_sm8750_v79_native_u8_masked_softmax_qairt249_20260822`
- 完整 2.49/P19 context：上述模型目录下的 `full_model_p19`。
- 完整模型 profiling：
  `/mnt/d/llm_exp/results/qwen3_sm8750_v79_w4a8_rmsnorm_u8_softmax_qairt249_20260822_205055`
- 主要报告：完整模型结果目录下的
  `qwen3-sm8750-v79-g32-e2e-critical-path.html`。

所有大文件均在 D 盘归档。可再生的 WSL MIR、profile-viewer 和临时目录已在
实验结束后删除，共回收约 2.1 GiB；2.47 SDK 与原构建目录未被替换。
