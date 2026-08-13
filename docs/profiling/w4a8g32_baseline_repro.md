# Qwen3-1.7B W4A8G32 实验 baseline 复现指南

本 baseline 从归档 W4A16G32 独立派生：LPBQ Int4/G32 权重不变，只把目标
Linear/Conv2D 的 per-tensor asymmetric UInt16 activation/output 改为 UInt8。
RMSNorm/Embedding 的 UInt16 权重和 UInt8 symmetric KV cache 不变。精度与速度
均不设门槛，但构建、运行、三轮 runner E2E、100 题 sanity、s1/s32 Optrace
和联合量化证据必须完整。

## 固定边界

```text
源码              /home/daniuniu/work/mllm-w4a8
模型和中间产物    /mnt/d/llm_exp/models
结果              /mnt/d/llm_exp/results
真机运行目录      /data/local/tmp/mllm_w4a8g32
QAIRT              /mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.47.0.260601
W4A16 参考结果     /mnt/d/llm_exp/results/qwen3_sm8750_v79_g32_20260807_230410
```

历史 W4A8 的源码、配置、模型、中间产物、日志、结果和真机目录均不得读取或
复用。归档 W4A16 只读。Git 只跟踪源码；模型、context、schematic、manifest、
profiling CSV/Optrace/HTML 均在 `/mnt/d/llm_exp`。

## 从原始模型构建

原始模型固定为 `/mnt/d/llm_exp/models/Qwen3-origin`。激活校准直接运行 mllm
observer 的 asymmetric 8-bit 模式，使用固定的 128×512 token corpus；不得把
A16 scale/zero-point 按比例压缩冒充 A8 校准。

```bash
cd /home/daniuniu/work/mllm-w4a8
bash scripts/build_qwen3_w4a8g32_model.sh
bash scripts/build_qwen3_w4a8g32_context.sh RUN_ID
```

其中 `RUN_ID` 是上一条命令打印的 staging 目录末段（也可显式把它作为第一条
命令的参数传入，使两个阶段使用同一 ID）。

正式候选产物：

```text
/mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8/
  qwen3-1.7B-w4a8g32-sha.bin
  schematics/model.0.s1_schematic.bin
  schematics/model.0.s32_schematic.bin
  manifests/model.0.s1_quant_manifest.json
  manifests/model.0.s32_quant_manifest.json
```

`profiles/qwen3_sm8750_v79_g32/baseline.env` 固定这些产物、tokenizer、config、
suite 和 runner 的 SHA-256。主机预检：

```bash
scripts/verify_qwen3_sm8750_v79_g32_baseline.sh
```

## 真机 profiling

Android runner 的根目录 build cache 保留在仓库默认位置，但由 `.gitignore`
排除。WSL 使用 Windows ADB wrapper：

```bash
cd /home/daniuniu/work/mllm-w4a8
export ADB_EXE=/mnt/c/adb/adb.exe
export QAIRT_SDK_ROOT=/mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.47.0.260601
export ANDROID_NDK_PATH=/home/daniuniu/toolchains/android-ndk-r26c
export MODEL_ROOT=/mnt/d/llm_exp/models
export RESULTS_BASE=/mnt/d/llm_exp/results
export ADB_BIN=/home/daniuniu/work/mllm-w4a8/scripts/adb_wsl_path_wrapper.sh
export BUILD_ANDROID=0
./run_qwen3_sm8750_v79_g32_profile.sh
```

若真机 capture 已完成、仅主机侧官方 Optrace 解码或报告生成失败，可在修复后
设置 `RESUME_RESULT_ROOT` 指向同一个候选结果目录。脚本只跳过已经具备完整
runner CSV、accuracy JSON 或 Chrome/QHAS 三件套的阶段；不完整的 graph 会重新
capture，避免把残缺结果误判为已完成。

脚本固定执行：三次 profiling-off fresh-process E2E（AR=32、64 tokens）、同一
100 题 suite 的信息性 sanity、s32 与 s1 各一次 fresh-process Optrace、量化
manifest/Optrace 联合验收、与指定 W4A16 speed JSON 的比较，以及最终报告。
正式比较强制三份 `qnn_runner_e2e.csv`，禁止用受 tracing 扰动且不含 host 工作的
QHAS 数据替代。

## 真实 W4A8 验收

每个 s1/s32 target LPBQ operation 必须同时满足：

- manifest 权重 recipe 是 LPBQ Int4，block size 32；
- manifest activation/output 是 asymmetric UInt8、量化范围 0..255，QNN dtype
  是 `UFIXED_POINT_8`；
- Optrace 中能观察到相同 QNN operation，且 target 路径不出现 `QUInt16`；
- execution domain 是 HTP；host/runtime 工作不在 Optrace 的证明范围内。

QAIRT 2.47 的 HTP RmsNorm 整数约束要求 input/gamma/bias/output 位宽一致。为了
保留归档方案中的 UInt16 gamma/bias，图中显式加入 A8→A16→A8 HTP Convert。
这属于已记录的 SDK 边界转换，不是 target Linear/Conv2D fallback；其代价必须
保留在 Optrace 和瓶颈解释中。

完成结果位于：

```text
/mnt/d/llm_exp/results/qwen3_sm8750_v79_w4a8g32_YYYYMMDD_HHMMSS/
```

重点查看 `qwen3-sm8750-v79-g32-e2e-critical-path.html`。报告包含 W4A8 与
`qwen3_sm8750_v79_g32_20260807_230410` 的 runner E2E 对比；无论净加速与否，
都必须报告转换、数据搬运和 HMX/HVX 等主要瓶颈。

## 首个正式实验结果

正式结果目录：

```text
/mnt/d/llm_exp/results/qwen3_sm8750_v79_w4a8g32_20260813_135938
```

联合验收通过：s1、s32 各有 1009 个 manifest target operation，运行时 Optrace
均匹配 1009/1009，target 的物理输入/输出均没有 `QUInt16`。这证明 target
Linear/Conv2D 已按 W4G32 + asymmetric UInt8 activation 在 HTP 图内执行；证明
范围不包括 Optrace 之外的 host/runtime 工作。

三轮 profiling-off runner 中位数与归档 W4A16 的对比如下：

| phase | W4A8 | W4A16 reference | change |
|---|---:|---:|---:|
| prefill E2E | 738.693 token/s | 859.964 token/s | -14.10% |
| decode E2E after first | 37.632 token/s | 45.490 token/s | -17.27% |

首版没有净加速。s1 关键路径以 MLP up（18.52%）、lm_head（18.25%）、MLP
gate（16.90%）、MLP down（14.28%）为主；s32 以 MLP up（16.16%）、MLP down
（15.03%）、lm_head（14.50%）、MLP gate（14.49%）为主。这些阶段仍属于
LPBQ weight stream + HMX。RMSNorm 的 A8↔A16 SDK 边界桥接由 HVX Convert 执行，
并保留在报告成本中；attention 内还可见 unsigned-to-signed 的 HVX 转换。因此，
只缩窄 activation 没有减少主导的 LPBQ 权重流/HMX 成本，新增转换进一步抵消收益。

100 题 sanity 为信息项而非门槛：执行完成且结果可解析，但得分 0/100，并观察到
11 个 NUL byte。这是严重数值质量风险；baseline 只可用于后续硬件实验，不能视为
可用精度模型。
