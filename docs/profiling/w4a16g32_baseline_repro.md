# Qwen3-1.7B W4A16G32 baseline：可迁移复现指南

这条分支只固化一个 baseline：Qwen3-1.7B、SM8750/V79、W4 LPBQ G32、A16
activation/output。它对应的参考结果是：

```text
results/qwen3_sm8750_v79_g32_20260807_160125
```

代码整理以 `ae8ac831` 为源提交；参考结果目录是在后续已验证的同一套
G32 runner/profiling 代码上生成的。两者的关系和所有二进制身份都记录在
`profiles/qwen3_sm8750_v79_g32/baseline.env`，不会靠当前工作树名称猜测。

参考结果不是“某次偶然速度数字”，而是由 contract 中的 context、runner、
tokenizer、config、100 题 suite 和两个 schematic 唯一确定。宿主机模型与
中间产物不进 Git，必须放在仓库上级的 `models/`；结果放在仓库上级的
`results/`。

## 1. 分支与目录

```bash
export LLM_EXP=/home/daniuniu/llm_exp   # 迁移到另一台机器时只改这一行
export REPO="$LLM_EXP/mllm"
export MODEL_ROOT="$LLM_EXP/models"
export RESULTS_BASE="$LLM_EXP/results"
export QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601

cd "$REPO"
git switch codex/w4a16g32-repro-baseline
```

如果仓库目录名改变，不要改脚本；设置 `REPO_ROOT` 或让脚本按自身位置
推导即可。`models/` 和 `results/` 必须与 `mllm/` 同级。

## 2. 必须准备的产物

contract 文件是：

```text
profiles/qwen3_sm8750_v79_g32/baseline.env
```

它锁定以下数据产物 SHA；任意一项不一致都不能与参考结果比较。
Android runner 的 SHA 保留为 VM 参考 provenance，但 WSL 构建的 ELF 会嵌入
源码提交和调试路径，因此 runner 只做存在性检查，不再作为阻断 gate：

| 产物 | 相对路径 | SHA-256 |
|---|---|---|
| Android runner | `mllm/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner` | `f8a00d53b001e017405b6d3061b57195580bfc2877bb14eb8e335f21c7f6452d` |
| QNN context | `models/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin` | `f637b4ddbd63478205679f40642fd24801093bb98ddf0808f13d99e3fb155d5d` |
| tokenizer | `models/Qwen3-origin/qwen3-tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| G32 config | `examples/qwen3_qnn_aot/config_1.7B_g32.json` | `1cf89d946a8138be13050bb2125b1302d9157a308d77e391d3ce7eceb2f22db0` |
| accuracy suite | `scripts/qwen3_sm8750_v79_accuracy.tsv` | `5bbcd2f39d176511d216c897191887c9e4577dc6a49d85c01c4dd7a4c6ad931c` |
| s1 schematic | `models/qwen3_sm8750_v79/g32/w4a16/schematics/model.0.s1_schematic.bin` | `eaa484fb1b5446c60c130235e256a485e92ddeccdea96fead0026e3a3f977648` |
| s32 schematic | `models/qwen3_sm8750_v79/g32/w4a16/schematics/model.0.s32_schematic.bin` | `df2b8776625b6167d1306bb7c26ad4c9d2e62a4e8b8954a4c7d3505614032434` |

迁移后先只做宿主机检查：

```bash
ARTIFACT_ROOT="$LLM_EXP" \
  "$REPO/scripts/verify_qwen3_sm8750_v79_g32_baseline.sh"
```

该检查不访问手机、不依赖 QAIRT，只确认本地文件没有拿错。

## 3. 构建 runner（需要时）

`mllm-qwen3-aot-runner` 是 Android runner；G32 context/schematic 是另一条
x86 QNN-AOT 产物链。不要把 Android runner 当成 context compiler。

```bash
cd "$REPO"
python3 task.py tasks/build_android_qnn.yaml
```

构建结束后必须重新执行第 2 节 preflight。preflight 会记录实际 runner SHA，
但不再因它与 VM 参考值 `f8a00d53…` 不同而拒绝运行。比较速度和精度时仍要
把实际 runner SHA、源码提交和工具链写入结果，不能把不同 runner 的结果当作
bit-for-bit 复现。

如果 context/schematic 丢失，不能从另一份 `Qwen3-1.7B-G32-base` 或 W4A8
产物替代。应恢复与这个 SHA 对应的 G32 export/AOT 输入，再重新生成并通过
其余数据产物的 hash gate；否则只能登记为新的实验版本。

## 4. 真机 profiling 一键入口

先确认 QAIRT 和手机：

```bash
test "${QAIRT_SDK_ROOT##*/}" = 2.47.0.260601
adb devices -l
```

VM/Linux 直接执行 canonical entry：

```bash
cd "$REPO"
ARTIFACT_ROOT="$LLM_EXP" \
MODEL_ROOT="$MODEL_ROOT" \
RESULTS_BASE="$RESULTS_BASE" \
QAIRT_SDK_ROOT="$QAIRT_SDK_ROOT" \
BUILD_ANDROID=0 \
PREPARE_DEVICE=1 \
BENCHMARK_RUNS=3 \
MAX_NEW_TOKENS=64 \
ACCURACY_MAX_NEW_TOKENS=64 \
AR_LEN=32 \
  ./run_qwen3_sm8750_v79_g32_profile.sh
```

如果 runner 尚未构建，把 `BUILD_ANDROID=0` 改成 `1`，脚本会先执行 Android
QNN build，再做同一套数据产物 SHA gate。

WSL/Windows ADB 使用薄 wrapper，所有 profiling、HTML 和分类后处理仍由
canonical script 完成：

```bash
ADB_BIN=/mnt/d/llm_exp/mllm/scripts/adb_wsl_path_wrapper.sh \
ADB_EXE=/mnt/c/adb/adb.exe \
QAIRT_SDK_ROOT=/mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.47.0.260601 \
ARTIFACT_ROOT=/mnt/d/llm_exp \
BUILD_ANDROID=0 \
  ./run_qwen3_sm8750_v79_g32_profile_wsl.sh
```

wrapper 不允许替换模型、config、suite 或预期 SHA；若想换产物，应创建新的
branch/contract，不要继续把结果放进 `qwen3_sm8750_v79_g32_*` baseline 目录。

## 5. 脚本实际做什么

一次执行固定包含以下阶段：

1. 本地 artifact SHA 和 QAIRT release 检查；
2. 将 runner、QNN `.so`、context、tokenizer、config 推到手机
   `/data/local/tmp`，并检查设备端 runner/context/tokenizer/config SHA；
3. profiling-off E2E benchmark，3 个 fresh process；
4. profiling-off 100 题短答 sanity，`max_new_tokens=64`；
5. `model.0.s32`、`model.0.s1` 各做一次 fresh-process Optrace；
6. 生成速度 JSON、operator/type summary、Qwen3 stage CSV、结构 HTML 和
   canonical e2e critical-path HTML；
7. 按 `CLEAN_REMOTE=1` 只删除本次 timestamp 的手机结果目录。

宿主机结果默认写入：

```text
$LLM_EXP/results/qwen3_sm8750_v79_g32_YYYYMMDD_HHMMSS/
```

手机侧 `/data/local/tmp` 只是运行目录；它不是宿主机模型目录，也不作为迁移
输入。结果目录内会保存 `experiment_script.sh`、`base_profile_script.sh`、
`experiment_metadata.txt`、`artifact_sha256.txt`、两套 Optrace 原始数据和
HTML，便于审计。

## 6. 与参考结果对齐的验收项

迁移后不要只看 token/s；先确认新结果的：

```text
context_sha256 = f637b4dd…
runner_sha256  = f8a00d53…
config_sha256  = 1cf89d94…
suite_sha256   = 5bbcd2f3…
accuracy       = 77/100（64 token 上限）
```

目标结果的速度样例为：prefill median `849.49 token/s`、decode median
`45.36 token/s`。真机温度和系统调度会带来小幅波动；若 artifact SHA、suite、
prompt、token budget 或 profiling graph 不同，则不能把速度差异归因于代码。

最终 HTML 应位于结果根目录：

```text
qwen3-sm8750-v79-g32-e2e-critical-path.html
```

如果没有这个文件，说明流程没有完成，不能把目录称为一次完整 profiling。
