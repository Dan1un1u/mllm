# Qwen3 SM8750/V79 G16/G32 profiling 可迁移复现指南

本文把 QNN-AOT、context 生成和真机 profiling 串成一条可迁移的路径。宿主机上的模型、中间产物和结果均放在 `llm_exp` 的 `models/`、`results/` 下；手机上的 `/data/local/tmp` 仍然只是设备侧临时运行目录。

本文对应的 VM 验证基线：

```text
repo    = /home/daniuniu/llm_exp/mllm
branch  = codex/quant-npu-w4a8-roadmap
commit  = 521941fb68056e8c63e67c9d9486ed20f151f8b7
QAIRT   = /opt/qcom/aistack/qairt/2.47.0.260601
```

不要把下面的绝对路径写死到脚本中。迁移到另一台机器时，只需修改 `LLM_EXP` 或通过环境变量覆盖 `ARTIFACT_ROOT`、`MODEL_ROOT`、`RESULTS_BASE`。

## 1. 产物布局与身份锁定

```bash
export LLM_EXP=/home/daniuniu/llm_exp
export REPO="$LLM_EXP/mllm"
export MODELS="$LLM_EXP/models"
export RESULTS="$LLM_EXP/results"
export QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601

G16_ROOT="$MODELS/qwen3_sm8750_v79/w4a16"
G32_ROOT="$MODELS/qwen3_sm8750_v79/g32/w4a16"
G32_EXPORT="$G32_ROOT/source_g32_export"
```

当前 VM 上的 canonical context 确实存在：

| 方案 | context | SHA-256 |
|---|---|---|
| G16 | `$G16_ROOT/qwen3-1.7B-lpbq-sha-sm8750-v79.bin` | `727fe97725abb0d6ff4efb5fa06564746bae89339b9079c88f8d6f865acb173d` |
| G32 | `$G32_ROOT/qwen3-1.7B-lpbq-sha-g32.bin` | `f637b4ddbd63478205679f40642fd24801093bb98ddf0808f13d99e3fb155d5d` |

G16 的旧 alias `$MODELS/output/qwen3-1.7B-lpbq-sha-sm8750-v79.bin` 与 canonical 文件也是同一个 `727fe…` 二进制。G32 的物理 profiling schematic 位于：

```text
$G16_ROOT/schematics/model.0.s1_schematic.bin
$G16_ROOT/schematics/model.0.s32_schematic.bin
$G32_ROOT/schematics/model.0.s1_schematic.bin
$G32_ROOT/schematics/model.0.s32_schematic.bin
```

先做身份检查，任何一项不匹配都不要把结果和历史结果直接比较：

```bash
sha256sum \
  "$G16_ROOT/qwen3-1.7B-lpbq-sha-sm8750-v79.bin" \
  "$G32_ROOT/qwen3-1.7B-lpbq-sha-g32.bin" \
  "$MODELS/Qwen3-origin/qwen3-tokenizer.json" \
  "$REPO/examples/qwen3_qnn_aot/config_1.7B_g32.json" \
  "$REPO/scripts/qwen3_sm8750_v79_accuracy.tsv"
```

## 2. 两类 build 必须分开

`mllm-qwen3-aot-sha-g32-c` 是在 x86 上运行的 QNN-AOT 编译器；Android build 只产生手机 runner。当前 Android build 没有这个 x86 编译器是正常的。

在一台新机器上，从项目根目录执行：

```bash
cd "$REPO"

# 需要 QAIRT x86 libraries 的 AOT/context 编译器
python3 task.py tasks/build_x86_qnn_aot.yaml

# 需要 Android NDK、QNN HTP custom-op package 的手机 runner
python3 task.py tasks/build_android_qnn.yaml
```

增量构建时可用：

```bash
cmake --build "$REPO/build-qnn-aot" \
  --target mllm-qwen3-aot-sha-g32-c -j"$(nproc)"
cmake --build "$REPO/build-android-arm64-v8a-qnn" \
  --target mllm-qwen3-aot-runner -j"$(nproc)"
```

应当看到以下文件：

```text
$REPO/build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-c
$REPO/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner
```

## 3. 从 BF16 source + 已校准 checkpoint 生成逻辑 G32 export

输入不是 `Qwen3-1.7B-G32-base` 的另一份模型，而是：

```text
$MODELS/Qwen3-origin/                 # 原始 BF16 shards、tokenizer、config
$MODELS/Qwen3-1.7B/model.safetensors  # 已有 A16/QDQ calibration 的 base checkpoint
```

G32 exporter 只重新编码 197 个 Linear/lm_head 权重，保留 base checkpoint 中已经校准的 A16/QDQ 张量。它导出的是 signed INT4 carrier、HWIO `[1,1,K,O]`、G32 level-1 UInt4 scale 和 per-output FP32 level-2 scale。命令如下：

```bash
cd "$REPO"
G32_WORK="$G32_ROOT/rebuild/source_g32_export"
mkdir -p "$G32_WORK"

python3 scripts/export_qwen3_lpbq_g32.py \
  --source-model "$MODELS/Qwen3-origin" \
  --base-quant-checkpoint "$MODELS/Qwen3-1.7B/model.safetensors" \
  --output-dir "$G32_WORK" \
  --group-size 32
```

随后转换为 mllm v2：

```bash
mllm-convertor \
  --input_path "$G32_WORK/model.safetensors" \
  --output_path "$G32_WORK/qwen3_1.7b_g32.mllm" \
  --model_name qwen3 --format v2 --verbose
```

如果要覆盖一个已有 export 目录，显式加 `--overwrite`；默认不要覆盖，这样每次迁移都有可审计的输入/输出目录。

### 历史归档与当前 VM 文件的差异

当前 VM 的 `source_g32_export` 是完整可用的，但其 hash 是：

```text
model.safetensors       cbf94972281923bec58115eded6f7090ddf7cfdf5e55ce25135d5712636b2195
qwen3_1.7b_g32.mllm     8240c2850154a9aaa9e5c938ebb114723bb7416e4c210698dfbfd893d6824001
rebuilt_context.bin     f637b4ddbd63478205679f40642fd24801093bb98ddf0808f13d99e3fb155d5d
```

历史报告中记录过另一份 export/mllm hash（`7a847…`/`ef8ce…`）。这说明 export 的字节级输入或序列化版本发生过变化；不要把 WSL 的 `Qwen3-1.7B-G32-base` 与 VM 归档混用。当前 `rebuilt_context.bin` 与 canonical G32 context 是同一个 `f637b4dd…` 二进制，因此可以直接复用该 context 做历史 baseline replay；若需要重新生成“完全相同”的历史 context，必须先恢复历史 export 输入并通过 hash gate。

## 4. 从 mllm 生成 G32 QNN context 与 schematics

`compile_sha.cpp` 会把 MIR 写到当前工作目录，HTP finalize 会在当前工作目录产生 `model.0.s1_schematic.bin`/`model.0.s32_schematic.bin`。因此必须先进入产物目录，不能在仓库根目录随意生成。

```bash
cd "$REPO"
mkdir -p "$G32_ROOT/rebuild/schematics"

AOT_BIN="$REPO/build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-c"
MODEL_MLLM="$G32_WORK/qwen3_1.7b_g32.mllm"
MODEL_CFG="$REPO/examples/qwen3_qnn_aot/config_1.7B_g32.json"
AOT_CFG="$REPO/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json"
QNN_ENV="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang"
SCHEMATICS="$G32_ROOT/rebuild/schematics"

cd "$SCHEMATICS"

# 先分别生成可供手机 Optrace viewer 使用的两个 schematic
"$AOT_BIN" \
  -m "$MODEL_MLLM" -c "$MODEL_CFG" -aot_cfg "$AOT_CFG" \
  -qnn_env "$QNN_ENV" --schematic_only --trace_seq 32 \
  2>&1 | tee schematic_s32.log

"$AOT_BIN" \
  -m "$MODEL_MLLM" -c "$MODEL_CFG" -aot_cfg "$AOT_CFG" \
  -qnn_env "$QNN_ENV" --schematic_only --trace_seq 1 \
  2>&1 | tee schematic_s1.log

# 再生成同时包含 s32/s1 graph 的 context.0 产物
"$AOT_BIN" \
  -m "$MODEL_MLLM" -c "$MODEL_CFG" -aot_cfg "$AOT_CFG" \
  -qnn_env "$QNN_ENV" \
  -o "$G32_ROOT/rebuild/rebuilt_context.bin" --trace_seq 0 \
  2>&1 | tee finalize_context.log
```

把 schematics 和 context 提升为 profiling 脚本使用的 canonical 路径前，先检查日志中必须出现 `SHA compilation completed successfully`，并检查 hash：

```bash
test -s "$G32_ROOT/rebuild/rebuilt_context.bin"
test -s "$SCHEMATICS/model.0.s1_schematic.bin"
test -s "$SCHEMATICS/model.0.s32_schematic.bin"
rg -q 'SHA compilation completed successfully' \
  "$SCHEMATICS/finalize_context.log"
sha256sum "$G32_ROOT/rebuild/rebuilt_context.bin"
```

只有在 hash 与期望值一致时才提升：

```bash
install -m 0644 "$G32_ROOT/rebuild/rebuilt_context.bin" \
  "$G32_ROOT/qwen3-1.7B-lpbq-sha-g32.bin"
install -m 0644 "$SCHEMATICS/model.0.s1_schematic.bin" \
  "$G32_ROOT/schematics/model.0.s1_schematic.bin"
install -m 0644 "$SCHEMATICS/model.0.s32_schematic.bin" \
  "$G32_ROOT/schematics/model.0.s32_schematic.bin"
```

G16 baseline 不需要重新量化；如果 canonical context 丢失，可用同一套 AOT 流程重建，但使用原来的 G16 编译器、mllm 和配置：

```bash
G16_WORK="$G16_ROOT/rebuild"
mkdir -p "$G16_WORK/schematics"
cd "$G16_WORK/schematics"

G16_AOT_BIN="$REPO/build-qnn-aot/bin/mllm-qwen3-aot-sha-c"
G16_MLLM="$MODELS/Qwen3-1.7B/qwen3_1.7b.mllm"
G16_CFG="$REPO/examples/qwen3_qnn_aot/config_1.7B.json"
G16_AOT_CFG="$REPO/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B.json"

"$G16_AOT_BIN" -m "$G16_MLLM" -c "$G16_CFG" \
  -aot_cfg "$G16_AOT_CFG" -qnn_env "$QNN_ENV" \
  --schematic_only --trace_seq 0 2>&1 | tee finalize_g16.log
"$G16_AOT_BIN" -m "$G16_MLLM" -c "$G16_CFG" \
  -aot_cfg "$G16_AOT_CFG" -qnn_env "$QNN_ENV" \
  -o "$G16_WORK/rebuilt_context.bin" --trace_seq 0 \
  2>&1 | tee finalize_g16_context.log
```

G16 只有在最终 hash 等于 `727fe977…` 时才能替换 `$G16_ROOT/qwen3-1.7B-lpbq-sha-sm8750-v79.bin`；否则它是一个新的 baseline，必须单独记录 SHA 和结果目录。

## 5. 一键跑真机 profiling

先确认手机在线：

```bash
adb devices -l
```

设备侧文件放在 `/data/local/tmp` 是脚本设计的一部分，不是宿主机的模型目录。宿主机的模型和结果仍由 `ARTIFACT_ROOT` 派生：

```bash
cd "$REPO"
export ARTIFACT_ROOT="$LLM_EXP"
export MODEL_ROOT="$MODELS"
export RESULTS_BASE="$RESULTS"
export QAIRT_SDK_ROOT="$QAIRT_SDK_ROOT"
export BUILD_ANDROID=0             # 已按第 2 节构建 runner
export PREPARE_DEVICE=1
export BENCHMARK_RUNS=3
export MAX_NEW_TOKENS=64
export ACCURACY_MAX_NEW_TOKENS=64

# 原 W4A16 G16 baseline，期望 context SHA = 727fe977…
./run_qwen3_sm8750_v79_profile.sh

# 原 W4A16 G32 baseline，期望 context SHA = f637b4dd…
./run_qwen3_sm8750_v79_g32_profile.sh

# WSL/Windows host 推荐使用项目内薄 wrapper；它仍然调用上面的 canonical
# 脚本，并保留完整 QNN viewer/HTML 后处理链。
./run_qwen3_sm8750_v79_g32_profile_wsl.sh
```

两个脚本都会在 `$RESULTS` 下创建带时间戳的目录，保存 runner、context、tokenizer、config、schematic、git commit、设备属性和 profiling 报告的 hash。脚本在启动阶段会拒绝错误 context SHA，因此“模型不同但脚本仍然运行”的情况不会被静默接受。

`run_qwen3_sm8750_v79_g32_profile_wsl.sh` 默认强制
`ACCURACY_MAX_NEW_TOKENS=64`。如果只是做短 smoke，显式设置
`STRICT_BASELINE=0`；短 smoke 的准确率不能和 VM baseline 比较。

若 context 尚未复制到脚本默认的 canonical 路径，也可以显式指定同一个文件，不要修改 `EXPECTED_CONTEXT_SHA`：

```bash
LOCAL_MODEL="$G32_ROOT/rebuild/rebuilt_context.bin" \
SCHEMATIC_DIR="$G32_ROOT/rebuild/schematics" \
./run_qwen3_sm8750_v79_g32_profile.sh
```

## 6. VM、WSL、Windows host 的复现规则

WSL wrapper 只能负责调用同一份项目脚本，不能偷偷替换模型目录。对比 VM 与 WSL 前，必须锁定以下六项：

1. `git_commit.txt` 相同；
2. `LOCAL_MODEL` 的 SHA 相同（G16 `727fe…` 或 G32 `f637…`）；
3. tokenizer、config、runner、两张 schematic 的 SHA 相同；
4. `QAIRT_SDK_ROOT`/QNN Build Id 相同；
5. `ACCURACY_MAX_NEW_TOKENS` 相同，历史 baseline 使用 64，不要一边使用 32；
6. `BENCHMARK_RUNS`、`AR_LEN`、prompt、profiling level 相同。

在 WSL 中可以直接调用 Windows ADB，例如：

```bash
ADB_BIN=/mnt/c/Android/platform-tools/adb.exe \
ARTIFACT_ROOT=/mnt/d/llm_exp \
MODEL_ROOT=/mnt/d/llm_exp/models \
RESULTS_BASE=/mnt/d/llm_exp/results \
QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601 \
BUILD_ANDROID=0 \
./run_qwen3_sm8750_v79_g32_profile.sh
```

如果真机 USB 被 VM 占用，则让 VM 运行 profiling；不要同时让 Windows/WSL/VM 抢同一 ADB USB 设备。WSL 运行 wrapper 时，禁止使用另一个 `Qwen3-1.7B-G32-base` export，否则得到的 context、准确率和速度不能与 VM 归档结果比较。

host 不能只执行 `adb shell`、`adb pull` 或旧的 `run_qnn_profile.sh` 来代替 canonical 脚本；那样最多得到 raw `qnn-profiling-data.log`，不会得到本项目的 QHAS、operator/layer CSV、structure HTML 和 E2E critical-path HTML。`qnn-profile-viewer` 是 Linux QNN SDK 工具，必须在 WSL 内以正确的 `QAIRT_SDK_ROOT` 执行；若 WSL 没有 QAIRT SDK，应把 raw capture 和对应 schematic 复制回 VM，再在 VM 运行 canonical 后处理。

薄 wrapper 在 strict baseline 模式下还会校验 VM 参考 runner、tokenizer、config 和 accuracy suite 的 SHA。这样即使 context/速度相同，只要 host runner 没有包含当前 accuracy 修复，也会在启动前明确失败，而不是生成一个看似可比较的低分。

结构分类器也有 semantic gate：只有 `model.layers.*.self_attn.*` 的 `CastType/Slice/Transpose` 才能进入 `kv_cache_update`；`model.layers.*.mlp.CastType.*` 会保留为 `unclassified`，`lm_head`/`model.lm_head` 会归入 LM head。重新生成 host 报告时必须使用当前仓库的 `scripts/qnn_optrace_qwen3_structure.py`，不能继续使用旧 wrapper 自带的分类器。

## 7. 已完成的链路证据

当前 VM 已完成以下验证：

- G16 context `727fe…` 和 G32 context `f637…` 均存在；
- G32 AOT 编译器 `build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-c` 增量构建成功；
- `source_g32_export/model.safetensors → qwen3_1.7b_g32.mllm → rebuilt_context.bin` 的现有产物、MIR、schematics 和 finalize log 齐全；
- `rebuilt_context.bin` 与 canonical G32 context 二进制 SHA 完全相同；
- 真机 smoke 结果目录：
  `$RESULTS/qwen3_sm8750_v79_g32_20260807_020505`；
- smoke 中设备端上传文件 SHA 为 `f637b4dd…`，QNN 成功加载 2 个 graph，S1/S32 Optrace 和 HTML 报告均生成，脚本退出码为 0。

该 smoke 的 `ACCURACY_MAX_NEW_TOKENS=2` 是链路测试参数，产生的 `49/100` 不可作为质量结论。正式比较必须使用第 5 节的 64 token 设置和相同的产物身份。
