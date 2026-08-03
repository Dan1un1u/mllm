# Qwen3 全模型 G32 W4A16 首轮闭环

## 范围

本轮先完成全模型 **真实 G32 LPBQ + A16** 的部署闭环。为了隔离变量，权重
不做 R1/R2 旋转；A16/QDQ 参数直接复用已有 G16 checkpoint 的校准结果。
因此这不是固定 H64 结果，也不是 W4A8 精度结果。固定 H64 后续需要重新做全
模型 activation calibration，不能直接复用这份 QDQ。

## 导出与验证

导出脚本：`scripts/export_qwen3_lpbq_g32.py`

```bash
python scripts/export_qwen3_lpbq_g32.py \
  --source-model /home/daniuniu/llm_exp/models/Qwen3-origin \
  --base-quant-checkpoint /home/daniuniu/llm_exp/models/Qwen3-1.7B/model.safetensors \
  --output-dir /tmp/Qwen3-1.7B-G32

mllm-convertor \
  --input_path /tmp/Qwen3-1.7B-G32/model.safetensors \
  --output_path /tmp/qwen3_1.7b_g32.mllm \
  --model_name qwen3 --format v2 --verbose
```

离线 gate：

- 197/197 个 Qwen3 Linear/lm_head 权重改为 G32；
- weight 为 `[1, 1, K, O]` INT8 carrier，signed code 为 `[-7, 7]`；
- `scale1` 为 UInt4 carrier、大小 `O * (K / 32)`；`scale2` 为 FP32 `[O]`；
- 总 tensor key 仍为 7,739；除 weight/scale1/scale2 外的 7,542 个 tensor 与
  原 G16 checkpoint bit-exact；
- safetensors：2,401,321,336 bytes；mllm：2,402,965,508 bytes。

```text
safetensors sha256: 7a84709fcdf851da6c131a827a64e9daa0f62d23fb1979242819ed9caac62735
mllm sha256:        ef8ce5fbe021603a6e3a4386dc63620e70af0300b1986780f6f32affd430cb87
```

## AOT G32 接线

原 Qwen3 AOT header 的 Conv2D property 原本写死 G16。本轮改为保留旧 target，
另加两个 target：

- `mllm-qwen3-aot-g32-c`
- `mllm-qwen3-aot-sha-g32-c`

通过 `MLLM_QWEN3_QNN_AOT_G32` compile definition 选择
`kQNN_LPBQ_w4a16o16_G32`，不改变旧 G16 target。对应配置为：

- `examples/qwen3_qnn_aot/config_1.7B_g32.json`
- `examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json`

SHA-G32 graph finalize 已完成，两个 MIR 中均为：

```text
QuantSpec(LPBQ(... block_size: 32 ... quant_to_type: Int4 ...))
```

`seq=1` 和 `seq=32` 均统计到 2,018 个 `block_size: 32`，没有
`block_size: 16`。当前 context：

```text
/tmp/qwen3-1.7B-lpbq-sha-g32.bin
size: 1,563,062,272 bytes
sha256: f637b4ddbd63478205679f40642fd24801093bb98ddf0808f13d99e3fb155d5d
```

## SM8750 真机 smoke

真机 `PJZ110`（V79/SM8750）已用隔离目录
`/data/local/tmp/qwen3_g32_test` 加载上述 context。为了让 Android shell
环境能够找到 RPC memory 库，运行时使用：

```bash
export LD_LIBRARY_PATH=.:/data/local/tmp
export ADSP_LIBRARY_PATH=/data/local/tmp
./mllm-qwen3-aot-runner \
  -m qwen3-1.7B-lpbq-sha-g32.bin \
  -t qwen3-tokenizer.json \
  -c config_1.7B_g32.json \
  --ar_len 32 --max_new_tokens 2 --perf
```

本次 shell smoke 仍显式把设备侧 RPC 库放入 `/data/local/tmp`，并把该目录加入
`LD_LIBRARY_PATH`，以避免不同 Android linker namespace 的差异。

QNN context、`model.0.s1`/`model.0.s32` 两张图和 HTP backend 均成功加载，短
prompt 返回 `Hello!`。这次仅是加载/执行 smoke，不是和 G16 的受控性能比较；在
默认 linting profiling 下该 prompt 的观测值为 prefill 3.214 s（18 tokens）、
decode 3.058 s（1 token），不能作为最终吞吐结论。

随后关闭 profiling（`MLLM_QNN_PROFILE_LEVEL=off`），用同一个 18-token prompt
各跑一次 G16/G32 context，得到：

| context | prefill | decode(1 token) | 输出 |
|---|---:|---:|---|
| G16 | 57.55 ms | 29.83 ms | `Hello!` |
| G32 | 52.68 ms | 25.72 ms | `Hello!` |

这是单次 smoke A/B，不能替代多轮稳态 benchmark；但至少说明 G32 在当前实现上
没有出现预期的反向性能退化（这次测量约快 8.5%/13.8%）。

## 目前结论与下一步

这证明全模型 G32 的权重布局、QNN AOT lowering、V79 graph finalize、context
serialization 和真机加载/执行均可行。还没有 G32 相对 G16 的受控速度/任务精度
结论；下一步需要关闭或统一 profiling 设置，把两种 context 做同 prompt、
prefill/decode、内存和输出对比。

之后再从同一 exporter 引入固定 H64，并用全模型 calibration 重新生成 A16/QDQ；
再继续研究 A8 activation scale，避免把旋转导致的旧 QDQ 失配误判为 G32 误差。

## G32-only profiling 脚本

已从原 G16 脚本复制出独立入口：
`run_qwen3_sm8750_v79_g32_profile.sh`。它固定使用 G32 context、G32 config、G32
context SHA 和 G32 schematic，不执行 G16 A/B 对比；流程包含 profiling-off E2E、
短答案 sanity、`model.0.s32`/`model.0.s1` Optrace，以及 QHAS/HTML 汇总。默认值
仍可用环境变量覆盖，例如首次 smoke 可缩短为：

```bash
QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601 \
RESULTS_BASE=/tmp/qwen3_g32_profile_results \
SCHEMATIC_DIR=/tmp/qwen3_sm8750_v79_g32_schematics \
BUILD_ANDROID=0 BENCHMARK_RUNS=1 MAX_NEW_TOKENS=8 \
ACCURACY_MAX_NEW_TOKENS=64 \
./run_qwen3_sm8750_v79_g32_profile.sh
```

在 SM8750/V79 真机上的一次完整 profile smoke 已成功生成；随后用同一设备和
同一 G32 context 补跑了 `accuracy_max_new_tokens=64`：

- 结果目录：`/tmp/qwen3_g32_profile_results/qwen3_sm8750_v79_g32_20260803_144447`
- prefill E2E：871.75 token/s（62 tokens，单轮）
- decode E2E：44.73 token/s（7 tokens，单轮）
- accuracy sanity：75/100（`accuracy_max_new_tokens=64`；同 suite 的 G16 真机基线为
  78/100）
- canonical report：`qwen3-sm8750-v79-g32-e2e-critical-path-accuracy64.html`

结果中的 `qwen3-sm8750-v79-g32-*` 文件是 G32 原始产物；脚本只在结果目录内为
现有 canonical report reader 建立临时旧前缀相对链接，不修改 G16 脚本或其结果。

注意：最初为缩短 smoke 时间使用 `accuracy_max_new_tokens=8` 时得到的 41/100
不是可比较的精度结论；大量答案在生成到最终数值/选项之前就被截断。正式 sanity
至少使用 64 个新 token。
