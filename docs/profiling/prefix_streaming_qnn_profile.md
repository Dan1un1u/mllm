# Prefix-streaming QNN profiling

The two entry points are:

```bash
./run_qwen3_sm8750_v79_prefix_streaming_mixed_profile.sh
./run_qwen3_sm8750_v79_prefix_streaming_allrisk_profile.sh
```

Both wrappers call `run_qwen3_sm8750_v79_g32_profile.sh` and write timestamped
results below `/mnt/d/llm_exp/results/prefix_streaming_mixed` or
`/mnt/d/llm_exp/results/prefix_streaming_allrisk` (the corresponding Windows
path is `D:\llm_exp\results`).  They also record the 28 per-layer scale-file
hashes, training manifest, precision map, native context hash, and the offline
held-out metrics.

The wrappers intentionally require a newly compiled native V79 context:

```text
/tmp/qwen3-1.7B-prefix-streaming-mixed-sm8750-v79.bin
/tmp/qwen3-1.7B-prefix-streaming-allrisk-sm8750-v79.bin
```

Set `LOCAL_MODEL` and, optionally, `EXPECTED_CONTEXT_SHA` when the contexts
are stored elsewhere.  The archived baseline and the older July actaware or
selective contexts are rejected by default; their SHA is not a valid
provenance marker for the August prefix-streaming artifacts.

For WSL, use the requested QAIRT release and Windows ADB interop explicitly:

```bash
QAIRT_SDK_ROOT=/mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.47.0.260601 \
ADB_BIN=adb.exe BUILD_ANDROID=0 \
  ./run_qwen3_sm8750_v79_prefix_streaming_mixed_profile.sh
```

`BUILD_ANDROID=1` remains the default so the same wrappers can be run in the
VM after the Android/QNN runner and the matching G32 schematics are built.
