# EXP-0014 QHPI integer-HMX feasibility probe

This package is an isolated feasibility probe. It is not an optimized MatMul
implementation and is not enabled by normal mllm model compilation.

The fixed micrograph contract is:

- activation: `[1, 8, 8, 32]`, asymmetric U8, scale 1, zero point 128;
- weight: `[1, 1, 32, 32]`, symmetric S8, scale 1;
- output: `[1, 8, 8, 32]`, asymmetric U8, scale 1, zero point 0;
- one QNN graph and one QHPI custom node;
- QHPI input/output storage restricted to TCM;
- HMX performs `activation.ub * weight.b` and writes saturated U8;
- the custom node must report zero DRAM reads and writes in Optrace.

HTP exposes the logical S8 weight input to QHPI as QUInt8 physical storage
with zero offset 128. The wrapper stages its signed representation and the
asymmetric zero-point correction in the output crouton before HMX overwrites
that crouton with the result. No separate scratch buffer is requested.

`HmxInt8Tile.hpp` uses HMX channel-major (`:cm`) activation loads and
channel-major saturated stores. These modifiers are required to bridge QNN's
Crouton8 view directly to HMX without a layout-conversion pass.

## Build

Set `QAIRT_SDK_ROOT`, `HEXAGON_SDK_ROOT`, and `ANDROID_NDK_ROOT`, then run:

```sh
make -C mllm/backends/qnn/custom-op-package/QhpiHmxProbePackage \
  all emulation
```

The host emulation test is:

```sh
mllm/backends/qnn/custom-op-package/QhpiHmxProbePackage/build/x86_64-linux-clang/hmx_int8_tile_emulation
```

The AOT compiler and Android runner targets are:

```text
mllm-qhpi-u8-hmx-probe-c
mllm-qhpi-u8-hmx-probe-runner
```

Compilation is opt-in through both environment variables:

```text
MLLM_QNN_AOT_OP_PACKAGE_PATH
MLLM_QNN_AOT_OP_PACKAGE_PROVIDER=QhpiHmxProbePackageInterfaceProvider
```

Runtime registration uses:

```text
MLLM_QNN_OP_PACKAGE_PATH=libQnnQhpiHmxProbePackage.so
MLLM_QNN_OP_PACKAGE_PROVIDER=QhpiHmxProbePackageInterfaceProvider
```

The compiler executable sets `MLLM_QNN_QHPI_U8S8_HMX_PROBE=1` internally, so
ordinary MatMul lowering remains `qti.aisw::MatMul`.

## Acceptance evidence

The gate passes only when all of the following are independently observed:

1. identity, signed-permutation, and structured patterns match the integer
   reference byte for byte on the target;
2. Optrace attributes the custom node to HMX and reports `uses_hmx`;
3. custom-node DRAM read and write counters are both zero;
4. no spill, fill, fallback, or graph split is present;
5. DSP disassembly contains `activation.ub`, `weight.b`, `:cm`, and `sat.ub`.

Graph input and output boundary DMA is outside the custom-node zero-DRAM
contract. Passing this probe establishes backend feasibility only; it does not
adopt or authorize a fused Attention implementation.
