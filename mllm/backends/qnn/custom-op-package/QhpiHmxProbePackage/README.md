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

## EXP-0015 mixed-resource plumbing mode

The same isolated package also registers `MixedResourceHmxHvxHmx`. It is a
Stage-A plumbing gate, not the fused GQA implementation. One resource-exclusive
QHPI invocation performs:

```text
HMX U8xS8 -> output crouton
HVX saturating byte add-one in place
HMX U8xS8 -> output crouton
```

The signed weight and asymmetric-correction bias blocks are staged first in
the output crouton and then in the now-dead activation crouton. The second HMX
tile reads and overwrites the output crouton in place. A reserved zero weight
word carries the phase between resource classes in VTCM; `sync_block_size` is
zero because QAIRT 2.49 serializes QHPI sync blocks as DDR tensors. No
graph-visible workspace is used.
Compile the existing probe with `--mode mixed-resource` and run it with
`--pipeline mixed-resource`. Formal acceptance still requires device output to
match the two-HMX host reference byte for byte and Optrace to prove both HMX and
HVX execution with zero custom-node DRAM.

### QAIRT 2.49 / V79 Stage-A finding

This candidate does not pass Stage A. With `QHPI_RESOURCE_EXCLUSIVE`, the
target invokes a main-control callback (`qhpi_thread_resources() == 0`) and an
HVX callback, but does not provide the HMX execution context used by the
accepted EXP-0014 `QHPI_RESOURCE_HMX` kernel. Issuing even one copy of the
EXP-0014 HMX tile from the exclusive main callback causes a CDSP subsystem
restart (`DspTransport 0x10`, graph error `1003`). Setting `multithreaded=true`
does not change the dispatched resource classes, and the apparent combined
flag `QHPI_RESOURCE_HVX | QHPI_RESOURCE_HMX` is rejected by the compiler as
resource flag `0x6`.

QHPI synchronization memory is also unsuitable for the zero-DRAM contract in
this SDK: an 8192-byte sync block produces `ddrTensorSize=8704`, while the
TCM-tensor phase-word variant produces `ddrTensorSize=0`. The guarded
`QHPI_MIXED_RESOURCE_AUDIT` and `QHPI_EXCLUSIVE_SINGLE_HMX_AUDIT` builds retain
the minimal reproductions. Stage B must not be started unless the experiment
contract is explicitly changed.

## EXP-0016 sequential HMX plumbing mode

`SequentialHmxHvxHmx` implements the approved single-callback alternative.
It reserves `QHPI_RESOURCE_HMX` and issues this complete sequence from the one
main-control callback used by QAIRT 2.49/V79:

```text
HMX U8xS8 -> output crouton
HVX saturating byte add-one in place
HMX U8xS8 -> consumed activation crouton
HVX TCM copy -> output crouton
```

Compile with `--mode sequential-hmx` and run with
`--pipeline sequential-hmx`. The extra final vector copy is required because
V79 does not preserve the activation crouton when the second HMX tile uses the
same block for input and output. An alias-audit build also established that
QHPI's `source_destructive=true` permits the first input and output to share
one physical block. The accepted implementation therefore keeps that
descriptor false, requires distinct input/output blocks, and reuses the
already-consumed activation block only after the first HMX tile.

### QAIRT 2.49 / V79 Stage-A finding

The Stage-A plumbing gate passes. Identity, signed-permutation, and structured
inputs each remain byte-exact over ten target invocations. The cached context
reports `ddrTensorSize=0` and `spillFillBufferSize=0`. Decoded Optrace reports
the one `SequentialHmxHvxHmx` node as `hmx=true`, with custom-node DRAM read
and write both zero. DSP disassembly independently contains two integer HMX
tiles, the intervening HVX saturated add, and the final HVX TCM copy.

This result proves only the one-node HMX/HVX scheduling and storage contract.
It does not establish a complete GQA implementation or a speed result.
