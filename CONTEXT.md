# mllm W4A8 Hardware Baseline

This context defines the shared language for deriving an experimental W4A8 hardware-inference baseline from the archived W4A16 baseline.

## Language

**Archived W4A16 baseline**:
The immutable reference experiment that defines the target hardware, backend, model, workloads, graph boundaries, and profiling protocol used for comparison.
_Avoid_: Current W4A16 mode, runtime W4A16 option

**Experimental W4A8 baseline**:
A correct, runnable hardware-speed baseline that preserves W4G32 weights and replaces the target activations with per-tensor asymmetric UInt8. It has no accuracy or speed threshold beyond the sanity-correct run and comparable profile requirements.
_Avoid_: Accuracy-qualified W4A8, optimized W4A8

**Target activation**:
An activation that the archived W4A16 baseline explicitly represents as per-tensor asymmetric UInt16 and that the experimental W4A8 baseline therefore represents as per-tensor asymmetric UInt8. Activations outside this set retain their archived-baseline precision.
_Avoid_: Every activation, KV-cache activation

**Activation-only down-quantization**:
The controlled change from UInt16 to UInt8 for target activations while preserving W4G32 weight quantization, the existing graph and conversion framework, and all non-target tensor precisions. It excludes accuracy-compensation algorithms and unrelated quantization changes.
_Avoid_: W4A8 redesign, mixed-precision optimization

**Pinned A8 calibration**:
The existing mllm activation-calibration method run directly in asymmetric 8-bit mode over one immutable, locally stored set of calibration inputs and token IDs. The data identity is pinned by digest so reruns do not depend on a changing online dataset. It is baseline construction, not accuracy optimization.
_Avoid_: A16 qparam compression, live-dataset calibration, A8 clipping optimization

**Clean-room W4A8 derivation**:
An experimental W4A8 baseline derived only from the archived W4A16 baseline and general QNN/QAIRT behavior. Prior W4A8 attempts and their code, documents, parameters, artifacts, logs, and results are outside the evidence base.
_Avoid_: Historical W4A8 continuation, prior-candidate recovery

**Source-to-context W4A8 pipeline**:
The reproducible clean-room transformation from the original full-precision model through a newly generated W4A8 `.mllm` parameter file to audited QNN context binaries and profiling evidence. A context-only patch or a reused historical W4A8 parameter file is not this pipeline.
_Avoid_: Context-only conversion, historical W4A8 parameter reuse

**Explicit W4A8 selection**:
The opt-in recipe choice that activates asymmetric UInt8 behavior in otherwise shared quantization infrastructure. Merely supporting UInt8 in the framework does not change another model or recipe.
_Avoid_: Global A8 default, implicit activation downgrade

**Paired Qwen3 AOT descriptions**:
The ordinary and split-head Qwen3 G32 graph descriptions that share the same W4A8 quantization contract. The split-head description is the authoritative build and acceptance path; the ordinary description is kept consistent to prevent a contradictory secondary path.
_Avoid_: Two W4A8 build targets, ordinary-path acceptance

**W4A8 implementation kind**:
The new internal named choice for W4A8G32 tensor layout and quantization behavior. It is selected only by the experimental Qwen3 G32 baseline and does not rename or change the existing W4A16 choice used by other models.
_Avoid_: Renamed W4A16 kind, global W4A8 switch

**Sanity-correct run**:
An experimental W4A8 execution that builds, loads, and completes with structurally valid, finite, repeatable outputs. It establishes implementation viability, not model accuracy.
_Avoid_: Accuracy pass, quality validation

**Comparable profile**:
A W4A8 performance result collected with the archived W4A16 baseline's existing profiling protocol and workloads, with enough breakdown to compare end-to-end and target-operation costs and explain observed bottlenecks.
_Avoid_: Theoretical speedup, isolated unpaired benchmark

**True W4A8 execution**:
Execution in which the target hardware backend actually runs the target W4G32 operations with per-tensor asymmetric UInt8 activations, as evidenced by optrace and, where necessary, compiled-graph metadata. Configuration names alone are not evidence.
_Avoid_: W4A8-configured execution, nominal W4A8

**Joint execution evidence**:
The combination of a compile-time quantization manifest and runtime s1/s32 optrace used to establish true W4A8 execution for every target operation. Either source alone is incomplete evidence.
_Avoid_: Optrace-only proof, configuration-only proof

**Isolated W4A8 artifact namespace**:
The dedicated model-artifact and result locations populated only by the clean-room W4A8 derivation. Archived W4A16 evidence is read-only, and prior W4A8 artifacts are neither inputs nor recovery sources.
_Avoid_: Shared candidate directory, device-restored artifact

**Added fallback**:
Any fallback introduced by the W4A8 change that moves a target operation to UInt16, floating point, CPU, or another non-target execution path. Pre-existing boundaries inherited unchanged from the archived W4A16 baseline are not added fallbacks.
_Avoid_: Compatibility path

**Documented SDK boundary conversion**:
An explicit HTP Convert required by a pinned QAIRT operator contract while preserving a non-target archived-baseline precision. For QAIRT 2.47 RmsNorm, this is the A8-to-A16 input and A16-to-A8 output bridge around retained UInt16 gamma/bias. It is measured and traced, and it must not conceal a target Linear/Conv2D fallback.
_Avoid_: Free conversion, implicit fallback

## Settled baseline outcome

The first formal experimental W4A8 baseline result is
`qwen3_sm8750_v79_w4a8g32_20260813_135938`. Its joint execution evidence passes
for both whole graphs: s1 and s32 each contain 1009 manifest target operations,
all 1009 are observed in runtime Optrace, and none has a physical UInt16 target
input or output.

Performance is informational. Against archived W4A16 result
`qwen3_sm8750_v79_g32_20260807_230410`, measured runner medians are 738.693
token/s prefill (-14.10%) and 37.632 token/s decode-after-first (-17.27%). The
dominant critical-path classes remain LPBQ weight streaming plus HMX compute in
the MLP, lm_head, and projection stages. Explicit SDK boundary conversions and
other HVX conversions add work, so activation-width reduction alone does not
produce a first-version speedup.

The 100-case sanity run is deliberately not an acceptance gate, but its outcome
is a material quality warning: 0/100 answers passed and 11 NUL bytes were
observed. The run completed and the result schema was parseable; this establishes
the requested experimental hardware baseline, not usable model quality.
