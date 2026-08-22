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

**Recipe-induced RMSNorm bridge**:
The explicit HTP A8-to-A16 input and A16-to-A8 output conversion around RmsNorm that preserves the baseline's UInt16 gamma and bias recipe. QAIRT 2.47 also exposes a native UInt8 RmsNorm configuration, so this bridge is a property of the baseline recipe rather than an SDK requirement.
_Avoid_: Required SDK conversion, free conversion, implicit fallback

**Native U8 RMSNorm experiment**:
An isolated derivative of the experimental W4A8 baseline that changes only RmsNorm parameter precision as required to use the hardware backend's native UInt8 input/output configuration and remove the recipe-induced RMSNorm bridges. All non-RmsNorm recipes, graph behavior, workloads, and profiling conditions remain fixed.
_Avoid_: W4A8 redesign, general mixed-precision optimization

**Zero-bias RMSNorm variant**:
One native U8 RMSNorm experiment variant distinguished only by the storage and quantization contract of the synthetic all-zero bias. Its bias has no learned information, so UInt8 and symmetric Int32 variants differ as backend configurations rather than as model-precision alternatives.
_Avoid_: Learned bias variant, higher-precision model bias

**Native U8 RMSNorm zero bias**:
The synthetic all-zero RmsNorm bias represented as asymmetric UInt8 with an encoding that preserves exact real zero. It is the sole bias configuration in the native U8 RMSNorm experiment; a wider bias is excluded because this model has no learned RmsNorm bias to preserve.
_Avoid_: Int32 precision variant, learned RMSNorm bias

**U8 Q/K normalization-to-RoPE path**:
The Qwen3 attention path in which each Q/K head RmsNorm consumes and produces asymmetric UInt8 and feeds the existing UInt8 RoPE decomposition directly. RoPE remains unchanged; eliminating the RmsNorm UInt16 island removes rather than relocates the precision conversion.
_Avoid_: UInt16 RoPE path, RmsNorm-only U8 island

**Native WSL build workspace**:
The Linux-filesystem workspace used for compilation, model transformation, context generation, temporary logs, and other small-file-intensive processing. It is disposable working state, not the artifact archive.
_Avoid_: D-drive build tree, archived runtime build

**Published experiment artifact**:
A completed model, context binary, or compact evidence bundle copied from the native WSL build workspace into its immutable experiment namespace under `D:\llm_exp`, with source and destination digests verified. Partial or failed working state is not a published artifact.
_Avoid_: Build cache, staging directory, unverified copy

**Custom masked E2Softmax experiment**:
An isolated derivative of the native U8 RMSNorm experiment that replaces only the existing attention masking and Softmax sequence with a U8-input/U8-output, base-2 integer approximation implemented as an HTP custom operation. Its placement, numerical validity, model quality, and performance are independent verdicts.
_Avoid_: Accepted Softmax baseline, model-accurate Softmax

**Interior zero-DRAM Softmax contract**:
The requirement that every real QK-to-probability-to-PV attention edge stays in Crouton VTCM with no MainMemory fallback, boundary VTCM conversion, compiler spill/fill, or custom-operation DRAM read/write for both s1 and s32. A graph-output numerical fixture is outside this placement measurement.
_Avoid_: Low-DRAM Softmax, micrograph-only placement proof

**Masked E2Softmax implementation validity**:
Evidence that device U8 outputs exactly match the independent integer contract and satisfy masking, nonzero-row, normalization, and top-1 invariants for both s1 and s32 over the full set of compiled score encodings. It does not imply acceptable floating-point approximation error or end-model quality.
_Avoid_: Accuracy pass, usable generation quality
