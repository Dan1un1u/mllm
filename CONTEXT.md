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

**MLP LPBQ operator-expression experiment**:
An isolated derivative of the native U8 RMSNorm experiment that changes only the QNN operation expression of the model's MLP gate, up, and down projections while preserving their deployed W4G32 values, activation encodings, and surrounding computation. Attention projections, the language-model head, normalization, and other operators remain outside its target set.
_Avoid_: Full-model Linear migration, MLP requantization, general LPBQ redesign

**MLP operator-expression candidate**:
A FullyConnected or MatMul representation of the same quantized MLP projection currently represented by Conv2D. Candidate identity describes the backend expression, not a different weight or activation quantization scheme.
_Avoid_: New quantization recipe, alternative MLP model

**Uniform MLP lowering**:
One operator-expression candidate used consistently for gate, up, and down projections in both single-token and multi-token graphs. A mixture chosen separately by shape or graph is outside the initial experiment.
_Avoid_: Per-shape winner, mixed FC/MatMul MLP

**MLP advancement gate**:
The paired operator-level and complete-single-layer checkpoint that a uniform MLP lowering must pass before full-model construction. It requires compatible tensor shapes, equivalent deployed mathematics, target-backend execution, and independently measured non-regression for every target projection shape and graph length.
_Avoid_: Compile-only gate, aggregate-speed gate, full-model-first trial

**Measured MLP non-regression**:
A candidate result no more than one percent slower than the paired Conv2D reference under the same profiling conditions. Each target shape and graph length is judged independently; improvements in one case cannot hide regression in another.
_Avoid_: Average non-regression, unpaired speed estimate, theoretical speedup

**Accepted MLP iteration**:
A full-model MLP LPBQ operator-expression result that retains the advancement-gate contracts and also avoids more than one percent regression in both end-to-end prefill and decode performance against the native U8 RMSNorm reference. A locally faster candidate that fails this full-model check remains a failed experiment.
_Avoid_: Micrograph-only success, partially accepted migration

**Adapter-free MLP boundary**:
The unchanged logical MLP projection interface whose shape and quantization encoding match the Conv2D reference. Metadata-only views may express an operator's rank convention, but physical copies, transposes, format changes, or datatype conversions are not part of this boundary.
_Avoid_: Layout-adapted projection, conversion-tolerant boundary

**Deployed-equivalent MLP projection**:
An operator-expression candidate whose operation-specific weight layout decodes exactly to the reference projection's signed W4G32 matrix and whose activation encodings are identical to the reference. Backend rounding may differ within the agreed output tolerance, but weight values, scales, zero-points, and logical tensor shapes may not.
_Avoid_: Requantized projection, approximately matched weights

**Reference-relative MLP correctness**:
Numerical correctness measured against a high-precision computation using the exact deployed weights and activation encodings, with the Conv2D expression serving as the accepted error baseline. A candidate may differ in backend rounding but may not worsen maximum, tail, or aggregate error by more than one output quantization step.
_Avoid_: Bit-identical operator output, accuracy-benchmark gate

**Paired MLP timing**:
Profiling-off device timing in which each Conv2D reference and operator-expression candidate uses the same inputs, thermal controls, warmup, repetition count, and fresh-process protocol. Runtime Optrace is separate execution evidence and diagnostic evidence rather than the sole speed measurement.
_Avoid_: Optrace-only timing, unpaired benchmark

**MLP candidate winner**:
The gate-passing uniform MLP lowering with the best worst-case normalized latency across the target projection shapes and graph lengths. When candidates are indistinguishable within the measurement band, FullyConnected is preferred for its direct projection semantics.
_Avoid_: Average-case winner, per-shape winner

**Representative middle-layer MLP**:
A preselected Qwen3 layer-14 MLP used for the complete-single-layer advancement check so that neither model entry nor model exit behavior determines the result. Its identity is fixed before candidate timing begins.
_Avoid_: First-layer gate, last-layer gate, best-performing layer

**Warm MLP timing**:
Paired steady-state timing collected after the target projection or MLP has executed enough times to remove initialization and warmup effects. It characterizes the resident kernel path but does not represent first access to previously untouched weights.
_Avoid_: Streaming-weight timing, first-execution timing

**Cold-weight MLP timing**:
Paired first-execution timing collected in fresh processes after the device is primed by an unrelated graph but before the target weights have been used. It is the experiment's proxy for the layer-to-layer weight streaming seen in the full model.
_Avoid_: Device-startup timing, warmed-weight timing

**Single-layer MLP integration check**:
The complete-single-layer advancement check comprising both an independently timed layer-14 MLP and an otherwise unchanged full graph in which only layer 14 uses the candidate expression. The full graph establishes real-neighbor compatibility; its end-to-end delta is informational because one changed layer is below the experiment's whole-model resolution.
_Avoid_: Standalone-operator-only gate, one-layer full-model speed gate

**Operator-aware LPBQ layout**:
An explicit mapping from each QNN projection expression to its weight channel and block axes. Conv2D, FullyConnected, and MatMul retain distinct physical layouts while decoding to the same logical W4G32 projection.
_Avoid_: Rank-relative axis guessing, shared physical weight layout

**Candidate-local advancement**:
Independent staged evaluation of each MLP operator-expression candidate from host correctness through hardware timing and single-layer integration. Failure removes only that candidate; full-model work stops when no uniform candidate remains or the selected candidate fails the complete-single-layer gate.
_Avoid_: First-candidate-wins, shared candidate gate, benchmark-before-correctness

**Layout-derived MLP artifact**:
An experiment model derived from the immutable native U8 RMSNorm artifact solely by changing the physical carrier layout of target MLP weights. Its canonical decoded weights and all activation encodings remain identical to its source artifact.
_Avoid_: Recalibrated MLP artifact, requantized candidate model

**Pinned middle-layer activation fixture**:
A digest-identified set of single-token and multi-token UInt8 activations generated for layer 14 from the experiment's fixed input data and stored outside source control. It provides real-value inputs for paired correctness and speed measurements.
_Avoid_: Random-only performance input, untracked activation dump

**Full-model MLP structural audit**:
The joint manifest and runtime check that every MLP projection uses the selected operator expression while all non-MLP projection, native U8 RMSNorm, W4G32, and activation contracts remain unchanged and no new conversion or fallback is present.
_Avoid_: Operator-count spot check, configuration-only audit

**Compact failed-gate evidence**:
The small, reproducible summaries, metadata, and decisive logs retained when a candidate or experiment fails a hard gate. Large contexts, raw traces, temporary models, and build caches are not retained for a failed experiment.
_Avoid_: No failure record, archived failed build tree

**Six-case MLP gate matrix**:
The independent cold-weight and warm timing cases formed by gate, up, and down projections in both single-token and multi-token graphs. Gate and up remain separate cases despite sharing a shape because their weights and activation encodings differ.
_Avoid_: Shape-only gate matrix, combined gate/up result

**Same-session full-model gate**:
The interleaved paired comparison between the native U8 RMSNorm reference and the full MLP candidate used to resolve the one-percent end-to-end threshold. Its paired result governs acceptance while the standard three-round summary preserves comparability with archived profiles.
_Avoid_: Archived-result-only gate, unpaired full-model timing

**Invalid MLP timing set**:
A timing set rejected because of abnormal thermal state, excessive process-to-process dispersion, or inconsistent paired direction. It may be recollected once after cooldown; repeated instability is a stop condition rather than permission to sample until a pass appears.
_Avoid_: Slow outlier deletion, retry-until-pass timing

**Front-end-only LPBQ MatMul lowering**:
An LPBQ graph whose pre-finalize QNN expression is MatMul but whose finalized V79 schematic uses the same ConvLayer, W4-to-Int8 expansion, weight-to-VTCM, and HMX physical families as the Conv2D reference. The front-end operator name alone is not evidence of a distinct or faster hardware kernel.
_Avoid_: Native MatMul LPBQ kernel, MatMul speedup

**Disqualified LPBQ MLP candidate**:
An operator-expression candidate removed from advancement after any independent real-shape gate fails. A favorable result in another shape may not be aggregated with it, and no mixed winner or full-model migration is inferred without a separately accepted experiment.
_Avoid_: Aggregate MLP pass, partial-shape winner

**MLP gate report**:
The compact comparison record that joins six-case cold and warm latency with physical weight-expansion, memory-traffic, DMA-wait, HMX, kernel-selection, and pass-state evidence. It accompanies rather than replaces the canonical end-to-end critical-path report.
_Avoid_: Latency-only summary, critical-path-only report

## Clean-room implementation status

The native-U8 RMSNorm experiment is implemented as a clean-room derivative of
the archived W4A16 path. No historical W4A8 code, artifact, log, or performance
result is an input to this implementation. The source-to-context build from
`Qwen3-origin` completes for both split-head graphs; the published manifests
contain 1009 W4G32 LPBQ targets and 729 native-U8 RmsNorm operations per graph,
with zero recipe-induced RMSNorm bridges.

Runtime evidence is now complete in
`D:\\llm_exp\\results\\qwen3_sm8750_v79_w4a8_rmsnorm_u8_20260813_220228`.
The Windows adb executable (`C:\\adb\\adb.exe`) addressed the connected
`PJZ110` device explicitly by serial; the separate old WSL adb server had no
device. Three profiling-off runner rounds, the 100-case informational sanity
suite, and fresh-process s1/s32 Optrace all completed. The measured medians
were 788.252 prefill tokens/s and 37.942 decode tokens/s after first token.
Against the requested W4A16 reference
`qwen3_sm8750_v79_g32_20260807_230410`, these are -8.34% and -16.59%; this
experiment has no speed or accuracy gate.

The joint runtime audit passes for both graphs: 1009/1009 target operations
were observed, 729/729 RmsNorm operations were traced, all target RmsNorm
operations were physical U8 with zero U16 RmsNorm operations and zero explicit
RmsNorm bridges. The canonical report is
`qwen3-sm8750-v79-g32-e2e-critical-path.html` in the result directory. Large
QAIRT viewer inputs/outputs are decoded in the native WSL workspace and then
copied back to the D-drive result namespace; raw Optrace and compact evidence
remain archived under `D:\\llm_exp`.
