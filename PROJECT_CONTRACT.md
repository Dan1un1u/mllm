# W4A8G32 Project Contract

This contract is authoritative. Changing any item requires explicit user
approval and a committed contract amendment.

## Provenance and scope

- **PC-001 — Clean-room provenance.** New W4A8 work derives independently from
  the archived W4A16 rotation line. Do not inspect or use Legacy W4A8-specific
  diffs, documents, parameters, artifacts, or results as evidence. General
  ancestral framework source remains usable. The exclusion boundary is recorded
  in `docs/exclusions/EXCLUDED-001.md`.
- **PC-002 — Weight contract.** The baseline weight contract is signed W4 with
  LPBQ group size 32. Do not change the weight quantization algorithm when the
  experiment claims activation-only attribution.
- **PC-003 — Activation contract.** Activations use asymmetric U8. Symmetric S8
  is outside the current contract.
- **PC-004 — Single-variable attribution.** Every experiment declares one
  primary physical variable. Undeclared algorithm, layout, precision, toolchain,
  or output-contract changes invalidate attribution.

## Hardware and software

- **PC-005 — Target.** PJZ110 / SM8750, Qualcomm HTP V79, QNN AOT.
- **PC-006 — Toolchain.** Active work uses QAIRT `2.49.0.260730`. QAIRT 2.47 is
  historical evidence only. Any other SDK requires a contract amendment and an
  isolated environment; never replace the retained SDK in place.
- **PC-007 — Device scope.** Use a project-specific directory below
  `/data/local/tmp`; do not inspect unrelated device artifacts.
- **PC-008 — No fallback.** Formal hardware evidence must show the intended HTP
  execution and disclose any CPU fallback.

## Storage and source control

- **PC-009 — Git boundary.** Git stores source and compact evidence only. Build
  products, models, contexts, traces, and profiling outputs are not committed.
- **PC-010 — Model boundary.** Models and intermediate artifacts use
  `D:\llm_exp\models`.
- **PC-011 — Result boundary.** Formal results use
  `D:\llm_exp\results`.
- **PC-012 — WSL execution.** Small-file-intensive work, builds, and runtime
  products stay on WSL ext4. Publish only retained large artifacts to the D:
  boundaries.
- **PC-013 — Source worktree.** Source work uses
  `/home/daniuniu/work/mllm-w4a8` and its registered Git worktrees. Project
  Memory uses its dedicated worktree and never carries implementation source.

## Evidence and comparison

- **PC-014 — Real W4A8 evidence.** Formal W4A8 claims require quantization
  manifests and Optrace evidence for the physical activation contract.
- **PC-015 — Profiling reuse.** Use the existing profiling components and retain
  raw evidence needed to compare with the registered comparator.
- **PC-016 — No mandatory speed or accuracy threshold.** An approved
  exploratory experiment may complete without net speedup and with only its
  declared numerical sanity check. It must still be mathematically implemented
  as declared and run successfully.
- **PC-017 — Separate tracks.** Report prefill and decode independently. A gain
  in one must not conceal a regression in the other.
- **PC-018 — Comparison key.** A speedup or regression claim requires matching
  device and firmware, HTP architecture, QAIRT release, model provenance,
  workload, runner protocol, warmup and repetitions, thermal acceptance, and
  schedule-search policy, except for the declared variable.
- **PC-019 — Fair scheduling.** W4A16 and W4A8 may use independently fastest
  schedules only under the same search budget and selection rule.
- **PC-020 — Compiler-only correctness.** A compiler scheduling change must
  preserve the declared upper-layer numerical contract before its performance
  result is accepted.

## Governance

- **PC-021 — Baseline authority.** Only the user may promote a Selected
  Baseline. A micrograph pass, local gate pass, faster branch, or completed run
  is not a baseline.
- **PC-022 — Four-axis state.** Execution state, evidence validity, local gate,
  and adoption status are independent fields.
- **PC-023 — Stateful preflight.** Stateful Work requires an approved immutable
  Experiment ID and a successful preflight.
- **PC-024 — Single running experiment.** At most one project-wide Experiment
  may have `execution_state: running`.
- **PC-025 — Rejected work.** Reopening rejected work requires its objective
  Reopen Condition and explicit user approval. Parameter tuning alone is not a
  new hypothesis.
- **PC-026 — Decision authority.** Codex may record observed facts, evidence,
  execution transitions, and a predeclared hard-gate failure. Contract changes,
  ADR acceptance, baseline promotion, `adoption_status: accepted`, and rejected
  track reopening require explicit user approval.
- **PC-027 — Sync hard stop.** Every project task must fetch, fast-forward only,
  and validate Project Memory first. Any failure immediately stops all project
  work for user discussion; stale local fallback is forbidden.
- **PC-028 — Branch-independent authority.** Source-branch copies of contracts,
  roadmaps, status, or `CONTEXT.md` are historical and non-authoritative.
- **PC-029 — Remote reachability.** A completed experiment record must resolve
  all required source commits and branches on the remote.
- **PC-030 — No automatic recovery.** Do not automatically stash, reset, clean,
  clear locks, rewrite history, force-push, delete tags, accept new hashes, or
  downgrade required evidence.
