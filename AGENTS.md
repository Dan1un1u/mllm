# Project-memory worktree instructions

This worktree contains the authoritative control plane for the mllm clean-room
W4A8G32 project. It is not a source-code branch.

- Run `python3 scripts/project_memory.py validate` before reading project state.
- Fetch and fast-forward only; never rebase, force-push, or rewrite this branch.
- Treat `PROJECT_CONTRACT.md`, `PROJECT_STATUS.yaml`,
  `experiments/index.yaml`, accepted ADRs, and experiment records as
  authoritative.
- Treat `CONTEXT.md` strictly as a glossary.
- Do not add model files, compiled contexts, traces, build products, or source
  implementation to this branch.
- Modify structured YAML through `scripts/project_memory.py` after bootstrap.
- Only the user may amend the contract, accept an ADR, promote a baseline,
  accept an experiment, or authorize reopening a rejected direction.
- Any sync, schema, reference, required-evidence, hash, or lock failure is a hard
  stop. Preserve the state and discuss it with the user.
