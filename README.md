# mllm W4A8 Project Memory

This orphan branch is the authoritative control plane for the clean-room W4A8G32
research line. It is intentionally separate from every source branch.

## Authority

Read these files in order before project work:

1. `PROJECT_CONTRACT.md`
2. `PROJECT_STATUS.yaml`
3. `CONTEXT.md`
4. `experiments/index.yaml`
5. the active experiment record, if one exists

Chat summaries, source-branch roadmaps, and generated session briefs are
non-authoritative. A conflict between authoritative files is a hard failure.

## Commands

```bash
./scripts/bootstrap.sh /absolute/source/worktree
python3 scripts/project_memory.py validate
python3 scripts/project_memory.py validate --full
python3 scripts/project_memory.py brief --source-worktree /absolute/source/worktree
python3 scripts/project_memory.py preflight --source-worktree /absolute/source/worktree
```

Structured status transitions must use `scripts/project_memory.py`. Contract,
ADR, glossary, and experiment narrative changes still require ordinary review,
commit, and push.

## Storage boundary

This branch stores only small control-plane text and schemas. Source code stays
on source branches. Models and intermediate artifacts stay under
`D:\llm_exp\models`; formal results stay under `D:\llm_exp\results`.
