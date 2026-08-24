# mllm W4A8 project-memory bootstrap

For any task in a Git worktree whose common directory is
/home/daniuniu/work/mllm-w4a8/.git and whose origin is
https://github.com/Dan1un1u/mllm.git:

1. Before reading project source, discussing project theory, editing files,
   building, generating models, using the device, or profiling, run:
   /home/daniuniu/work/mllm-w4a8-project-memory/scripts/bootstrap.sh WORKTREE
2. Treat any fetch, fast-forward, schema, reference, evidence, hash, origin,
   branch, or lock failure as an immediate hard stop. Do not use a stale local
   copy. Preserve state and discuss the failure with the user.
3. Read the authority files in the order printed by project memory. Chat
   summaries and source-branch status files are non-authoritative.
4. Before any stateful work, run project_memory.py preflight with the approved
   Experiment ID and source worktree. Do not create or modify source, build,
   generate a model, run hardware, or profile without a successful preflight.
5. Never inspect Legacy W4A8-specific work identified by EXCLUDED-001.
6. Never auto-stash, reset, clean, clear a lock, force-push, rewrite history,
   delete a tag, accept a new artifact hash, or downgrade required evidence.
