#!/usr/bin/env bash
set -euo pipefail

MEMORY_WORKTREE=/home/daniuniu/work/mllm-w4a8-project-memory
MEMORY_BRANCH=codex/w4a8g32-project-memory
EXPECTED_ORIGIN=https://github.com/Dan1un1u/mllm.git
SOURCE_WORKTREE=${1:-}

hard_stop() {
    printf 'PROJECT_MEMORY_HARD_STOP=%s\n' "$*" >&2
    exit 2
}

[[ -d "$MEMORY_WORKTREE" ]] ||
    hard_stop "missing worktree: $MEMORY_WORKTREE"

actual_branch=$(git -C "$MEMORY_WORKTREE" symbolic-ref --quiet --short HEAD) ||
    hard_stop "project-memory is detached"
[[ "$actual_branch" == "$MEMORY_BRANCH" ]] ||
    hard_stop "wrong branch: $actual_branch"

actual_origin=$(git -C "$MEMORY_WORKTREE" remote get-url origin) ||
    hard_stop "origin is unavailable"
[[ "${actual_origin%.git}" == "${EXPECTED_ORIGIN%.git}" ]] ||
    hard_stop "unexpected origin: $actual_origin"

[[ -z "$(git -C "$MEMORY_WORKTREE" status --porcelain=v1 --untracked-files=all)" ]] ||
    hard_stop "project-memory worktree is dirty"

git -C "$MEMORY_WORKTREE" fetch origin +    "+refs/heads/$MEMORY_BRANCH:refs/remotes/origin/$MEMORY_BRANCH" ||
    hard_stop "fetch failed; stale fallback is forbidden"

local_head=$(git -C "$MEMORY_WORKTREE" rev-parse HEAD) ||
    hard_stop "local HEAD is unavailable"
remote_head=$(git -C "$MEMORY_WORKTREE" rev-parse "refs/remotes/origin/$MEMORY_BRANCH") ||
    hard_stop "remote project-memory branch is unavailable"

git -C "$MEMORY_WORKTREE" merge-base --is-ancestor "$local_head" "$remote_head" ||
    hard_stop "local and remote project-memory history diverged"

if [[ "$local_head" != "$remote_head" ]]; then
    git -C "$MEMORY_WORKTREE" merge --ff-only "refs/remotes/origin/$MEMORY_BRANCH" ||
        hard_stop "fast-forward failed"
fi

python3 "$MEMORY_WORKTREE/scripts/project_memory.py" validate ||
    hard_stop "quick validation failed"

if [[ -n "$SOURCE_WORKTREE" ]]; then
    python3 "$MEMORY_WORKTREE/scripts/project_memory.py" brief +        --source-worktree "$SOURCE_WORKTREE" ||
        hard_stop "source identity check failed"
else
    python3 "$MEMORY_WORKTREE/scripts/project_memory.py" brief ||
        hard_stop "brief generation failed"
fi
