---
status: accepted
---

# Keep project governance on an independent orphan branch

The project uses `codex/w4a8g32-project-memory` as an orphan, metadata-only
control plane instead of copying evolving constraints into every experiment
branch. This avoids divergent contracts across worktrees, at the cost of a
mandatory external bootstrap and a two-phase source/evidence closeout.
