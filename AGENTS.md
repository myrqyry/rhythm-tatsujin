# Agent Instructions

Read `.github/copilot-instructions.md` for shared project coding rules, data contracts, runtime boundaries, and validation commands. It remains the existing domain instruction authority. Commit and push actions require explicit user authorization under the current host's rules; its automatic checkpoint guidance does not override that requirement.

<!-- meristem:start -->
## Meristem project continuity

This repository uses Meristem project state.

Before non-trivial work:

1. Treat current repository/runtime evidence as the authority for what exists now.
2. Read `.meristem/PROJECT.md` for durable project purpose, invariants, and architecture boundaries when relevant.
3. If `ACTIVE_WORK.md` exists, read it for the current objective, locked decisions, constraints, verification state, and unresolved uncertainty.
4. Consult `.meristem/DECISIONS.md`, `LEARNINGS.md`, `ERRORS.md`, or `FEATURE_REQUESTS.md` only when they can materially change the current task.
5. Do not promote session detail into durable files merely because it is recent. Persist consequential state before dependent continuation when losing it would make later work expensive or misleading.
6. Preserve existing project conventions and working state. Do not broadly rewrite configuration or clean up unrelated code for convenience.
7. Verify user-visible outcomes before claiming completion.
8. Treat canonical `.meristem/*.md` files and `ACTIVE_WORK.md` as commit-safe repository knowledge. Keep secrets, personal continuity, raw conversations, and machine-specific state out of them; use `.meristem/local/` only for repo-local private/machine state.

Use the narrowest capable tool or specialist. Current source and runtime truth outrank stale continuation notes.
<!-- meristem:end -->
