<!-- meristem-template:v1 -->
# Active Work

## Outcome

Deliver a compact expressive treatment for the revived studio while preserving generation, library, evaluation, and research boundaries. Human-reference reranker quality validation remains the separate product follow-up.

## Why it matters

The chart workspace needs usable viewport space, readable controls, tactile feedback, and clear state cues. Separately, lower reference distance does not establish musicality, playability, coherence, or fun for players.

## Current state

- **Branch context:** `revive-runtime` is three commits beyond `origin/main`: `274ba81` revives runtime/mapping workflows; `c57945d` records mapper-style analysis; `4c423c3` adds human-reference candidate reranking. These are committed history, not unfinished working-tree implementation.
- **Current working tree:** at context curation, no changes are staged. Tracked unstaged changes are the bounded `.gitignore` privacy block and studio/editor HTML and JavaScript changes. Untracked work is the Meristem substrate/entrypoint, this continuation file, and `web/static/css/expressive.css`. Editor tools move to a compact bottom edge; resize redraw and scrolled hit-testing are corrected. Recover the current classification with `git status --short --branch` and `git diff --cached --stat` before staging or continuation.
- **Verification state:** the preceding UI pass ran Chromium against the live FastAPI server across all six studio views at desktop and two phone widths, responsive editor layout, native keyboard menu/slider interaction, selected difficulty/tool semantics, slider value feedback, visible error toasts, reduced motion, and scrolled-note hit-testing. Configure/evaluation content used local UI fixtures; no generation/training/save/rating was submitted. These browser checks are preceding UI-task evidence, not rerun by this context-curation step. Prior reranker evidence remains in `docs/reference-reranker.md` and `docs/wolfram-mapper-analysis.md` and was not rerun.
- **Interpretation / acceptance:** applying the expressive-interface invocation to the existing studio was the agent's inferred task scope. The resulting design is implemented and checked, but has not been explicitly accepted by the user. Current composition choices are reviewable implementation choices, not new universal preferences or an irreversible project design decision.

## Locked decisions

- Production SoftChart and research transformer remain separate.
- Reference-guided best-of-four selection keeps the deterministic baseline eligible; missing/unusable references return that baseline.
- Human references are calibration/evaluation inputs by default, not automatically added to research training.
- Reference-distance scores are selection heuristics, not human quality ratings.
- Preserve the established uv/FastAPI/SQLite and plain HTML/CSS/JavaScript stack for this visual pass. New-project React/TypeScript/Vite defaults do not imply a migration here.

## Constraints / do-not-regress

Follow `.github/copilot-instructions.md` and `.meristem/PROJECT.md`, including checkpoint safety, data preservation, pinned vendor provenance, and excluding runtime/user assets from Git. Follow the current host's explicit-authorization rule for commits/pushes.

## Current objective

The expressive UI pass is implemented and browser-checked. The next useful acceptance check is to review the treatment with actual chart content and normal network resources; further design scope is not settled. The documented next product validation remains human A/B playtesting of greedy versus reference-selected charts, comparing persisted scores with player ratings.

## Verification

Substrate adoption: initializer completed; doctor reports `ready` with zero missing files and zero warnings. A repeated initialization dry-run preserves every file. Git checks confirm the private local lane is ignored, canonical documents remain visible, existing instruction/project/package files and original ignore rules are preserved, and no changes are staged. These establish substrate readiness, not application runtime correctness.

UI verification: JavaScript syntax checks for both clients and `git diff --check` passed. Browser checks used 1440px, 390px, and 320px studio viewports with no document overflow, and 1440px/390px editor viewports. Screenshots were reviewed for evaluation and editor geometry. Slider interaction height is 48px; CSS and screenshot inspection establish matching 20px rail/puck geometry. Representative text and control-border token contrast was calculated; this is not a complete accessibility audit.

External font and Socket.IO CDN requests were blocked in the browser checks to exercise resource-unavailable rendering. System font fallback was verified; Bricolage rendering and live Socket.IO delivery were not. Existing inline icons were retained. Browser processes were stopped after verification.

Backend suite, route audit, fresh generation/export/client smoke flow, physical touch-device behavior, and human A/B testing were not run for the UI pass. Established application commands remain `uv run pytest -q`, `uv run python tools/route_audit.py`, and the relevant real generation/export/client smoke flow.

## Open uncertainty

- Whether lower reference distance predicts better player experience remains unverified.
- Existing consolidation/checklist and legacy packaging conflicts are indexed in `.meristem/PROJECT.md`; reconcile before selecting work from those checklists.
- Full network-resource rendering and populated-state review with real generated charts remain useful visual checks; no backend/API/storage migration is part of this UI work.
- The temporary browser probe and screenshots are session artifacts, not a committed regression suite. Recreate the relevant browser checks when changing the editor or responsive interaction behavior; syntax checks alone cannot establish canvas correctness.
