<!-- meristem-template:v1 -->
# Learnings

Project-local reusable lessons. A learning should prevent future work or error, not merely narrate what happened.

Use entries like:

## YYYY-MM-DD — Learning

**Signal / failure:**  
**Mechanism:**  
**Scope:**  
**Evidence:**  
**Recommended behavior:**  
**Validation status:** observed | tested | unvalidated

## Editor canvas coordinates and resize lifecycle

**Signal / failure:** The prior editor added the scroll offset twice when hit-testing a scrolled canvas, and resize changed canvas dimensions without redrawing the chart.

**Mechanism:** `getBoundingClientRect().left` already incorporates the canvas's displacement inside its scrolled parent. `clientX - rect.left` is therefore the canvas-local coordinate; adding the parent's `scrollLeft` shifts the hit target incorrectly. Assigning a canvas's `width` or `height` clears its drawing surface, so resize must explicitly redraw. Recomputing width from the parent viewport and last note avoids depending on the previous canvas's scroll width.

**Scope:** `web/static/js/editor.js`; applies to this editor's CSS-pixel canvas coordinate model. Revisit the conversion if backing-store scaling or transforms are introduced.

**Evidence:** The uncommitted UI pass updates `resizeCanvas()` and `handleCanvasClick()`. A Chromium probe resized the actual editor canvas, scrolled its parent, and verified that clicking the existing note deleted it rather than adding a displaced note. Responsive editor screenshots showed the ruler after resize. Inspect current source and rerun the behavior before assuming a later revision retains the fix.

**Recommended behavior:** Preserve viewport-to-canvas conversion and redraw after dimension changes. Verify an existing-note click after horizontal scrolling and a resize with populated chart data; empty-canvas layout alone is insufficient.

**Validation status:** tested in the UI pass's temporary Chromium probe; no committed automated regression test currently covers this mechanism.
