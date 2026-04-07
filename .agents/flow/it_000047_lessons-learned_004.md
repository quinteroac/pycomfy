# Lessons Learned — Iteration 000047

## US-001 — Run design audit and resolve critical issues

**Summary:** Ran a full UI audit of the chat frontend (`App.tsx`, `ChatBubble`, `MediaTypeSelector`, `ParameterPanel`, `ImageUpload`), produced the report at `.agents/flow/it_000047_audit-report_001.md`, and resolved all 7 critical/high findings.

**Key Decisions:**

- **Accent colour change**: Changed `--accent` from `#8b5cf6` (~4.05:1 contrast with white, fails WCAG AA) to `#7c3aed` (~5.25:1, passes AA). Also introduced `--accent-rgb: 124, 58, 237` as a CSS custom property so all `rgba(…)` usages across four CSS files could be updated consistently via `rgba(var(--accent-rgb), 0.x)` — avoids future drift.
- **`role="radiogroup"` vs `role="group"`**: For a set of `role="radio"` buttons, the correct container is `role="radiogroup"`. This broke the existing `MediaTypeSelector.test.tsx` assertion (`getByRole("group")`) which was also updated.
- **Remove-button hit area**: The 20×20px ✕ button was expanded to a 44×44px touch target using `padding: 12px; margin: -12px; box-sizing: content-box` — this keeps the visual size intact while expanding the interactive area without affecting surrounding layout.
- **`role="log"` for chat messages**: Instead of `aria-live="polite"` + `aria-atomic="false"`, used `role="log"` which implies both, and is semantically correct for a chat timeline.
- **ARIA on progress bar**: Added `role="progressbar"` with `aria-valuenow`, `aria-valuemin`, `aria-valuemax`, and `aria-label` to the fill div inside ChatBubble. The outer wrapper div is not the progressbar — it's the fill div that carries the ARIA role.

**Pitfalls Encountered:**

- The existing `MediaTypeSelector.test.tsx` used `getByRole("group")` — this fails after the fix to `role="radiogroup"` because testing-library's `getByRole` does not query `radiogroup` as `group` (despite ARIA inheritance). The test needed updating.
- `rgba(139, 92, 246, …)` was hardcoded in **four** CSS files (App.css, MediaTypeSelector.module.css, ParameterPanel.module.css, ImageUpload.module.css). Centralising via `--accent-rgb` took care of all of them.
- The `composer-send` hover colour was previously `#7c3aed` (the old accent) and needed darkening to `#6d28d9` after the accent change so hover still shows a visible state change.

**Useful Context for Future Agents:**

- `--accent-rgb` CSS variable now exists in `:root` (App.css) for use in `rgba(var(--accent-rgb), alpha)` — use it in any new CSS that needs a transparent variant of the accent colour.
- All interactive elements in the existing components already have `aria-label` set — new components should follow this pattern.
- The `role="log"` + `aria-label="Chat messages"` is on the `.chat-messages` div inside `App.tsx` (only rendered when messages exist — the empty-state div does not have it).
- Medium/low issues from the audit remain open: no responsive breakpoints, `field-sizing: content` compatibility, video/audio accessible labels, colour-only error states, skip link. These are documented in the audit report for future iterations.
- Frontend tests run via `bun run test` in `frontend/`; TypeScript check via `bun run lint`.
