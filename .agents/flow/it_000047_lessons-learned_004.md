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

## US-002 — Distill the interface to its essential structure

**Summary:** Applied the distill skill to the chat UI, removing decorative elements and visual noise. Changes focused on stripping ornamentation that didn't serve communication or function while preserving all interactive elements and accessibility attributes.

**Key Decisions:**

- **Removed `◈` logo glyph** from app header — purely decorative, no semantic value. The `Parallax` text title already identifies the app.
- **Removed `composer-divider`** — the 1px separator between `MediaTypeSelector` and `ParameterPanel` was redundant; flex layout + gap provides sufficient spatial separation.
- **Removed glow `box-shadow` from textarea focus** (`0 0 0 3px --accent-glow`) — a prominent purple ring is decorative bloom. A simple `border-color` change (already present) suffices to communicate focus state accessibly.
- **Removed `--accent-glow` CSS variable** — after removing all glow shadows it was unused; removing it prevents future accidental reuse of the pattern.
- **Removed glow `box-shadow` from active MediaType button** (`0 1px 6px rgba(accent-rgb, 0.4)`) — decorative emphasis that competed with the active background colour.
- **Removed `text-transform: uppercase` and `letter-spacing: 0.06em`** from ParameterPanel field labels — over-designed for utility labels. Labels are now plain text, making the parameters visually recessive (secondary priority) vs. the MediaTypeSelector and textarea.
- **Fixed stale hardcoded colour** `rgba(139, 92, 246, 0.6)` in textarea focus border-color → `rgba(var(--accent-rgb), 0.6)` to use the CSS variable set in US-001.
- **Trimmed empty-state copy** to "Type a prompt and press Generate to begin." — shorter, imperative, zero redundancy.

**Pitfalls Encountered:**

- `--accent-glow` was referenced only in `App.css` (`textarea:focus`). After removing that glow rule, the variable was completely unused; removing it is safe and reduces `:root` noise.
- The `composer-divider` CSS rules in `App.css` needed to be deleted alongside the HTML element removal — otherwise dead CSS would linger.

**Useful Context for Future Agents:**

- `:root` in `App.css` no longer has `--accent-glow`. Do not re-introduce it. The focus affordance pattern for this UI is `border-color: rgba(var(--accent-rgb), 0.6)` only — no box-shadow glow.
- ParameterPanel field labels are intentionally plain-cased and at 50% opacity (`rgba(255,255,255,0.5)`) to stay recessive relative to the active MediaTypeSelector pill and the textarea.
- `AC03` tests in `DistillUS002.test.tsx` assert the absence of `◈` and `.composer-divider` — if either is accidentally reintroduced, these tests will fail.

## US-003 — Harden the interface against edge cases

**Summary:** Implemented five resilience improvements to the chat UI: long-prompt wrapping (AC01), 30-second SSE inactivity timeout (AC02), testable empty-state placeholder (AC03), keyboard-focusable controls with visible focus indicators (AC04), and full server error message surfacing (AC05).

**Key Decisions:**

- **SSE timeout via captured timer ID**: The 30s timeout is implemented with a `sseTimeoutId` variable local to the EventSource setup closure. `armSseTimeout()` clears any existing timer and sets a new one; it is called once when the EventSource is created and again on every `onmessage` event (resetting the window). `clearSseTimeout()` is called on `done`, `error`, and `onerror` to cancel cleanly.
- **`"timeout"` added to `MessageStatus`**: Rather than reusing `"connection-lost"`, a distinct `"timeout"` status makes it easier to style and test independently. Both use the same amber warning colour in `ChatBubble.module.css`.
- **Server error body surfacing**: Changed the fetch error path to read `res.text()` and append the body to the error message. The `.catch(() => "")` guard ensures a failed body read doesn't cascade.
- **`data-testid="chat-empty"` for AC03**: Added to the existing empty-state div to enable reliable assertions.
- **`focus-visible` outline on textarea**: Added `.composer-textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }` after the `:focus` rule. Higher specificity of `:focus-visible` overrides `outline: none` for keyboard navigation only.

**Pitfalls Encountered:**

- **`tabIndex` in JSDOM**: Native `<button>` and `<textarea>` elements return `tabIndex === -1` in JSDOM without an explicit attribute, even though they are natively focusable in real browsers. Use `element.focus()` + `expect(document.activeElement).toBe(element)` instead.
- **setTimeout spy must pass through non-30s calls**: When mocking `setTimeout` to capture the 30s SSE timeout callback, all other delays must be forwarded to the original `setTimeout` to avoid breaking `userEvent` internals.
- **`clearTimeout` spy for fake timer ID**: When returning a fake timer ID from the mocked `setTimeout`, the corresponding `clearTimeout` spy must no-op for that fake ID to avoid passing an invalid ID to the real `clearTimeout`.

**Useful Context for Future Agents:**

- `MessageStatus` in `types.ts` now includes `"timeout"`. Style it in `ChatBubble.module.css` via `.status-timeout .bubble-content`.
- The `armSseTimeout` / `clearSseTimeout` pattern lives entirely within the `handleSubmit` closure — no separate ref in component scope.
- The `.composer-textarea:focus-visible` rule in `App.css` sits after the `:focus` rule; do not reorder.
- `data-testid="chat-empty"` is on the empty-state wrapper in `App.tsx` (conditionally rendered — absent when messages exist).
- All 119 frontend tests pass; TypeScript compiles clean (`bun run lint`).
