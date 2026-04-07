# Frontend Design Refinement — Impeccable Pass

## Context

Once the chat UI (PRD 001) is functional, it will be a working but unstyled prototype. The target aesthetic is **clean and elegant**: restrained palette, generous whitespace, minimal chrome — the media output is the hero, not the UI itself. This PRD covers a structured design refinement pass using the impeccable skill suite: audit → distill → harden → polish → animate. The goal is to elevate the frontend from functional to production-grade before the iteration is approved.

## Goals

- Identify and fix design, accessibility, and resilience issues surfaced by the audit skill.
- Strip unnecessary visual complexity and align to a clean, focused chat aesthetic.
- Harden the interface against edge cases (long prompts, slow connections, failed jobs, empty states).
- Apply final polish and purposeful micro-interactions that improve perceived responsiveness.

## User Stories

### US-001: Run design audit and resolve critical issues
**As a** developer, **I want** to run the `audit` skill against the chat UI **so that** I have a prioritised list of design, accessibility, and UX issues to address before shipping.

**Acceptance Criteria:**
- [ ] The `audit` skill is run against the frontend and produces a report covering: visual hierarchy, accessibility (contrast, keyboard nav, ARIA), responsive behaviour, and error states.
- [ ] All issues rated `critical` or `high` in the audit report are resolved before this PRD is considered complete.
- [ ] The audit report is saved to `.agents/flow/it_000047_audit-report_001.md`.

### US-002: Distill the interface to its essential structure
**As a** developer, **I want** to run the `distill` skill on the chat UI **so that** unnecessary visual noise is removed and the interface communicates its purpose clearly.

**Acceptance Criteria:**
- [ ] The `distill` skill is applied and its recommendations are reviewed; accepted changes are implemented.
- [ ] After distillation, the chat input, media type selector, and parameter panel are visually distinct without competing for attention.
- [ ] No decorative elements remain that do not serve a functional or communicative purpose.

### US-003: Harden the interface against edge cases
**As a** developer, **I want** to run the `harden` skill on the chat UI **so that** the interface handles failure modes, slow responses, and unexpected input gracefully.

**Acceptance Criteria:**
- [ ] Prompts longer than 500 characters are handled without layout breaking (text wraps or truncates with ellipsis in the bubble).
- [ ] If SSE progress events stop arriving for > 30 seconds, the UI shows a timeout warning inside the assistant bubble.
- [ ] Empty state (no messages yet) shows a helpful placeholder that explains what the user can do.
- [ ] All interactive controls have visible focus states for keyboard navigation.
- [ ] Error messages from the server are displayed in full inside the assistant bubble, not swallowed silently.

### US-004: Apply final polish and micro-interactions
**As a** developer, **I want** to run the `polish` and `animate` skills on the chat UI **so that** the interface feels refined and responsive to user actions.

**Acceptance Criteria:**
- [ ] The `polish` skill is applied; spacing, alignment, and typographic inconsistencies flagged by the skill are resolved.
- [ ] The `animate` skill is applied; accepted animations are implemented (e.g. bubble entrance, progress pulse, media reveal).
- [ ] All animations respect `prefers-reduced-motion` — they are disabled or minimised when the media query matches.
- [ ] No animation blocks the user from interacting with the interface.

---

## Functional Requirements

- FR-1: Skills are applied in this order: `audit` → `distill` → `harden` → `polish` → `animate`.
- FR-2: Each skill's recommendations are reviewed before applying — not all suggestions need to be accepted, but rejections must be noted.
- FR-3: `teach-impeccable` must be run once before the audit pass to establish design context for the project if it has not been run already.
- FR-4: No new runtime dependencies may be introduced during this pass — only CSS/JS changes to the existing React app.
- FR-5: The frontend must still build cleanly with `bun run build` after all refinements are applied.

## Non-Goals

- Full WCAG 2.1 AA compliance audit (best-effort only in this iteration).
- Dark/light theme implementation.
- Mobile-responsive layout.
- Rebranding or logo design.

## Open Questions

- None.
