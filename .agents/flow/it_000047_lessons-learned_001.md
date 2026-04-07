# Lessons Learned — Iteration 000047

## US-001 — Select media type and configure parameters

**Summary:** Implemented the React+Bun SPA foundation for the Parallax chat UI, covering the media type selector (image/video/audio) and the corresponding parameter panel (model, pipeline, width, height, duration). Created the full frontend scaffold in `frontend/` with TypeScript, Vite, and 35 passing Vitest tests.

**Key Decisions:**
- Used **Vitest** (not `bun test`) as the test runner because it reads `vite.config.ts` directly, enabling happy-dom and `globals: true` without extra configuration files.
- Placed all state in a custom `useGenerationParams` hook so the media-type reset logic (AC06) lives in one testable place, separate from rendering concerns.
- Used CSS Modules for component scoping to keep styles co-located without a heavy CSS-in-JS library.
- `data-testid` attributes on every interactive element and field container to make assertions deterministic and decoupled from visual copy.

**Pitfalls Encountered:**
- `bun test` does not read `vite.config.ts` test settings; calling `vitest run` directly is the correct command for this project.
- TypeScript needs `"types": ["vitest/globals"]` in `tsconfig.json` to resolve `describe`/`it`/`expect` as globals; otherwise `tsc --noEmit` reports TS2582 errors.
- CSS Modules need an explicit `*.module.css` ambient declaration (`vite-env.d.ts`) so `tsc` doesn't error on the imports.

**Useful Context for Future Agents:**
- Frontend lives at `frontend/` (Vite + React 18 + TypeScript). Dev command: `bun dev`; build: `bun build`; test: `bun run test` (runs `vitest run`); type check: `bun run lint` (runs `tsc --noEmit`).
- `PARALLAX_API_URL` build-time env var controls the server base URL (default `http://localhost:5000`), exposed via `__PARALLAX_API_URL__` in `vite.config.ts`.
- Models and pipeline options are centralised in `frontend/src/types.ts` — add or rename them there when the server exposes new models.
- The `ParameterPanel` is intentionally stateless — all state lives in `useGenerationParams`; pass `params` + `onUpdate` from the parent.
