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

## US-002 — Upload input image for i2v and is2v pipelines

**Summary:** Added conditional image upload with thumbnail preview to the composer area. The `ImageUpload` component appears only when pipeline is `i2v`, `is2v`, or `ia2v`. It renders a hidden `<input type="file">` with `accept=".jpg,.jpeg,.png,.webp"`, shows a thumbnail preview (with object URL lifecycle managed by `useEffect`), and passes the `File` object up to `App`. The submit handler in `App` validates the image requirement and serialises everything as `multipart/form-data` to `POST /create/{mediaType}`.

**Key Decisions:**
- `FILE` object managed in `App.tsx` state (not inside `GenerationParams`) because `File` is not a serialisable param.
- `requiresInputImage()` utility added to `types.ts` so both component rendering and tests share the same predicate.
- Object URL lifecycle managed with `useEffect` (create on `value` change, revoke on cleanup) to avoid memory leaks.
- `__PARALLAX_API_URL__` global declared in `vite-env.d.ts` and also declared inline with `declare const` in `App.tsx` (both are needed for tsc and Vite runtime respectively).

**Pitfalls Encountered:**
- `global.fetch` does not exist in the happy-dom TypeScript environment; use `globalThis.fetch` or `vi.stubGlobal("fetch", ...)` for mocking.
- `URL.createObjectURL` is not implemented in happy-dom; must be mocked with `Object.defineProperty(globalThis, "URL", ...)` in `beforeEach`.
- Declare-only globals (like `__PARALLAX_API_URL__`) need to be declared in `vite-env.d.ts` for tsc, even though Vite's `define` replaces them at build/test time.

**Useful Context for Future Agents:**
- `IMAGE_REQUIRED_PIPELINES` in `types.ts` is the canonical source for which pipelines require an image input (`["i2v", "is2v", "ia2v"]`).
- `App.tsx` now owns `inputImage: File | null`, `prompt: string`, `imageError: string | null` state and a `handleSubmit` that builds `FormData`.
- The `ImageUpload` component is fully controlled: pass `value` + `onChange` + optional `error` prop; it handles the hidden input, preview, and remove button internally.
- `vi.stubGlobal` / `vi.unstubAllGlobals` is the correct pattern for mocking browser globals in Vitest + happy-dom.
