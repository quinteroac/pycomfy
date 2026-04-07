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

## US-003 — Submit prompt and stream generation progress

**Summary:** Added chat timeline with user/assistant bubbles, Enter-to-submit keyboard shortcut, SSE progress streaming via `EventSource`, connection-lost handling, and in-flight disabled state. New files: `ChatBubble.tsx` + `ChatBubble.module.css`. Updated: `App.tsx`, `types.ts` (ChatMessage type), `App.css`. 13 new tests in `ChatStream.test.tsx`.

**Key Decisions:**
- `ChatMessage` type added to `types.ts` to keep all domain types in one place.
- `EventSource` used directly (not wrapped in a hook) since it's a single-use pattern inside `handleSubmit`.
- `isSubmitting` state disables both the textarea and button simultaneously; the `handleSubmit` guard (`if (isSubmitting) return`) provides a programmatic safety net regardless of the DOM disabled state.
- Assistant bubble is created with `status: "streaming"` immediately on submit (before the fetch resolves) so AC03 is satisfied.
- SSE `pct` values come as `0.0–1.0` floats from `PythonProgress`; multiplied by 100 for display.
- `esRef` stores the current `EventSource` so it can be closed on unmount cleanup.

**Pitfalls Encountered:**
- `EventSource` is not implemented in happy-dom; requires a hand-rolled `MockEventSource` class with `emit()` and `emitError()` helpers. Use `vi.stubGlobal("EventSource", MockEventSource)` to inject it.
- `act()` is required when calling `es.emit(...)` inside tests to flush React state updates synchronously.
- The counter-based `nextId()` helper at module level persists across test renders in the same file, which is fine as long as IDs are only used as React keys and `data-testid` values are role-based (`bubble-user`, `bubble-assistant`).

**Useful Context for Future Agents:**
- `App.tsx` now owns `messages: ChatMessage[]` and `isSubmitting: boolean` state.
- `ChatBubble` renders a `data-testid="bubble-user"` or `data-testid="bubble-assistant"` wrapper. Progress row is `data-testid="progress-row"` (only visible when `status === "streaming"`).
- SSE terminal events: `step="done"` → complete, `step="error"` → error. All other steps update the bubble with percentage progress.
- `MockEventSource.instances` is an array; always read the last instance in tests after `waitFor`.

## US-004 — Display generated media inline and allow download

**Summary:** Implemented inline media rendering (image/video/audio) and a Download button in the assistant chat bubble. Added `GET /jobs/{job_id}/result` server endpoint to serve the output file. Extended `ChatMessage` with `mediaUrl`, `mediaType`, and `filename` fields. When SSE `step="done"` fires, the media URL is constructed and stored on the message; `ChatBubble` renders the appropriate element plus a download anchor.

**Key Decisions:**
- Added `mediaUrl`, `mediaType`, and `filename` directly to `ChatMessage` in `types.ts` so the bubble component stays purely presentational (receives everything it needs via props).
- Media URL is constructed eagerly on `step="done"` without an additional fetch: `{apiUrl}/jobs/{jobId}/result`. This avoids an extra round-trip and matches the server pattern for all previously implemented endpoints.
- Server endpoint `GET /jobs/{job_id}/result` uses FastAPI's `FileResponse` with `Content-Disposition: attachment` to enable browser downloads even from cross-origin requests where the `download` attribute alone would not suffice.
- Added `mimetypes.guess_type` for automatic `Content-Type` inference from file extension.
- Download anchor uses `data-testid="download-btn"` and `download={filename}` attribute; the extension is derived from `params.mediaType` at submit time (`.png` / `.mp4` / `.wav`).

**Pitfalls Encountered:**
- The `FileResponse` import must be added alongside the existing `StreamingResponse` import — easy to miss since the original gateway only used streaming responses.
- `mimetypes` is a Python stdlib module but must be explicitly imported; it's not pulled in transitively.
- Route order in FastAPI matters: the new `/jobs/{job_id}/result` route must be placed **before** `/jobs/create/image` (which uses a fixed path prefix) to avoid route-matching conflicts if both are under the same router — in this case there's no conflict, but placing it near the other `/jobs/{job_id}/...` routes keeps the file readable.

**Useful Context for Future Agents:**
- `ChatMessage.mediaUrl` is set to `{apiUrl}/jobs/{jobId}/result` on `step="done"`. The endpoint streams the file from `output_path` stored in the job result JSON.
- `ChatBubble` renders media **outside** the `.bubble-content` div (in a sibling `.media-container` div) so the download button and media are visually separated from the text label.
- Test pattern for AC04 (in-memory persistence): render once, complete one job, wait for re-enable, then submit again and assert `getAllByTestId("bubble-user").length === 2`.
- `vi.spyOn(Storage.prototype, "setItem")` is the correct way to assert no `localStorage` writes in Vitest + happy-dom.
