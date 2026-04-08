# Audit Report — Iteration 000047 · PRD-001

## Executive Summary

The PRD-001 implementation is largely complete and all 140 automated tests pass. All four user stories are fully covered: media type selection with conditional parameter panels, conditional image upload for i2v/is2v/ia2v pipelines, SSE-based progress streaming, and inline media playback with download. Two functional requirements have minor deviations: FR-3 and FR-4 both specify that requests without an image should be sent as `application/json`, but the implementation always uses `multipart/form-data` regardless of whether an image is present. FR-1 specifies "Bun as the bundler/dev server", but Vite is the actual bundler/dev server — Bun is used only as the package manager and script runner.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ⚠️ Partially comply | React 18 used correctly. Vite — not Bun — is the bundler/dev server. Bun is the package manager/script runner only. |
| FR-2 | ✅ Comply | All communication targets the configured base URL (default `http://localhost:5000`). No new routes introduced. |
| FR-3 | ⚠️ Partially comply | All required fields sent in the request body, but as FormData fields — not a JSON body — even when no image is attached. |
| FR-4 | ⚠️ Partially comply | Image correctly sent via multipart/form-data when present. When absent, still uses FormData instead of `application/json`. |
| FR-5 | ✅ Comply | Pure client-side SPA, no runtime Node/Bun process needed after build. |
| FR-6 | ✅ Comply | Vite configured with `outDir: 'dist'`, produces self-contained `index.html` + hashed assets. |
| FR-7 | ✅ Comply | `PARALLAX_API_URL` injected via `vite.config.ts` `define` block, defaulting to `http://localhost:5000`. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ Comply | All AC01–AC06 satisfied. MediaTypeSelector, ParameterPanel with conditional fields, sensible defaults (w=768, h=512, dur=5), pipeline reset on media type change. |
| US-002 | ✅ Comply | All AC01–AC05 satisfied. Conditional render via `requiresInputImage()`, correct accept types, thumbnail preview, FormData upload, inline validation error. |
| US-003 | ✅ Comply | All AC01–AC06 satisfied. Enter submits, Shift+Enter newline, user/assistant bubbles, SSE progress, connection-lost message, submit disabled while in-flight. |
| US-004 | ✅ Comply | All AC01–AC04 satisfied. Inline `<img>`, `<video>`, `<audio>` rendering, Download anchor with correct filename, error message on failure, in-memory history. |

---

## Minor Observations

1. **FR-3/FR-4 content-type gap** — `App.tsx` always sends `FormData` even when `inputImage` is null. If the server strictly validates `Content-Type` for non-image requests this will fail at runtime. Switch to `application/json` body when no image is attached.

2. **Bun vs Vite bundler** — Vite handles bundling; Bun is the runtime for npm scripts. Pragmatic and valid, but deviates from the literal FR-1 wording. No action required unless Bun-native bundling is a hard requirement.

3. **SSE stall timeout wording** — The 30-second stall timeout message ("No response from server for 30 seconds — generation may have stalled.") is distinct from the `onerror` message ("Connection lost — check job status."). This is an improvement but the timeout text is not specified in the PRD.

4. **ImageUpload accessibility** — The hidden `<input type="file">` is triggered via `inputRef.current?.click()` from a `<button>`. Some screen readers may not properly associate the button with the file input. Consider wrapping in a `<label>` element instead.

5. **Missing Shift+Enter regression test** — `handleKeyDown` correctly implements the behavior, but no automated test covers the Shift+Enter newline path.

---

## Conclusions and Recommendations

The implementation is production-ready for PRD-001 scope. All user stories comply and all 140 tests pass. The two key refactor items are:

1. **Fix FR-3/FR-4** — Conditionally switch the fetch body between `application/json` (no image) and `multipart/form-data` (with image). This aligns with the PRD contract and ensures server compatibility.
2. **Confirm FR-1 tooling choice** — Clarify whether Vite-over-Bun is the accepted decision. No code change required if Bun-as-package-manager is sufficient.

Low-priority improvements: ImageUpload label-based accessibility and Shift+Enter test coverage.

---

## Refactor Plan

### P1 · Fix content-type switching (FR-3, FR-4)

**File:** `frontend/src/App.tsx`

**Change:** In `handleSubmit`, replace the unconditional `FormData` approach with a conditional that:
- Sends `application/json` (with all params + prompt as JSON body) when `inputImage` is null.
- Sends `multipart/form-data` (FormData with all fields + the image file) when `inputImage` is present.

```ts
// Pseudocode
let body: BodyInit;
let headers: HeadersInit = {};
if (inputImage) {
  const fd = new FormData();
  fd.append("image", inputImage);
  // ... append all other fields
  body = fd;
} else {
  body = JSON.stringify({ media_type: params.mediaType, model: params.model, /* ... */ });
  headers["Content-Type"] = "application/json";
}
fetch(`${__PARALLAX_API_URL__}/create/${params.mediaType}`, { method: "POST", headers, body })
```

**Tests to update:** `App.test.tsx`, `HardenUS003.test.tsx` — assert that requests without an image use `application/json`.

### P2 · ImageUpload label-based trigger (minor accessibility)

**File:** `frontend/src/components/ImageUpload.tsx`

**Change:** Replace the `<button onClick={() => inputRef.current?.click()}>` trigger with a proper `<label htmlFor="image-upload-input">` wrapping pattern so the file input is directly associated.

### P3 · Add Shift+Enter regression test (coverage gap)

**File:** `frontend/src/__tests__/HardenUS003.test.tsx` or `App.test.tsx`

**Change:** Add a test that fires a Shift+Enter keydown event on the textarea and asserts that `handleSubmit` is NOT called and the textarea value is unchanged.
