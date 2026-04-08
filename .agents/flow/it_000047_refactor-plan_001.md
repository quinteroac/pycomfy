# Refactor Plan — Iteration 000047 — Pass 001

## Summary of changes

### R-001 — Content-type switching in `App.tsx`
Replaced the unconditional `FormData` body in `handleSubmit` with a conditional branch:
- **No image attached** (`inputImage === null`): builds a JSON string via `JSON.stringify` and sets `Content-Type: application/json` in fetch headers.
- **Image attached** (`inputImage !== null`): builds a `FormData` with all fields plus the image file — behaviour unchanged from before.

This resolves the FR-3 and FR-4 partial-comply findings: the server now receives `application/json` when no image is present, and `multipart/form-data` only when an image is included.

### R-002 — `ImageUpload` accessibility: label-based file trigger
Replaced the `<button onClick={() => inputRef.current?.click()}>` trigger with a `<label htmlFor={INPUT_ID}>` that natively associates with the hidden `<input id={INPUT_ID}>`. The `inputRef` is retained only for the `handleRemove` path that resets `inputRef.current.value`. The `data-testid="image-upload-trigger"` attribute is preserved on the label so all existing tests continue to pass. Screen readers can now correctly associate the trigger with the file input without relying on an imperative click.

### R-003 — Shift+Enter regression test
Added a new `describe` block `"US-003-AC01 — Shift+Enter newline (no submit)"` in `HardenUS003.test.tsx`. The test types `hello{shift>}{enter}{/shift}` into the prompt textarea and asserts that `fetch` is **not** called, confirming that `Shift+Enter` inserts a newline rather than triggering submission.

### Supporting test for R-001
Added `"AC04b: sends JSON body with Content-Type application/json for image media type (no image)"` to `App.test.tsx`. This test submits a prompt with the default `image` media type (no image file attached) and verifies that `fetch` receives a JSON string body and `Content-Type: application/json` header.

## Quality checks

| Check | Command | Outcome |
|---|---|---|
| Unit tests | `bun run test` (`vitest run`) | ✅ 142/142 tests passed across 11 test files |
| TypeScript typecheck | `bun run lint` (`tsc --noEmit`) | ✅ No errors |

No regressions introduced. The new tests added by this refactor pass as well (17 in `HardenUS003.test.tsx`, 13 in `App.test.tsx`).

## Deviations from refactor plan

None. All three refactor items (R-001, R-002, R-003) were fully implemented as specified in the audit JSON.
