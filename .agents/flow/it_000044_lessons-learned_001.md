# Lessons Learned — Iteration 000044

## US-001 — Define shared job types

**Summary:** Created `packages/parallax_sdk/src/jobs.ts` exporting `ParallaxJobData`, `ParallaxJobResult`, and `PythonProgress` interfaces, and re-exported them from `src/index.ts`.

**Key Decisions:** Added a dedicated `jobs.ts` module rather than appending to `types.ts`, keeping request/response HTTP types separate from job queue types. All fields from the ACs were typed with the appropriate primitives; optional fields (`frame`, `total`, `output`, `error` on `PythonProgress`) use the `?` modifier.

**Pitfalls Encountered:** None. The SDK package is straightforward — `bun run typecheck` delegates to `tsc --noEmit` and passes cleanly with zero config changes.

**Useful Context for Future Agents:** `@parallax/sdk` uses `"type": "module"` and points `"main"` directly at the TypeScript source (`./src/index.ts`). There is no build step — Bun resolves `.ts` directly. The typecheck script is `tsc --noEmit` defined in `package.json`. Re-export all new modules from `src/index.ts` to keep the public API surface in one place.
