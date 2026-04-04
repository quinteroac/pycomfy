# Lessons Learned — Iteration 000044

## US-001 — Define shared job types

**Summary:** Created `packages/parallax_sdk/src/jobs.ts` exporting `ParallaxJobData`, `ParallaxJobResult`, and `PythonProgress` interfaces, and re-exported them from `src/index.ts`.

**Key Decisions:** Added a dedicated `jobs.ts` module rather than appending to `types.ts`, keeping request/response HTTP types separate from job queue types. All fields from the ACs were typed with the appropriate primitives; optional fields (`frame`, `total`, `output`, `error` on `PythonProgress`) use the `?` modifier.

**Pitfalls Encountered:** None. The SDK package is straightforward — `bun run typecheck` delegates to `tsc --noEmit` and passes cleanly with zero config changes.

**Useful Context for Future Agents:** `@parallax/sdk` uses `"type": "module"` and points `"main"` directly at the TypeScript source (`./src/index.ts`). There is no build step — Bun resolves `.ts` directly. The typecheck script is `tsc --noEmit` defined in `package.json`. Re-export all new modules from `src/index.ts` to keep the public API surface in one place.

## US-002 — Bunqueue singleton in parallax_sdk

**Summary:** Created `packages/parallax_sdk/src/queue.ts` exporting `getQueue(): Queue` (with a `Bunqueue` type alias) — a lazy singleton backed by `~/.config/parallax/jobs.db` in embedded mode. Added `bunqueue` and `bun-types` as dependencies, and updated `tsconfig.json` to include `"types": ["bun-types"]`.

**Key Decisions:**
- Used `Queue` from `bunqueue/client` (not the all-in-one `Bunqueue` class). The `Bunqueue` class mandates a `processor`, `routes`, or `batch` option at construction time — incompatible with AC04 ("no processor, consumers attach their own"). `Queue` supports `embedded: true` with no processor requirement.
- Exported `Queue as Bunqueue` as a type alias so callers can import `Bunqueue` as a type name from the SDK even though the underlying class is `Queue`.
- Added `bun-types` devDependency and `"types": ["bun-types"]` to `tsconfig.json` because the SDK previously had no Node.js/Bun built-ins typed. Without this, `import os from "os"` and `import path from "path"` produce TS2307 errors even in Bun.

**Pitfalls Encountered:**
- `Bunqueue` class from `bunqueue/client` throws at runtime (`Bunqueue requires "processor", "routes", or "batch"`) when constructed with no handler — the AC assumption was incorrect for that class. Switch to `Queue` which is the producer-only interface.
- `@parallax/sdk` `tsconfig.json` had no `types` array, causing `os`/`path` module not-found errors. Other packages (e.g. `parallax_ms`) already use `"types": ["bun-types"]` — apply the same pattern.

**Useful Context for Future Agents:** `bunqueue/client` exports both `Queue` (producer + job management) and `Worker` (consumer/processor) separately. Prefer `Queue` when building a shared queue handle with no processing logic attached. The `Bunqueue` convenience class is unsuitable when you want to defer processor assignment. `QueueOptions` accepts `embedded: boolean` and `dataPath: string` for SQLite path configuration.

