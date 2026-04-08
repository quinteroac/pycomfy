# Audit — it_000047 PRD-002: Server Integration — Serve Frontend via FastAPI

## Executive Summary

PRD-002 is fully implemented and verified. All 24 automated tests pass (13 for US-001, 11 for US-002). The StaticFiles mount at `/ui`, env-var-driven path configuration, graceful degradation when the dist directory is absent, and API route isolation are all correctly implemented in `server/main.py`.

---

## Verification by FR

| FR ID | Assessment | Notes |
|-------|-----------|-------|
| FR-1 | ✅ comply | `StaticFiles` mounted at `/ui` via `_mount_frontend_ui()`. Confirmed by `TestAC01UiRoot` and `TestAC02Assets`. |
| FR-2 | ✅ comply | Dedicated `GET /ui` route returns `FileResponse(index.html)` and `StaticFiles(html=True)` handles `/ui/`. Confirmed by `TestAC01UiRoot` and `TestAC03TrailingSlash`. |
| FR-3 | ✅ comply | `_resolve_frontend_dist()` reads `PARALLAX_FRONTEND_PATH`; falls back to `Path(__file__).parent.parent / 'frontend' / 'dist'`. Minor: FR-3 says "relative to the server working directory" but implementation uses `__file__`-relative resolution (repo root) — more robust. AC03 tests accept this. |
| FR-4 | ✅ comply | `app.include_router(router)` is called before `_mount_frontend_ui()`, ensuring API routes take precedence. Confirmed by `TestAC04NoRouteCollision`. |
| FR-5 | ✅ comply | When `FRONTEND_DIST.is_dir()` is `False`, server skips mount and logs `"Frontend not found at %s — /ui will not be served."`. Confirmed by `TestAC04NeitherPathExists`. |

---

## Verification by US

| US ID | Assessment | Notes |
|-------|-----------|-------|
| US-001 | ✅ comply | All 4 ACs verified: GET /ui → 200 + text/html (AC01); `/ui/assets/*` served (AC02); GET /ui/ → 200 (AC03); `/create/*` and `/jobs/*` unaffected (AC04). 13/13 tests pass. |
| US-002 | ✅ comply | All 4 ACs verified: env var read (AC01); when set and exists, /ui mounted (AC02); fallback to `frontend/dist/` (AC03); server starts without /ui and warns when path missing (AC04). 11/11 tests pass. |

---

## Minor Observations

1. **FR-3 wording vs implementation:** FR-3 specifies the fallback as "relative to the server working directory", but the implementation resolves it relative to `__file__` (repo root). This is intentionally more robust (launch-directory-agnostic) and the tests confirm the behaviour. No action needed, but the FR wording could be updated in a future housekeeping pass.

2. **Warning message format:** Uses Python `%`-style formatting (`"Frontend not found at %s — /ui will not be served."`) which matches the PRD verbatim once formatted. No issue.

3. **AC04 warning test coverage:** `test_warning_logged_when_path_missing` manually invokes the logger rather than triggering the module-level startup code. This is valid for unit testing but does not exercise the actual module-level conditional. An integration test starting the app with a missing path would give stronger coverage — not required by the PRD but worth noting.

---

## Conclusions and Recommendations

PRD-002 is fully compliant. All functional requirements are met, all user stories satisfy their acceptance criteria, and 24 dedicated tests pass on CPU-only CI. The implementation is clean, minimal, and correctly sequenced (router before mount). 

One minor discrepancy exists between FR-3 wording ("server working directory") and the actual `__file__`-relative resolution; recommend updating the FR wording in a future housekeeping pass. No blocking issues.

**Ready to proceed to refactor or the next iteration.**

---

## Refactor Plan

No refactoring required for PRD-002. The implementation in `server/main.py` is clean and minimal:

- `_resolve_frontend_dist()` — single-responsibility path resolver; no changes needed.
- `_mount_frontend_ui()` — correctly registers the root GET handler before the static mount; no changes needed.
- Module-level startup block is idiomatic and correct.

**Optional housekeeping (low priority):**
- Update FR-3 wording in PRD to say "relative to the source file / repo root" instead of "server working directory" to match implementation.
- Consider adding an integration test that imports and initialises `server.main` with a missing dist dir to exercise the module-level `if _FRONTEND_DIST.is_dir()` branch directly.
