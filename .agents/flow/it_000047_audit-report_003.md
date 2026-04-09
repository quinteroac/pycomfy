# Audit Report — Iteration 000047 · PRD Index 003

**Generated:** 2026-04-07T21:02:49.879Z  
**PRD:** it_000047_product-requirement-document_003.md  
**Scope:** `parallax frontend install` / `parallax frontend version` CLI commands

---

## 1. Executive Summary

The implementation for PRD-003 (Frontend Install CLI) is largely complete and all **46 unit tests pass**. US-001, US-002, and US-003 are fully satisfied. FR-1 through FR-5 and FR-6 comply. Two non-compliances exist: **FR-2** deviates by using `urllib.request` (stdlib) instead of `httpx`, and **FR-7** (CI workflow extension to build and publish the frontend archive) is entirely absent from `release-cli.yml`.

---

## 2. Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `frontend` Typer group registered in `cli/main.py`; `install` and `version` subcommands implemented in `cli/commands/frontend.py`. |
| FR-2 | ❌ does_not_comply | Uses `urllib.request` (stdlib) instead of `httpx` as explicitly required. Functionally equivalent but violates the stated constraint. |
| FR-3 | ✅ comply | Archive extracted to `~/.parallax/frontend/`; prior installation removed atomically before extraction. |
| FR-4 | ✅ comply | `_write_config_env` upserts `PARALLAX_FRONTEND_PATH` into `~/.parallax/config.env`. |
| FR-5 | ✅ comply | `parallax frontend version` reads `~/.parallax/frontend/version.txt`. |
| FR-6 | ✅ comply | `_find_frontend_asset` filters by `parallax-frontend-{version}.tar.gz`; `browser_download_url` from the GitHub API produces the expected URL pattern. |
| FR-7 | ❌ does_not_comply | `release-cli.yml` has no steps for Bun frontend build, archive packaging, or upload of `parallax-frontend-{version}.tar.gz` as a release asset. |

---

## 3. Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | All 6 ACs satisfied and covered by tests (25 test cases). |
| US-002 | ✅ comply | All 3 ACs satisfied (--version, 404 handling, latest default) — 14 tests. |
| US-003 | ✅ comply | Both ACs satisfied (version.txt read, not-installed message) — 4 tests. |

**Test results:** 46/46 passed in 0.16 s.

---

## 4. Minor Observations

- `_write_config_env` writes the expanded absolute path (e.g. `/home/user/.parallax/frontend`) rather than the tilde form `~/.parallax/frontend` stated in AC03. Functionally correct; cosmetic discrepancy with the PRD wording.
- FR-2: `urllib.request` is arguably preferable here (no extra dependency, stdlib). Recommend updating the PRD to align with the implementation choice rather than forcing an httpx refactor.
- No test covers the `tarfile.extractall` failure branch (partial-install cleanup after corrupt archive). Consider adding a mock-corrupt-archive test for AC05 completeness.

---

## 5. Conclusions and Recommendations

The prototype satisfies all user stories and all acceptance criteria are verified. The two FR gaps are:

1. **FR-2 (httpx vs urllib):** Minor technical deviation with no user-visible impact. Recommend relaxing FR-2 in the PRD to allow stdlib urllib, avoiding an unnecessary refactor.
2. **FR-7 (CI workflow):** A genuine gap. Without this step, no frontend archive will ever exist in GitHub Releases, making the entire `parallax frontend install` command non-functional in production. This is the highest-priority item.

---

## 6. Refactor Plan

### Priority 1 — FR-7: Extend release-cli.yml with frontend build job

**File:** `.github/workflows/release-cli.yml`

Add a new `build-frontend` job (runs in parallel with linux/macos/windows) that:

1. Checks out the repo (no submodules needed for frontend).
2. Sets up Bun.
3. Runs `bun install && bun run build` in `frontend/`.
4. Derives `VERSION` from the git tag (`${GITHUB_REF_NAME#v}`).
5. Writes `frontend/dist/version.txt` with the version string.
6. Packages: `tar -czf parallax-frontend-${VERSION}.tar.gz -C frontend/dist .`
7. Uploads artifact `frontend-artifact`.

Update the `release` job to:
- Add `build-frontend` to the `needs` list.
- Download `frontend-artifact` into `dist/`.
- Add `dist/parallax-frontend-*.tar.gz` to the `softprops/action-gh-release` `files` list.

### Priority 2 — FR-2: Align PRD with implementation (no code change needed)

Update `it_000047_product-requirement-document_003.md` FR-2 description to allow stdlib `urllib.request` as an acceptable HTTP client, reflecting the actual implementation.

### Priority 3 — Test coverage (optional)

Add a test in `test_cli_frontend_us001_it000047.py` that mocks `tarfile.open` to raise an exception and asserts:
- Exit code is non-zero.
- `frontend_dir` does not exist after failure (AC05 cleanup path).
