# Refactor Plan — Iteration 000044 (Pass 001)

> Completion report for the audit-001 remediation pass.

---

## Summary of changes

### 1. `PARALLAX_DB_PATH` environment variable support (FR-2)

`server/queue.py` previously hard-coded the default DB path without reading the
environment. The `_DB_PATH` module-level constant was updated to resolve via
`os.environ.get("PARALLAX_DB_PATH", <default>)`, satisfying FR-2.  
Committed in: `dce1bee` (prior remediation commit).

### 2. Standardise initial job status to `'queued'` (US-004-AC01)

`server/queue.py :: JobQueue.enqueue()` was inserting `status='pending'`.
Changed to `status='queued'` to align with US-004-AC01 language and the
worker's `_PROCESSABLE_STATUSES` guard.  
`server/worker.py :: _PROCESSABLE_STATUSES` was tightened to `{'queued'}` only,
removing the now-unnecessary `'pending'` fallback.  
The corresponding test assertion and an obsolete worker test were updated.  
Committed in: `dce1bee`.

### 3. `aiosqlite` added to dev dependency group

`pyproject.toml` `[dependency-groups.dev]` now includes `aiosqlite>=0.22.1`.
This ensures `uv run pytest` (without any `--extra` flag) can execute the 38
queue-related tests without `ModuleNotFoundError`.  
Committed in: `dce1bee`.

### 4. Ruff lint cleanup (style pass)

`server/queue.py` and `comfy_diffusion/progress.py` were cleaned up after
`uv run ruff check --fix`:

- Removed unused `sys` import from `progress.py` (F401).
- Replaced quoted type annotations with bare annotations throughout
  `queue.py` (UP037) — safe because `from __future__ import annotations` is
  present.
- Replaced `timezone.utc` with `datetime.UTC` alias (UP017).
- Split a 111-character SQL `INSERT` string literal across three lines (E501).

Committed in: `d0ce4cf`.

---

## Quality checks

| Check | Command | Result |
|-------|---------|--------|
| Linter | `uv run ruff check server/queue.py server/worker.py server/jobs.py server/submit.py comfy_diffusion/progress.py` | ✅ All checks passed |
| US-001 model tests | `.venv/bin/pytest tests/test_server_jobs_us001.py` | ✅ 25 passed |
| US-002 queue tests | `uv run pytest tests/test_server_queue_us002.py -x --tb=short -q` | ✅ 38 passed |
| US-003/004/005 tests | `uv run pytest tests/test_server_submit_us003.py tests/test_server_worker_us004.py tests/test_progress_us005.py tests/test_progress_reporter.py -q` | ✅ 88 passed |

**Note on test-exit warnings:** When the queue test suite runs in an isolated
process, aiosqlite's background worker thread emits a `PytestUnhandledThreadExceptionWarning`
about a closed event loop during interpreter shutdown. This is a known benign
artefact of using `asyncio.run()` in synchronous test helpers — the tests
themselves all pass and no assertion fails. No fix applied; this is a pre-existing
test-design concern unrelated to the current refactor items.

---

## Deviations from refactor plan

None. All three actionable fixes identified in `it_000044_audit-report_001.json`
were applied:

1. ✅ `PARALLAX_DB_PATH` env-var support added to `queue.py`.
2. ✅ `aiosqlite` moved into `[dependency-groups.dev]` so stock `uv run pytest` works.
3. ✅ `status='pending'` standardised to `status='queued'` throughout.

An additional style pass (ruff lint cleanup) was performed on the modified files
as part of normal code hygiene; this is not a deviation from the plan.
