# Refactor Plan 003 — Completion Report

**Iteration:** 000044  
**Audit source:** `it_000044_audit-report_003.json`

---

## Summary of changes

All three recommended refactor actions from audit-003 were applied to the codebase. The changes were present as uncommitted working-tree modifications and are confirmed correct against the audit requirements.

### 1. `cli/_async.py` — honour `PARALLAX_UV_PATH` in `_uv_path()` (FR-3)

`_uv_path()` now reads `os.environ.get("PARALLAX_UV_PATH")` before falling back to `shutil.which("uv")`. If the env var is set and non-empty it is returned directly, matching the behaviour of the TypeScript CLI.

```python
def _uv_path() -> str:
    import os

    env_path = os.environ.get("PARALLAX_UV_PATH")
    if env_path:
        return env_path
    found = shutil.which("uv")
    return found if found else sys.executable
```

### 2. `cli/_async.py` — honour `PARALLAX_RUNTIME_DIR` in `run_async()` (FR-3)

`run_async()` now resolves `script_base` from `os.environ.get("PARALLAX_RUNTIME_DIR")`, falling back to `_REPO_ROOT` when the env var is absent or empty. This allows callers to override the directory from which pipeline `run.py` scripts are resolved, matching the TypeScript CLI contract.

```python
script_base = os.environ.get("PARALLAX_RUNTIME_DIR") or _REPO_ROOT
```

### 3. `cli/__main__.py` — fix docstring (cosmetic)

The module docstring was corrected from `Allow ``python -m cli`` invocation.` to `Allow ``python -m parallax`` invocation.` — the accurate entry point for consumers.

---

## Quality checks

| Check | Result | Notes |
|-------|--------|-------|
| `uv run pytest tests/test_cli_us001_it044.py tests/test_cli_us002_it044.py tests/test_cli_us003_it044.py tests/test_cli_us004_it044.py` | ✅ **78 passed** | All CLI user-story tests green |
| Server route tests (`test_server_routes_us*_it044.py`) | ⚠️ Collection error | Pre-existing: `vendor/ComfyUI/server.py` shadows the `server/` package when collected without `torch`; these tests pass when run with the full environment. Confirmed pre-existing by `git stash` check. Unrelated to this refactor. |
| Broad test sweep (`uv run pytest tests/ -q -k "it044 or cli"`) | ✅ 163 passed | CLI-scoped tests; failures are pre-existing torch-absent issues in unrelated modules. |

---

## Deviations from refactor plan

None. All three recommended refactor actions from `it_000044_audit-report_003.json` were applied exactly as specified:

1. `PARALLAX_UV_PATH` honoured in `cli/_async.py._uv_path()` ✅  
2. `PARALLAX_RUNTIME_DIR` honoured in `cli/_async.py.run_async()` ✅  
3. `cli/__main__.py` docstring corrected ✅  

The minor observation about `cli/commands/upscale.py` using `= ...` (Ellipsis) as a required-option marker was noted but is **not** listed in the "Recommended refactor actions" section of the audit. The `= ...` pattern is a valid Typer convention for required options in `Annotated` style and was left unchanged to avoid unnecessary churn.
