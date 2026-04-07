# Lessons Learned — Iteration 000047

## US-001 — Install the frontend via CLI

**Summary:** Implemented `parallax frontend install` command as a new `cli/commands/frontend.py` Typer sub-app. The command downloads the latest pre-built frontend archive from GitHub Releases (`parallax-frontend-{version}.tar.gz`), extracts it to `~/.parallax/frontend/`, and writes `PARALLAX_FRONTEND_PATH=~/.parallax/frontend` to `~/.parallax/config.env`. Also added `BASE_DIR`, `FRONTEND_DIR`, and `CONFIG_ENV_PATH` constants to `cli/commands/_common.py`.

**Key Decisions:**
- Added `BASE_DIR`, `FRONTEND_DIR`, `CONFIG_ENV_PATH` to `_common.py` rather than hardcoding paths in `frontend.py`, following the `ENV_DIR` pattern already there.
- Used stdlib `urllib.request` and `tarfile` exclusively — no new dependencies.
- Placed download inside a `tempfile.TemporaryDirectory()` so a failed download or extraction never leaves a partial install in `~/.parallax/frontend/`.
- `_write_config_env` upserts (not appends blindly) the `PARALLAX_FRONTEND_PATH` key so re-running updates it cleanly without duplicates.
- Added `filter="data"` to `tarfile.extractall()` to suppress the Python 3.14 deprecation warning.

**Pitfalls Encountered:**
- **Typer single-command app invocation:** When a `typer.Typer()` instance has exactly one `@app.command("name")` registered and `no_args_is_help=True`, calling `runner.invoke(app, ["name"])` raises "Got unexpected extra argument (name)" with exit code 2. Typer treats the single command as the app itself, not a subgroup routing to that name. Fix: in tests, wrap the sub-app in a parent: `_cli = typer.Typer(); _cli.add_typer(app, name="frontend")` and invoke with `["frontend", "install"]`.

**Useful Context for Future Agents:**
- The `_common.py` now exports `BASE_DIR`, `FRONTEND_DIR`, and `CONFIG_ENV_PATH` — use these in any new command that needs `~/.parallax/` paths.
- `config.env` is a shell-style `KEY=VALUE` file written at `~/.parallax/config.env`. It stores persistent runtime configuration like `PARALLAX_FRONTEND_PATH`. Any command that needs to persist a path should use `_write_config_env()` (or a similar upsert approach on this file).
- When testing a Typer sub-app in isolation, always wrap it in a parent app (`_cli.add_typer(sub_app, name="sub")`) and invoke with `["sub", "cmd"]` to ensure correct subgroup routing.
- The GitHub repo is `quinteroac/comfy-diffusion` — used in `_GITHUB_REPO` for the Releases API URL.
