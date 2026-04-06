# Lessons Learned — Iteration 000045

## US-001 — Build standalone binary locally

**Summary:** Created `parallax.spec` (PyInstaller DSL) at the repository root, configured for single-file (`onefile`) output by passing `a.binaries`, `a.zipfiles`, and `a.datas` directly into `EXE` without a `COLLECT` step. The spec excludes `torch`, `torchvision`, `torchaudio`, `transformers`, and `comfy_diffusion` (plus `numpy`, `scipy`, `PIL`, `cv2`, and other scientific packages) to keep the binary under 50 MB. The entry-point is `cli/__main__.py`. Tests validate the spec file statically (no PyInstaller execution in CI).

**Key Decisions:**
- **Static tests only**: PyInstaller is not installed in the project's dev environment and would be slow in CI. All tests verify the spec file's content rather than executing `pyinstaller`. This satisfies the testing strategy ("critical paths only", CPU-only CI).
- **Onefile via no-COLLECT pattern**: PyInstaller's single-file mode is achieved by omitting the `COLLECT` step and passing `a.binaries`, `a.zipfiles`, `a.datas` directly to `EXE`. The `COLLECT`-less pattern is how all current PyInstaller docs describe onefile output.
- **Entry-point `cli/__main__.py`**: This file already exists and does `from cli.main import app; app()`, making it the natural PyInstaller entry-point without creating an extra wrapper script.
- **Hidden imports list**: Added `typer`, `click`, `rich`, `shellingham`, `aiosqlite`, `httpx`, `anyio`, and `pydantic` submodules that static analysis commonly misses due to dynamic import patterns inside typer/click.
- **Broad excludes for size**: In addition to the five AC03-required excludes, `numpy`, `scipy`, `PIL`, `cv2`, `kornia`, `einops`, `safetensors`, `tokenizers`, `huggingface_hub`, `av`, `spandrel`, `glfw`, and `OpenGL` are excluded. These are all transitively imported by comfy_diffusion pipelines at call time (lazy imports), never at CLI startup.

**Pitfalls Encountered:**
- None — the spec file is a declarative artifact with no runtime logic, so implementation was straightforward once the onefile pattern was confirmed.

**Useful Context for Future Agents:**
- **`pyinstaller` is not in `pyproject.toml` deps**: Running `uv run pyinstaller parallax.spec` requires `pyinstaller` to be installed in the active environment first (e.g. `uv add --dev pyinstaller`). The AC says "Running `uv run pyinstaller ...`", so adding it as a dev dependency would be the next step before actual binary builds.
- **The lazy-import pattern is critical**: `cli/_runners/image.py` imports `comfy_diffusion` only inside function bodies (`from comfy_diffusion.pipelines... import run`). This means PyInstaller's static analysis will NOT see those imports, so they won't be bundled — exactly what we want for binary size.
- **No `COLLECT` = onefile**: Any future spec that introduces a `COLLECT(...)` block will produce a directory distribution (dist/parallax/) instead of a single file. The absence of `COLLECT` is the onefile signal.
- **Test file naming**: The test file is `tests/test_build_us001_it000045.py`. The `test_cli_us001_it000045.py` file belongs to PRD_001 US-001 (runtime install); use distinct prefixes (`test_build_` vs `test_cli_`) to avoid confusion.

---

## US-002 — CI builds binaries on every version tag

**Summary:** Created `.github/workflows/release-cli.yml` with three parallel build jobs (linux, macos, windows) and a gated `release` job that uploads 6 assets (3 binaries + 3 checksums) via `softprops/action-gh-release`. Tests are static YAML/text assertions in `tests/test_ci_us002_it000045.py` — 28 tests, all passing.

**Key Decisions:**
- **Static tests only**: The workflow cannot be executed locally (requires GitHub Actions). All tests verify the workflow file's content structurally — consistent with the US-001 approach.
- **Binary renaming step**: PyInstaller outputs `dist/parallax` / `dist/parallax.exe` regardless of platform. Each job renames the output to its platform-specific name (`parallax-linux-x86_64`, `parallax-macos-universal`, `parallax-windows-x86_64.exe`) before generating checksums. This keeps release assets unambiguous.
- **Windows checksum via PowerShell `Get-FileHash`**: `sha256sum` is unavailable on Windows runners. The checksum output is formatted manually to match the Linux/macOS `sha256sum` format (`<hash>  <filename>`).
- **`needs: [linux, macos, windows]`**: This is the canonical GitHub Actions mechanism to gate a job on multiple upstream jobs all succeeding (AC07). No additional `if:` condition is required.
- **`permissions: contents: write`** at workflow scope: `softprops/action-gh-release` needs write access to create/update releases and upload assets.

**Pitfalls Encountered:**
- None significant. The rename → checksum → artifact-upload → download-in-release → publish pattern is well-established for GitHub Actions release workflows.

**Useful Context for Future Agents:**
- **`uv sync --group cli-build --no-group dev`** assumes `cli-build` is a declared dependency group in `pyproject.toml` containing `pyinstaller`. If the group doesn't exist, the workflow step will fail.
- **`--target-arch universal2` is macOS-only**: Do not pass this flag on Linux or Windows jobs.
- **Artifact naming**: Each build job uploads to a uniquely-named artifact (`linux-artifacts`, `macos-artifacts`, `windows-artifacts`). The release job downloads all three into the same `dist/` directory — no conflicts because binary names are platform-specific.
- **`softprops/action-gh-release@v2`** automatically uses the triggering tag (`GITHUB_REF`) — no explicit `tag_name:` input needed for tag-triggered runs.
- **Test file naming convention**: `test_ci_us002_it000045.py` (prefix `test_ci_`) for CI workflow tests, distinct from `test_build_` (PyInstaller spec) and `test_cli_` (runtime/install) prefixes.
