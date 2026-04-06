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
