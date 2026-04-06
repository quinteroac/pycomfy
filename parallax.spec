# parallax.spec — PyInstaller build spec for the ``parallax`` CLI binary.
#
# Usage (Linux / macOS):
#     uv run pyinstaller parallax.spec
#
# Produces:
#     dist/parallax          (Linux / macOS)
#     dist/parallax.exe      (Windows)
#
# The binary is a single self-contained executable that does NOT require a
# Python installation on the target machine.  Heavy ML packages (torch,
# torchvision, torchaudio, transformers, comfy_diffusion) are explicitly
# excluded so that the binary stays well under 50 MB.
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    # Entry-point: the CLI __main__ module calls cli.main:app().
    ["cli/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Typer / Click internals that static analysis may miss
        "typer",
        "typer.main",
        "typer.utils",
        "typer.params",
        "typer.models",
        "click",
        "click.core",
        "click.decorators",
        "click.exceptions",
        "click.formatting",
        "click.shell_completion",
        "click.types",
        # Rich (used by Typer for output and by jobs command at call-time)
        "rich",
        "rich.console",
        "rich.markup",
        "rich.text",
        "rich.table",
        "rich.progress",
        "rich.theme",
        "rich.highlighter",
        "rich.logging",
        # Shell detection (optional Typer dependency)
        "shellingham",
        # Standard-library async / IO used by jobs command
        "asyncio",
        "aiosqlite",
        # HTTP client used by CLI commands
        "httpx",
        "anyio",
        "anyio._backends._asyncio",
        "anyio._backends._trio",
        # Pydantic (used by server modules imported lazily)
        "pydantic",
        "pydantic.v1",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ── Heavy ML packages — must NOT be bundled (AC03) ────────────────
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "comfy_diffusion",
        # ── ComfyUI vendored submodule ────────────────────────────────────
        "comfy",
        "comfy_extras",
        # ── Scientific / vision stacks (pulled in transitively) ───────────
        "numpy",
        "scipy",
        "PIL",
        "cv2",
        "kornia",
        "einops",
        "safetensors",
        "tokenizers",
        "sentencepiece",
        "huggingface_hub",
        # ── Multimedia / GPU libraries ────────────────────────────────────
        "av",
        "spandrel",
        "glfw",
        "OpenGL",
        # ── Matplotlib / test utilities (not needed at runtime) ───────────
        "matplotlib",
        "pytest",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single-file executable: pass binaries + datas directly to EXE (no COLLECT).
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="parallax",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
