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

# ── Bake version from pyproject.toml into cli/_version.py ─────────────────
# This runs at spec-parse time (before Analysis), so the version constant is
# embedded in the bundled source rather than looked up from an installed
# package at runtime (comfy-diffusion is NOT bundled in the binary).
import tomllib as _tomllib
import pathlib as _pathlib
import textwrap as _textwrap

_repo_root = _pathlib.Path(SPECPATH)
with open(_repo_root / "pyproject.toml", "rb") as _f:
    _project_version = _tomllib.load(_f)["project"]["version"]

(_repo_root / "cli" / "_version.py").write_text(
    _textwrap.dedent(f'''\
        """Build-time version constant for the ``parallax`` CLI binary.

        Generated automatically by ``parallax.spec`` — do not edit by hand.
        Source of truth: ``[project].version`` in ``pyproject.toml``.
        """

        __version__ = "{_project_version}"
        '''),
    encoding="utf-8",
)
# ──────────────────────────────────────────────────────────────────────────

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
        # Standard-library modules imported transitively by torch + full ComfyUI stack.
        # Traced via import hook against comfy.* / nodes / execution with torch installed.
        # Top-level modules
        "abc", "argparse", "array", "ast", "asyncio", "atexit",
        "base64", "binascii", "bisect", "builtins", "bz2",
        "cProfile", "calendar", "cmath", "codecs", "collections",
        "colorsys", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "csv", "ctypes",
        "dataclasses", "datetime", "decimal", "difflib", "dis",
        "email", "encodings", "enum", "errno", "fcntl",
        "filecmp", "fileinput", "fnmatch", "fractions", "functools",
        "gc", "getpass", "gettext", "glob", "grp", "gzip",
        "hashlib", "heapq", "hmac", "html", "http",
        "importlib", "inspect", "io", "ipaddress", "itertools",
        "json", "keyword", "linecache", "locale", "logging",
        "lzma", "marshal", "math", "mimetypes", "mmap",
        "modulefinder", "multiprocessing", "numbers", "opcode",
        "operator", "os", "pathlib", "pickle", "pickletools",
        "pkgutil", "platform", "posixpath", "pprint", "profile",
        "pstats", "pwd", "pydoc", "pyexpat", "queue", "quopri",
        "random", "re", "reprlib", "resource", "runpy",
        "select", "selectors", "shlex", "shutil", "signal",
        "socket", "sqlite3", "ssl", "stat", "statistics",
        "string", "struct", "subprocess", "sysconfig", "tarfile",
        "tempfile", "termios", "textwrap", "threading", "time",
        "timeit", "token", "tokenize", "trace", "traceback",
        "types", "typing", "unicodedata", "unittest", "urllib",
        "uuid", "warnings", "weakref", "zipfile", "zipimport", "zlib",
        # Submodules that PyInstaller won't auto-detect
        "asyncio.base_futures", "asyncio.base_tasks",
        "asyncio.coroutines", "asyncio.events", "asyncio.exceptions",
        "collections.abc",
        "concurrent.futures", "concurrent.futures._base",
        "ctypes._endian", "ctypes.util",
        "email._encoded_words", "email._parseaddr", "email._policybase",
        "email.base64mime", "email.charset", "email.encoders",
        "email.errors", "email.feedparser", "email.iterators",
        "email.message", "email.parser", "email.quoprimime", "email.utils",
        "encodings.aliases",
        "html.entities", "html.parser",
        "http.client", "http.cookiejar",
        "importlib.abc", "importlib.machinery",
        "importlib.metadata", "importlib.resources", "importlib.util",
        "multiprocessing.connection", "multiprocessing.resource_sharer",
        "multiprocessing.resource_tracker", "multiprocessing.util",
        "os.path",
        "sqlite3.dbapi2",
        "unittest.case", "unittest.mock", "unittest.util",
        "urllib.error", "urllib.parse", "urllib.request", "urllib.response",
        "xml.etree.ElementPath", "xml.etree.ElementTree",
        # Standard-library async / IO used by jobs command (asyncio already above)
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
