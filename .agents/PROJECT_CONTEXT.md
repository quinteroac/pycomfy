# Project Context

<!-- Created or updated by `bun nvst create project-context`. Cap: 250 lines. -->

## Conventions
- Naming: snake_case for files, variables, functions; PascalCase for classes
- Formatting: no enforced formatter yet — follow PEP 8 conventions
- Git flow: feature branches per iteration (`feature/it_XXXXXX`), merge to `main` via PR
- Workflow: all agent commands via `bun nvst <command>`; all Python via `uv`
- Language: all generated resources must be in English

## Tech Stack
- Language: Python 3.12+
- Runtime: CPython
- Frameworks: none (pure library)
- Key libraries: torch (optional, via extras), ComfyUI (vendored submodule)
- Package manager: uv (no pip/venv)
- Build / tooling: pyproject.toml (PEP 621), uv for install/sync/run

## Code Standards
- Style: PEP 8, type hints on public API
- Error handling: `check_runtime()` returns error dicts (no exceptions for expected failures)
- Module organisation: src-less layout — `pycomfy/` package at repo root, vendored deps in `vendor/`
- Forbidden patterns: no hardcoded torch versions; no manual `sys.path` manipulation outside pycomfy internals; no pip/venv commands

## Testing Strategy
- Approach: critical paths only
- Runner: pytest (via `uv run pytest`)
- Coverage targets: none enforced
- Test location: `tests/` at repo root
- Constraint: CI is CPU-only — all tests must pass without GPU

## Product Architecture
- pycomfy is a standalone Python library exposing ComfyUI's inference engine as importable modules
- No server, no UI, no application layer — import and run inference in your own code
- ComfyUI vendored as git submodule at `vendor/ComfyUI`, pinned to a stable release tag

### Data Flow
1. Consumer does `import pycomfy`
2. `pycomfy/__init__.py` adds `vendor/ComfyUI` to `sys.path` (absolute paths from `__file__`)
3. ComfyUI internals (e.g. `comfy.model_management`) become importable
4. Consumer calls `pycomfy.check_runtime()` → returns structured diagnostics dict

## Modular Structure
- `pycomfy/`: main package — public API, path management, runtime diagnostics
- `vendor/ComfyUI/`: vendored ComfyUI submodule (not edited directly)
- `tests/`: pytest test files

## ComfyUI API Notes
- `clip.encode_from_tokens_scheduled(tokens)` is the canonical method for text conditioning — mirrors `CLIPTextEncode` in `nodes.py`.
- `clip.encode_from_tokens(tokens, return_pooled=True)` is exclusive to GLIGEN (spatial conditioning) — do not use for standard text encoding.

## Implemented Capabilities
<!-- Updated at the end of each iteration -->
- (none yet — populated after first Refactor)
