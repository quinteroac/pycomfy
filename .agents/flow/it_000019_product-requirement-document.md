# Requirement: Variadic CLIP Loader (`load_clip(*paths)`)

## Context

`ModelManager.load_clip()` currently accepts a single path and wraps `comfy.sd.load_clip(ckpt_paths=[single_path], ...)`.
However `comfy.sd.load_clip()` already accepts a list of paths and merges them into one combined text-encoder object — this is the
mechanism used by SDXL (CLIP-L + OpenCLIP-ViT-bigG), SD3 (CLIP-L + CLIP-G + T5-XXL), Flux (CLIP-L + T5-XXL), and any other
architecture whose text encoder spans multiple checkpoint files.

The wrapper must be updated to expose this capability without breaking existing single-path call sites.

## Goals

- Allow callers to load any multi-file text encoder (SDXL, SD3, Flux, HunyuanVideo, etc.) with a single `load_clip()` call.
- Preserve full backward compatibility: every existing single-path call site continues to work unchanged.
- No new package dependencies; all tests pass in a CPU-only environment.

## User Stories

### US-001: Variadic path acceptance in `load_clip`

**As a** library consumer, **I want** to call `manager.load_clip(path1, path2, clip_type="sdxl")` **so that** I can load
dual-encoder models (e.g. SDXL, Flux) without resorting to `load_ltxav_text_encoder` or manual ComfyUI internals.

**Acceptance Criteria:**
- [ ] `ModelManager.load_clip` signature is `load_clip(self, *paths: str | Path, clip_type: str = "stable_diffusion") -> Any`.
- [ ] Calling with a single path (e.g. `manager.load_clip("clip_l.safetensors")`) continues to work identically to the previous behaviour.
- [ ] Calling with two paths (e.g. `manager.load_clip("clip_l.safetensors", "t5xxl_fp16.safetensors", clip_type="flux")`) resolves both paths and passes them as `ckpt_paths=[path1, path2]` to `comfy.sd.load_clip`.
- [ ] Each individual path is resolved independently using the same rule as before: absolute paths are validated to be existing files; relative names are looked up via `folder_paths.get_full_path_or_raise("text_encoders", name)`.
- [ ] If any one path does not resolve to an existing file, a `FileNotFoundError` is raised before the ComfyUI loader is called, with the message `"clip file not found: <path>"`.
- [ ] Calling with zero paths raises a `ValueError` with message `"load_clip requires at least one path"`.
- [ ] An unrecognised `clip_type` string raises `ValueError` with a message containing the invalid value and listing all valid type names (lowercased).
- [ ] Type annotations on the public API are updated to reflect the variadic signature.
- [ ] Typecheck / lint passes.

### US-002: Backward-compatible test suite

**As a** maintainer, **I want** the existing `test_model_manager_clip_loading.py` tests to continue passing **so that** no regression
is introduced in the single-path code path.

**Acceptance Criteria:**
- [ ] All existing tests in `tests/test_model_manager_clip_loading.py` pass without modification to the test file.
- [ ] A new test covers the two-path scenario: two files created, both resolved to absolute paths, `comfy.sd.load_clip` receives `ckpt_paths` with both resolved strings.
- [ ] A new test covers the zero-path scenario: `load_clip()` with no arguments raises `ValueError`.
- [ ] A new test covers the partial-missing-file scenario: first path exists, second path does not → `FileNotFoundError` raised before the loader is called.
- [ ] `uv run pytest` passes in a CPU-only environment.

## Functional Requirements

- FR-1: `ModelManager.load_clip` MUST accept one or more positional `str | Path` arguments via `*paths`.
- FR-2: Zero positional arguments MUST raise `ValueError("load_clip requires at least one path")`.
- FR-3: Each path in `*paths` MUST be resolved independently: absolute existing files are used as-is; relative names are resolved via `folder_paths.get_full_path_or_raise("text_encoders", name)`. An absolute path that does not point to an existing file MUST raise `FileNotFoundError`.
- FR-4: All resolved paths are forwarded to `comfy.sd.load_clip(ckpt_paths=[...], embedding_directory=..., clip_type=...)` as a list in the same order they were provided.
- FR-5: `clip_type` is a keyword-only `str` argument with default `"stable_diffusion"`. It is mapped to `comfy_sd.CLIPType` by looking up `clip_type.upper()` as an enum member name. If the name does not match any member, a `ValueError` MUST be raised with a message that includes the invalid value and lists all valid `CLIPType` member names (lowercased).
- FR-6: The lazy-import pattern is preserved — no `torch`, `comfy.*`, or `folder_paths` imports at module top level.
- FR-7: `load_ltxav_text_encoder` is left untouched (backward compatibility); no deprecation warning is added in this iteration.

## Non-Goals (Out of Scope)

- Deprecating or removing `load_ltxav_text_encoder`.
- Adding a `clip_type` value for architectures not yet supported by the existing `comfy_sd.CLIPType` enum.
- Any GPU-dependent tests or CI changes.
- Updating `__init__.py` re-exports (no change to the public convenience imports).
- Validating that the combination of encoders is sensible for the given `clip_type`.

## Open Questions

_(None — all questions resolved during the requirements interview.)_
