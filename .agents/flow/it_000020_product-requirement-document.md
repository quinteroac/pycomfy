# Requirement: `textgen` — LLM/VLM Text Generation Module

## Context
`comfy-diffusion` exposes ComfyUI's inference engine as importable Python modules. ComfyUI ships two text-generation nodes — `TextGenerate` (general-purpose LLM/VLM inference) and `TextGenerateLTX2Prompt` (LTX-Video 2 prompt enhancer) — but they are only accessible via the node graph. This iteration wraps those nodes as plain Python functions in a new `comfy_diffusion.textgen` module, letting developers call LLM inference directly from Python without a running server.

## Goals
- Expose `generate_text()` and `generate_ltx2_prompt()` as importable functions in `comfy_diffusion.textgen`.
- Extend `ModelManager` with a `load_llm()` method that follows the existing loader pattern (`load_clip`, `load_vae`, etc.).
- All functions must be testable on CPU-only environments (mocked LLM calls in tests).

## User Stories

### US-001: Load an LLM model via `ModelManager.load_llm()`
**As a** developer, **I want** to load an LLM/VLM model through `ModelManager.load_llm(path)` **so that** I can obtain a model object to pass to text-generation functions without manually managing `folder_paths` or ComfyUI internals.

**Acceptance Criteria:**
- [ ] `ModelManager` gains a `load_llm(path: str | Path) -> Any` method.
- [ ] If `path` is an absolute path pointing to an existing file, that file is loaded directly.
- [ ] If `path` is a relative name, it is resolved under the `llm` subfolder of `models_dir` (e.g. `models_dir/llm/<name>`).
- [ ] If the file does not exist, `FileNotFoundError` is raised with a descriptive message.
- [ ] `folder_paths` registers `models_dir/llm` as the `llm` model folder during `ModelManager.__init__`.
- [ ] No `torch` or `comfy.*` imports at module top level — all deferred to call time.
- [ ] Typecheck / lint passes.

### US-002: Generate text with `generate_text()`
**As a** developer, **I want** to call `generate_text(clip, prompt, ...)` **so that** I can run LLM/VLM inference and receive the generated string directly in my Python script.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.textgen` module exists and exports `generate_text`.
- [ ] Signature: `generate_text(clip: Any, prompt: str, *, image: Any | None = None, max_length: int = 256, do_sample: bool = True, temperature: float = 0.7, top_k: int = 64, top_p: float = 0.95, min_p: float = 0.05, repetition_penalty: float = 1.05, seed: int = 0) -> str`.
- [ ] Internally calls `clip.tokenize()`, `clip.generate()`, and `clip.decode()` — mirroring `TextGenerate.execute()` without depending on ComfyUI's node wiring.
- [ ] Returns the decoded string (no trailing whitespace beyond what `clip.decode` produces).
- [ ] When `do_sample=False`, sampling parameters (`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `seed`) are not forwarded (matches node `sampling_mode="off"` behaviour).
- [ ] No `torch` or `comfy.*` imports at module top level.
- [ ] Typecheck / lint passes.

### US-003: Generate an LTX-Video 2 enhanced prompt with `generate_ltx2_prompt()`
**As a** developer, **I want** to call `generate_ltx2_prompt(clip, prompt, ...)` **so that** I can refine a raw user prompt into a detailed video-generation prompt suitable for LTX-Video 2, using the LTX2 system prompt template automatically.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.textgen` exports `generate_ltx2_prompt`.
- [ ] Signature matches `generate_text` exactly (same parameters, same return type).
- [ ] When `image=None`, the T2V system prompt template is prepended to `prompt` before calling `generate_text` (mirrors `TextGenerateLTX2Prompt.execute()` — `LTX2_T2V_SYSTEM_PROMPT`).
- [ ] When `image` is provided, the I2V system prompt template is prepended (mirrors `LTX2_I2V_SYSTEM_PROMPT`).
- [ ] The system prompt templates are copied verbatim from `comfy_extras/nodes_textgen.py` (do not paraphrase or truncate).
- [ ] Typecheck / lint passes.

## Functional Requirements
- FR-1: `comfy_diffusion/textgen.py` is the implementation file for `generate_text` and `generate_ltx2_prompt`.
- FR-2: `ModelManager.load_llm(path: str | Path) -> Any` loads an LLM via the ComfyUI CLIP loader with the appropriate type for LLM models.
- FR-3: `ModelManager.__init__` registers `models_dir/llm` in `folder_paths` under the `"llm"` key.
- FR-4: `comfy_diffusion/__init__.py` does **not** re-export `generate_text` or `generate_ltx2_prompt` at package level — callers must use `from comfy_diffusion.textgen import generate_text` (consistent with non-convenience-exported modules).
- FR-5: All lazy-import rules apply: no `torch` or `comfy.*` at module top level in `textgen.py`.
- FR-6: Tests live in `tests/test_textgen.py` and must pass on CPU-only CI (use mocks/stubs for `clip` object — no real model weights required).

## Non-Goals (Out of Scope)
- Loading or wrapping any LLM nodes beyond `TextGenerate` and `TextGenerateLTX2Prompt`.
- Streaming / token-by-token output.
- Quantization, GGUF, or model-format conversion helpers.
- Fine-tuning or training utilities.
- A high-level `TextPipeline` abstraction (no pipeline pattern — callers compose directly).
- Changes to the vendored ComfyUI submodule.

## Open Questions
- None.
