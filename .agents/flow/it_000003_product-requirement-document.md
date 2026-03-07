# Requirement: `conditioning` Module — Prompt Encoding via CLIP

## Context

Iterations 1 and 2 established the package foundation (`_runtime`, `check_runtime`) and model loading
(`ModelManager`, `load_checkpoint`). Iteration 3 delivers the `conditioning` module: a thin, side-effect-free
public API that accepts a loaded CLIP object and a text prompt, and returns the conditioning tensor that
ComfyUI's sampler expects. This is the bridge between model loading and sampling — nothing downstream can
run without it.

## Goals

- Expose a single `encode_prompt(clip, text)` function that wraps ComfyUI's native CLIP encoder without
  modifying or stripping the prompt.
- Ensure prompt-weighting syntax (`(word:1.3)`) is passed through unmodified and handled by ComfyUI
  internally.
- Keep the module import-safe on CPU-only environments with no model files present.
- Return the raw conditioning object produced by ComfyUI (no wrapping), so it is directly passable to
  the future `sampling` module.
- Keep static analysis (ruff, mypy) green with no regressions.

## User Stories

### US-001: Encode a text prompt to a conditioning tensor

**As a** library consumer (or an internal pycomfy module), **I want** to call `encode_prompt(clip, text)`
**so that** I receive a conditioning object ready to pass directly to ComfyUI's sampler.

**Acceptance Criteria:**
- [ ] `from pycomfy.conditioning import encode_prompt` imports without error on CPU with no model files.
- [ ] `encode_prompt(clip, "a portrait of a woman, studio lighting")` returns a non-`None` value.
- [ ] The returned object is the raw conditioning output from ComfyUI's CLIP encoder (no additional wrapping).
- [ ] No ComfyUI loaders or `torch` are imported as a side effect of `import pycomfy.conditioning`.
- [ ] `ruff check .` and `mypy pycomfy/` pass with no new errors.

---

### US-002: Positive and negative prompts share the same function

**As a** library consumer, **I want** `encode_prompt` to work identically for both positive and negative
prompts **so that** I am responsible for the distinction and the API stays minimal.

**Acceptance Criteria:**
- [ ] Calling `encode_prompt(clip, "ugly, blurry")` (negative) returns a non-`None` value using the same
  code path as a positive prompt.
- [ ] No separate `encode_negative_prompt` function or `is_negative` parameter is added to the public API.

---

### US-003: Prompt weighting syntax passes through unchanged

**As a** library consumer, **I want** weighted prompts like `"a portrait of a woman, (studio lighting:1.3)"`
to be accepted by `encode_prompt` without modification **so that** ComfyUI's native weighting logic applies.

**Acceptance Criteria:**
- [ ] `encode_prompt(clip, "a portrait of a woman, (studio lighting:1.3)")` returns a non-`None` value.
- [ ] pycomfy does not parse, strip, or transform the prompt string before passing it to the ComfyUI encoder.
- [ ] No exception is raised for valid ComfyUI weighting syntax.

---

### US-004: Empty string encodes without crashing

**As a** library consumer, **I want** `encode_prompt(clip, "")` to succeed **so that** the common
unconditional / empty negative prompt case works without special-casing.

**Acceptance Criteria:**
- [ ] `encode_prompt(clip, "")` returns a non-`None` value.
- [ ] No exception is raised for an empty string input.

---

### US-005: CPU-only import never crashes

**As a** developer running CI (CPU-only, no model files), **I want** `from pycomfy.conditioning import encode_prompt`
to succeed **so that** the module can be tested and linted in CI without GPU or model files.

**Acceptance Criteria:**
- [ ] Running `uv run python -c "from pycomfy.conditioning import encode_prompt; print('ok')"` exits with
  code 0 on a CPU-only machine with no model files present.
- [ ] No `torch`, `comfy.sd`, or any heavy loader is imported at module import time.

## Functional Requirements

- FR-1: `pycomfy/conditioning.py` exposes a public function `encode_prompt(clip: Any, text: str) -> Any`.
- FR-2: `encode_prompt` calls ComfyUI's CLIP encoder using the two-call pattern from `CLIPTextEncode`
  (`clip.tokenize(text)` → `clip.encode_from_tokens_scheduled(tokens)`), deferred to call
  time (not import time).
- FR-3: The prompt string is passed to the ComfyUI encoder unchanged — no pre-processing, stripping, or
  transformation by pycomfy.
- FR-4: `encode_prompt` accepts an empty string (`""`) and returns a valid conditioning object.
- FR-5: `pycomfy/conditioning.py` must not import `torch`, `comfy.sd`, or any model loader at module level.
- FR-6: `encode_prompt` is exported from `pycomfy/conditioning.py` and optionally re-exported from
  `pycomfy/__init__.py` (or left to explicit submodule import — implementation decision).
- FR-7: Type hints use `Any` for `clip` and the return type, consistent with the pattern established in
  `pycomfy/models.py` for ComfyUI objects not representable by a concrete type.

## Non-Goals (Out of Scope)

- Flux dual-encoder loading (CLIP L + T5-XXL as separate files) — deferred to the iteration that implements standalone CLIP loading in ModelManager.
- Prompt concatenation / long-prompt handling beyond 77 tokens — handled transparently by ComfyUI.
- Returning attention masks, hidden states, or intermediate embeddings — raw conditioning tensor only.
- A `ConditioningManager` class or stateful conditioning object — the function form is sufficient for the MVP.
- Any prompt pre-processing utilities (tokenizer inspection, prompt splitting, etc.).

## Open Questions

- ~~Should `encode_prompt` be re-exported from `pycomfy/__init__.py`?~~
  **Resolved:** Explicit submodule import only — `from pycomfy.conditioning import encode_prompt`.
  `__init__.py` stays clean and only exports `check_runtime`, consistent with the `models` pattern.

- ~~Does ComfyUI's `clip.tokenize` + `clip.encode_from_tokens` or a single combined call better reflect
  the expected upstream API?~~
  **Resolved:** Two separate calls — this is the canonical pattern from ComfyUI's `CLIPTextEncode` node
  in `nodes.py` (verified against `vendor/ComfyUI/nodes.py` line 79-80). Implementation follows:
  ```python
  tokens = clip.tokenize(text)
  conditioning = clip.encode_from_tokens_scheduled(tokens)
  ```
  Note: `encode_from_tokens(tokens, return_pooled=True)` is used only by specialized nodes (e.g.
  `GLIGENTextBoxApply`) that need pooled embeddings for spatial conditioning — not the standard
  text encoding path.
