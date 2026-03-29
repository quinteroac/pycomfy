# Requirement: Expose Phase 4 Secondary Nodes

## Context

The comfy-diffusion library has successfully wrapped core LTX sampling and VAE nodes (iterations 000001–000026). Phase 4 of the roadmap calls for exposing the secondary utility and ControlNet nodes that are required by the remaining LTX2 ControlNet workflows (canny, depth, pose) and all LTX23 workflows. Without these building blocks, callers cannot compose those pipelines. This iteration adds 10 wrappers spread across `image.py`, `latent.py`, `controlnet.py`, and a new `textgen.py` entry point, following the established lazy-import pattern.

## Goals

- Expose all Phase 4 nodes as importable Python functions in the appropriate `comfy_diffusion` submodules.
- Follow the existing lazy-import pattern (no top-level `comfy.*` / `torch` imports).
- Maintain CPU-only test coverage so CI remains green with no GPU required.

## User Stories

### US-001: Resize image + mask together (ResizeImageMaskNode)

**As a** developer writing an LTX i2v or ControlNet pipeline, **I want** a `resize_image_mask(image, mask, width, height, interpolation)` function **so that** I can resize both the input image and its mask to the required dimensions in a single call.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.resize_image_mask(image, mask, width, height, interpolation="bilinear")` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_post_processing.ResizeImageMaskNode`.
- [ ] Returns a `(image_tensor, mask_tensor)` tuple.
- [ ] CPU pytest passes with mocked `ResizeImageMaskNode`.
- [ ] Typecheck / lint passes.

---

### US-002: Resize images by longer edge (ResizeImagesByLongerEdge)

**As a** developer preparing images for LTX i2v or LTX23 workflows, **I want** a `resize_images_by_longer_edge(images, size)` function **so that** the image's longer dimension is scaled to `size` while preserving aspect ratio.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.resize_images_by_longer_edge(images, size)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_dataset.ResizeImagesByLongerEdgeNode`.
- [ ] Returns an image tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-003: Create a solid-color blank image (EmptyImage)

**As a** developer building an LTX t2v, i2v, or LoRA pipeline, **I want** an `empty_image(width, height, batch_size, color)` function **so that** I can create a blank image tensor as a placeholder or starting frame.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.empty_image(width, height, batch_size=1, color=0)` is callable.
- [ ] The function lazily imports `nodes.EmptyImage` via `ensure_comfyui_on_path()`.
- [ ] Returns an image tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-004: Evaluate a math expression (ComfyMathExpression)

**As a** developer building an LTX23 pipeline that requires dynamic parameter computation, **I want** a `math_expression(expression, **variables)` function **so that** I can evaluate parameterised numeric expressions at runtime (e.g. deriving frame counts from duration and fps).

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.math_expression(expression: str, **kwargs: float)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_math.MathExpressionNode`.
- [ ] Returns a numeric value (int or float).
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-005: Split video into frames + audio (GetVideoComponents)

**As a** developer running canny, depth, or pose ControlNet workflows, **I want** a `get_video_components(video)` function **so that** I can decompose a loaded video tensor into its frame images and audio track for per-frame processing.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.video.get_video_components(video)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_video.GetVideoComponents`.
- [ ] Returns a `(images_tensor, audio)` tuple (matching the node's output).
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-006: Inject guide frames for LTX ControlNet (LTXVAddGuide)

**As a** developer building an LTX canny, depth, or pose workflow, **I want** an `ltxv_add_guide(conditioning, image, mask, strength, start_percent, end_percent)` function **so that** I can attach guide-frame conditioning for spatially controlled video generation.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.controlnet.ltxv_add_guide(conditioning, image, mask, strength, start_percent, end_percent)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_lt.LTXVAddGuide`.
- [ ] Returns a conditioning tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-007: Edge detection via Canny (Canny)

**As a** developer preparing control images for LTX canny-to-video, **I want** a `canny(image, low_threshold, high_threshold)` function **so that** I can generate Canny edge maps directly from a PIL/tensor image.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.canny(image, low_threshold=100, high_threshold=200)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_canny.Canny`.
- [ ] Returns an image tensor of the same spatial dimensions.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-008: Depth map conditioning via Lotus (LotusConditioning)

**As a** developer building an LTX depth-to-video pipeline, **I want** a `lotus_conditioning(model, image)` function **so that** I can apply Lotus-model depth conditioning to the generation.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.controlnet.lotus_conditioning(model, image)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_lotus.LotusConditioning`.
- [ ] Returns a conditioning tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-009: Override first sigma in sampler (SetFirstSigma)

**As a** developer composing an LTX depth workflow, **I want** a `set_first_sigma(sigmas, sigma_override)` function **so that** I can adjust the first noise level of the sampler schedule for depth conditioning.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.sampling.set_first_sigma(sigmas, sigma_override)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_custom_sampler.SetFirstSigma`.
- [ ] Returns a modified sigmas tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-010: Invert an image (ImageInvert)

**As a** developer post-processing a depth map for LTX depth-to-video, **I want** an `image_invert(image)` function **so that** I can invert pixel values (1 − pixel) as required by the depth workflow.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.image_invert(image)` is callable.
- [ ] The function lazily imports `nodes.ImageInvert` via `ensure_comfyui_on_path()`.
- [ ] Returns an image tensor.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-011: LLM-based LTX23 prompt generation (TextGenerateLTX2Prompt)

**As a** developer using an LTX23 workflow, **I want** a `text_generate_ltx2_prompt(text, model, ...)` function **so that** the model's internal LLM can expand or refine prompts for LTX23.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.textgen.text_generate_ltx2_prompt(text, model, seed, ...)` is callable.
- [ ] The function lazily imports `comfy_extras.nodes_textgen.TextGenerateLTX2Prompt`.
- [ ] Returns a string or text-conditioning object as returned by the node.
- [ ] CPU pytest passes with mocked node.
- [ ] Typecheck / lint passes.

---

### US-012: CPU-only test coverage for all new wrappers

**As a** CI system, **I want** pytest tests for every new function that mock out all `comfy.*` / `torch` dependencies, **so that** all tests pass on CPU with no model files present.

**Acceptance Criteria:**
- [ ] One test file per new function group (or added to existing test files that cover the same module).
- [ ] Every new function is covered by at least one passing test.
- [ ] `uv run pytest` exits 0 with no GPU required.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- **FR-1:** `resize_image_mask` added to `comfy_diffusion/image.py`; lazily imports `comfy_extras.nodes_post_processing.ResizeImageMaskNode`.
- **FR-2:** `resize_images_by_longer_edge` added to `comfy_diffusion/image.py`; lazily imports `comfy_extras.nodes_dataset.ResizeImagesByLongerEdgeNode`.
- **FR-3:** `empty_image` added to `comfy_diffusion/image.py`; lazily imports `nodes.EmptyImage` via `ensure_comfyui_on_path()`.
- **FR-4:** `math_expression` added to `comfy_diffusion/image.py`; lazily imports `comfy_extras.nodes_math.MathExpressionNode`.
- **FR-5:** `get_video_components` added to `comfy_diffusion/video.py`; lazily imports `comfy_extras.nodes_video.GetVideoComponents`.
- **FR-6:** `ltxv_add_guide` added to `comfy_diffusion/controlnet.py`; lazily imports `comfy_extras.nodes_lt.LTXVAddGuide`.
- **FR-7:** `canny` added to `comfy_diffusion/image.py`; lazily imports `comfy_extras.nodes_canny.Canny`.
- **FR-8:** `lotus_conditioning` added to `comfy_diffusion/controlnet.py`; lazily imports `comfy_extras.nodes_lotus.LotusConditioning`.
- **FR-9:** `set_first_sigma` added to `comfy_diffusion/sampling.py`; lazily imports `comfy_extras.nodes_custom_sampler.SetFirstSigma`.
- **FR-10:** `image_invert` added to `comfy_diffusion/image.py`; lazily imports `nodes.ImageInvert` via `ensure_comfyui_on_path()`.
- **FR-11:** `text_generate_ltx2_prompt` added to `comfy_diffusion/textgen.py`; lazily imports `comfy_extras.nodes_textgen.TextGenerateLTX2Prompt`.
- **FR-12:** All new functions follow the `str | Path` annotation convention for any path parameters.
- **FR-13:** No top-level `torch`, `comfy.*`, or `comfy_extras.*` imports in any modified module — all deferred to call time.
- **FR-14:** All new functions have a docstring with parameter descriptions and a return-type annotation.
- **FR-15:** `DWPreprocessor` (pose estimation) is **not** included — it lives in a third-party ComfyUI extension (`comfyui_controlnet_aux`) not present in the vendor submodule; it is deferred to a future iteration once that extension is vendored.

## Non-Goals (Out of Scope)

- Pipeline files (canny-to-video, depth-to-video, pose-to-video) — those are Phase 5 and will be built in a subsequent iteration.
- `DWPreprocessor` wrapper — requires the `comfyui_controlnet_aux` community extension; deferred until that extension is vendored (see FR-15).
- Updating `__init__.py` re-exports — new functions are accessible via submodule imports only (e.g. `from comfy_diffusion.image import canny`); no change to the public re-export list in `comfy_diffusion/__init__.py` unless a function qualifies as a top-level convenience export.
- GPU smoke tests — CPU pytest coverage is sufficient for this iteration.
- Pipeline folder restructure from the it_000026 audit (FR-6 of that audit) — that is a separate cleanup concern not in scope here.

## Open Questions

- None
