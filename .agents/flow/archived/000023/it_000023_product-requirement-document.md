# Requirement: WAN Nodes — Full Coverage

## Context

`comfy-diffusion` already exposes `wan_image_to_video`, `wan_first_last_frame_to_video`, and
`wan_vace_to_video` (the last one implemented but not exported). The ComfyUI vendor at
`vendor/ComfyUI/comfy_extras/nodes_wan.py` and `nodes_camera_trajectory.py` ships 15+ additional
WAN-specific conditioning/latent nodes that are entirely absent from the library. Developers who
build WAN-based video pipelines (Fun, VACE, Camera, Phantom, Sound, Track, HuMo, Animate, SCAIL,
InfiniteTalk, WAN 2.2) cannot use them without falling back to raw ComfyUI, which defeats the
purpose of `comfy-diffusion`.

## Goals

- Expose all missing WAN nodes as importable Python functions following the lazy-import, `str | Path`
  type-annotation, and `__all__` export conventions already established in `conditioning.py` and
  `latent.py`.
- Fix the existing `wan_vace_to_video` export gap.
- Keep every new function CPU-testable (mock VAE / mock tensors; no GPU required).

## User Stories

Each story maps one-to-one with a logical group of related nodes.

---

### US-001: Export `wan_vace_to_video`

**As a** developer, **I want** `wan_vace_to_video` to be importable via
`from comfy_diffusion.conditioning import wan_vace_to_video` **so that** I can use VACE-based
control conditioning without reaching into private internals.

**Acceptance Criteria:**
- [ ] `wan_vace_to_video` is added to `__all__` in `conditioning.py`
- [ ] `from comfy_diffusion.conditioning import wan_vace_to_video` works without error
- [ ] Typecheck / lint passes

---

### US-002: WAN Fun Control (`wan_fun_control_to_video`, `wan22_fun_control_to_video`)

**As a** developer, **I want** `wan_fun_control_to_video` and `wan22_fun_control_to_video`
**so that** I can build WAN Fun / WAN 2.2 Fun controllable video pipelines with optional start
image and control video inputs.

**Acceptance Criteria:**
- [ ] `wan_fun_control_to_video(positive, negative, vae, width, height, length, batch_size, *, clip_vision_output=None, start_image=None, control_video=None)` is implemented in `conditioning.py`, mirroring `WanFunControlToVideo.execute`
- [ ] `wan22_fun_control_to_video(positive, negative, vae, width, height, length, batch_size, *, ref_image=None, control_video=None)` is implemented, mirroring `Wan22FunControlToVideo.execute`
- [ ] Both functions use lazy imports via a dedicated helper or reuse the existing `_get_wan_vace_dependencies()`
- [ ] Both exported in `__all__`
- [ ] Unit test: CPU mock verifies return type is `tuple[any, any, dict]` (positive, negative, latent)
- [ ] Typecheck / lint passes

---

### US-003: WAN Fun Inpaint (`wan_fun_inpaint_to_video`)

**As a** developer, **I want** `wan_fun_inpaint_to_video` **so that** I can run WAN Fun inpaint
conditioning (first/last frame with optional CLIP vision) through a single function call.

**Acceptance Criteria:**
- [ ] `wan_fun_inpaint_to_video(positive, negative, vae, width, height, length, batch_size, *, clip_vision_output=None, start_image=None, end_image=None)` implemented in `conditioning.py`, mirroring `WanFunInpaintToVideo.execute` (thin wrapper over `wan_first_last_frame_to_video`)
- [ ] Exported in `__all__`
- [ ] Typecheck / lint passes

---

### US-004: WAN Camera (`wan_camera_embedding`, `wan_camera_image_to_video`)

**As a** developer, **I want** `wan_camera_embedding` and `wan_camera_image_to_video` **so that**
I can generate camera trajectory embeddings and apply them to WAN image-to-video conditioning.

**Acceptance Criteria:**
- [ ] `wan_camera_embedding(camera_pose, width, height, length, *, speed=1.0, fx=0.5, fy=0.5, cx=0.5, cy=0.5)` implemented, mirroring `WanCameraEmbedding.execute`; `camera_pose` accepts one of: `"Static"`, `"Pan Up"`, `"Pan Down"`, `"Pan Left"`, `"Pan Right"`, `"Zoom In"`, `"Zoom Out"`, `"Anti Clockwise (ACW)"`, `"ClockWise (CW)"`
- [ ] Returns `(camera_embedding, width, height, length)` tuple
- [ ] `wan_camera_image_to_video(positive, negative, vae, width, height, length, batch_size, *, clip_vision_output=None, start_image=None, camera_conditions=None)` implemented, mirroring `WanCameraImageToVideo.execute`
- [ ] Both use lazy imports; both exported in `__all__`
- [ ] `wan_camera_embedding` may live in `conditioning.py` (preferred) or a new `camera.py` module — decision recorded in docstring
- [ ] Unit test: CPU mock verifies `wan_camera_image_to_video` returns `(positive, negative, latent_dict)`
- [ ] Typecheck / lint passes

---

### US-005: WAN Phantom Subject (`wan_phantom_subject_to_video`)

**As a** developer, **I want** `wan_phantom_subject_to_video` **so that** I can animate a subject
image within a WAN video using Phantom subject conditioning.

**Acceptance Criteria:**
- [ ] `wan_phantom_subject_to_video(positive, negative, vae, width, height, length, batch_size, *, images=None)` implemented, mirroring `WanPhantomSubjectToVideo.execute`
- [ ] Returns `(positive, negative_text, negative_img_text, latent)` — four-tuple (Phantom produces two distinct negative conditionings)
- [ ] Exported in `__all__`
- [ ] Typecheck / lint passes

---

### US-006: WAN Track to Video (`wan_track_to_video`)

**As a** developer, **I want** `wan_track_to_video` **so that** I can drive WAN video generation
with point-track / trajectory conditioning (motion tracking).

**Acceptance Criteria:**
- [ ] `wan_track_to_video(positive, negative, vae, tracks, width, height, length, batch_size, *, temperature=220.0, topk=2, start_image=None, clip_vision_output=None)` implemented, mirroring `WanTrackToVideo.execute`; `tracks` accepts a JSON string (list of `{x, y}` points per track)
- [ ] When `tracks` is empty / invalid JSON, falls back to `wan_image_to_video` behavior (same as the ComfyUI node)
- [ ] Helper functions `parse_json_tracks`, `process_tracks`, `pad_pts`, `patch_motion` ported into `conditioning.py` as module-private helpers; not in `__all__`
- [ ] Exported in `__all__`
- [ ] Unit test: empty tracks string → returns valid `(positive, negative, latent_dict)`
- [ ] Typecheck / lint passes

---

### US-007: WAN Sound nodes (`wan_sound_image_to_video`, `wan_sound_image_to_video_extend`, `wan_humo_image_to_video`)

**As a** developer, **I want** WAN audio-driven conditioning functions **so that** I can build
sound-to-video and talking-head (HuMo) pipelines.

**Acceptance Criteria:**
- [ ] `wan_sound_image_to_video(positive, negative, vae, width, height, length, batch_size, *, audio_encoder_output=None, ref_image=None, control_video=None, ref_motion=None)` implemented, mirroring `WanSoundImageToVideo.execute`
- [ ] `wan_sound_image_to_video_extend(positive, negative, vae, length, video_latent, *, audio_encoder_output=None, ref_image=None, control_video=None)` implemented, mirroring `WanSoundImageToVideoExtend.execute`
- [ ] `wan_humo_image_to_video(positive, negative, vae, width, height, length, batch_size, *, audio_encoder_output=None, ref_image=None)` implemented, mirroring `WanHuMoImageToVideo.execute`
- [ ] Internal helpers `wan_sound_to_video`, `linear_interpolation`, `get_audio_embed_bucket_fps`, `get_sample_indices`, `get_audio_emb_window` ported as module-private; not in `__all__`
- [ ] All three public functions exported in `__all__`
- [ ] Typecheck / lint passes

---

### US-008: WAN Animate / InfiniteTalk / SCAIL (`wan_animate_to_video`, `wan_infinite_talk_to_video`, `wan_scail_to_video`)

**As a** developer, **I want** the remaining WAN generative-conditioning functions **so that**
I have full coverage of all WAN nodes in `nodes_wan.py`.

**Acceptance Criteria:**
- [ ] `wan_animate_to_video(...)` implemented, mirroring `WanAnimateToVideo.execute`; signature matches node inputs exactly
- [ ] `wan_infinite_talk_to_video(...)` implemented, mirroring `WanInfiniteTalkToVideo.execute`
- [ ] `wan_scail_to_video(...)` implemented, mirroring `WanSCAILToVideo.execute`
- [ ] All three exported in `__all__`
- [ ] Typecheck / lint passes

---

### US-009: WAN 2.2 Image-to-Video Latent (`wan22_image_to_video_latent`)

**As a** developer, **I want** `wan22_image_to_video_latent` **so that** I can build WAN 2.2
image-to-video latent tensors (distinct from WAN 2.1 layout).

**Acceptance Criteria:**
- [ ] `wan22_image_to_video_latent(positive, negative, vae, width, height, length, batch_size, *, start_image=None, end_image=None, clip_vision_output=None)` implemented, mirroring `Wan22ImageToVideoLatent.execute`
- [ ] Exported in `__all__`
- [ ] Typecheck / lint passes

---

### US-010: Trim Video Latent (`trim_video_latent`)

**As a** developer, **I want** `trim_video_latent(samples, trim_amount)` in `latent.py`
**so that** I can trim the leading frames added by VACE reference image padding.

**Acceptance Criteria:**
- [ ] `trim_video_latent(samples: dict, trim_amount: int) -> dict` implemented in `latent.py`, mirroring `TrimVideoLatent.execute`
- [ ] Exported in `latent.__all__`
- [ ] Unit test: given `{"samples": torch.zeros([1,16,5,8,8])}` and `trim_amount=2`, output tensor has shape `[1,16,3,8,8]`
- [ ] Typecheck / lint passes

---

## Functional Requirements

- FR-1: All new functions follow the lazy-import pattern — no `torch`, `comfy.*`, or `numpy` at module top level; imports deferred inside function bodies or private helpers.
- FR-2: Type annotations use `str | Path` for path arguments (none expected here); all other args typed with standard Python types or `Any` for opaque ComfyUI internal types.
- FR-3: All public functions are added to `__all__` in their respective module (`conditioning.py` or `latent.py`).
- FR-4: Internal helpers ported from `nodes_wan.py` (`wan_sound_to_video`, `patch_motion`, `parse_json_tracks`, etc.) are not exported — they remain module-private.
- FR-5: `wan_camera_embedding` depends on camera math helpers from `nodes_camera_trajectory.py` (`get_camera_motion`, `process_pose_params`, `CAMERA_DICT`); these must be accessible at call time via lazy import of the vendor module.
- FR-6: No new top-level dependencies introduced — `numpy` (already a transitive dep) is acceptable for `wan_track_to_video` and `wan_camera_embedding`.
- FR-7: All tests must pass on CPU-only CI (`uv run pytest`).

## Non-Goals (Out of Scope)

- Re-exporting new symbols at the `comfy_diffusion` package level (`__init__.py`).
- Adding high-level pipeline wrappers that compose multiple WAN functions.
- Porting WAN nodes from discarded modules (`nodes_train.py`, `nodes_dataset.py`, etc.).
- Adding new optional extras to `pyproject.toml`.
- Updating `ROADMAP.md` or bumping the package version (deferred to approve-prototype step).

## Open Questions

- None
