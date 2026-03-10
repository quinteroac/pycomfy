# Requirement: Audio Module — LTXV Audio VAE + ACE Step 1.5

## Context
Iteration 11 introduces `pycomfy/audio.py`, the audio inference module. It wraps ComfyUI's LTXV
audio VAE nodes and the ACE Step 1.5 text-to-audio conditioning nodes as clean, importable Python
functions. This is the minimum audio foundation required before the `v0.1.0-preview` release.

Stable Audio and ACE Step 1.0 are explicitly out of scope for this iteration.

## Goals
- Expose LTXV audio VAE load / encode / decode / empty-latent as pycomfy API calls.
- Expose ACE Step 1.5 text encoder loading, conditioning encoding, and empty latent creation.
- Follow all existing project conventions: lazy imports, `ModelManager.load_*` loader pattern,
  `str | Path` type annotations, CPU-safe module top level.
- All 7 use cases covered by CPU-passing pytest tests using mock objects.

## User Stories

### US-001: Load LTXV Audio VAE
**As a** Python developer, **I want** to load an LTXV audio VAE checkpoint via `ModelManager`
**so that** I can use it for audio encode/decode without manually managing ComfyUI internals.

**Acceptance Criteria:**
- [ ] `ModelManager` gains a `load_ltxv_audio_vae(path: str | Path) -> object` method.
- [ ] Method follows the lazy-import pattern (no `comfy.*` at module top level).
- [ ] Returns the loaded VAE object on success.
- [ ] `from pycomfy.models import ModelManager` and calling `mm.load_ltxv_audio_vae(path)` works.
- [ ] Typecheck / lint passes.

### US-002: Encode audio to latent using LTXV Audio VAE
**As a** Python developer, **I want** to encode a raw audio tensor into a latent representation
**so that** I can feed it into a diffusion sampler.

**Acceptance Criteria:**
- [ ] `pycomfy.audio.ltxv_audio_vae_encode(vae, audio) -> tensor` function exists.
- [ ] Wraps `LTXVAudioVAEEncode` ComfyUI node logic (calls `vae.encode(audio)`).
- [ ] Lazy import — no `torch` / `comfy.*` at module top level.
- [ ] `from pycomfy.audio import ltxv_audio_vae_encode` works.
- [ ] Typecheck / lint passes.

### US-003: Decode latent to audio using LTXV Audio VAE
**As a** Python developer, **I want** to decode a latent tensor back to audio
**so that** I can export the generated audio from a sampling run.

**Acceptance Criteria:**
- [ ] `pycomfy.audio.ltxv_audio_vae_decode(vae, latent) -> tensor` function exists.
- [ ] Wraps `LTXVAudioVAEDecode` ComfyUI node logic (calls `vae.decode(latent)`).
- [ ] Lazy import.
- [ ] `from pycomfy.audio import ltxv_audio_vae_decode` works.
- [ ] Typecheck / lint passes.

### US-004: Create empty audio latent for LTXV
**As a** Python developer, **I want** to create a blank audio latent of a given duration
**so that** I can use it as the starting noise for LTXV audio generation.

**Acceptance Criteria:**
- [ ] `pycomfy.audio.ltxv_empty_latent_audio(audio_vae, frames_number: int, frame_rate: int = 25, batch_size: int = 1) -> dict` function exists.
- [ ] Wraps `LTXVEmptyLatentAudio` node logic; requires `audio_vae` to compute latent dims via `audio_vae.num_of_latents_from_frames`; returns `{"samples": tensor, "sample_rate": int, "type": "audio"}`.
- [ ] Lazy import.
- [ ] `from pycomfy.audio import ltxv_empty_latent_audio` works.
- [ ] Typecheck / lint passes.

### US-005: Load LTXV text encoder (LTXAVTextEncoderLoader)
**As a** Python developer, **I want** to load the LTXAV text encoder via `ModelManager`
**so that** I can encode text prompts for ACE Step 1.5 conditioning.

**Acceptance Criteria:**
- [ ] `ModelManager` gains a `load_ltxav_text_encoder(path: str | Path) -> object` method.
- [ ] Follows the lazy-import pattern.
- [ ] Returns the loaded text encoder object.
- [ ] `from pycomfy.models import ModelManager` and calling `mm.load_ltxav_text_encoder(path)` works.
- [ ] Typecheck / lint passes.

### US-006: Encode text + audio metadata into ACE Step 1.5 conditioning
**As a** Python developer, **I want** to encode a text prompt and audio metadata (duration,
sample rate, etc.) into a conditioning tensor
**so that** I can guide ACE Step 1.5 audio generation.

**Acceptance Criteria:**
- [ ] `pycomfy.audio.encode_ace_step_15_audio(clip, tags: str, lyrics: str = "", seed: int = 0, bpm: int = 120, duration: float = 120.0, timesignature: str = "4", language: str = "en", keyscale: str = "C major", generate_audio_codes: bool = True, cfg_scale: float = 2.0, temperature: float = 0.85, top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0) -> conditioning` function exists.
- [ ] Wraps `TextEncodeAceStepAudio1.5` node logic (calls `clip.tokenize(tags, lyrics=lyrics, bpm=bpm, ...)` then `clip.encode_from_tokens_scheduled(tokens)`). No `sample_rate` parameter.
- [ ] Lazy import.
- [ ] `from pycomfy.audio import encode_ace_step_15_audio` works.
- [ ] Typecheck / lint passes.

### US-007: Create empty audio latent for ACE Step 1.5
**As a** Python developer, **I want** to create a blank audio latent sized for ACE Step 1.5
**so that** I can use it as the starting noise input to the sampler.

**Acceptance Criteria:**
- [ ] `pycomfy.audio.empty_ace_step_15_latent_audio(seconds: float, batch_size: int = 1) -> dict` function exists.
- [ ] Wraps `EmptyAceStep1.5LatentAudio` logic: `length = round(seconds * 48000 / 1920)`, shape `[batch, 64, length]`. Returns `{"samples": tensor, "type": "audio"}`.
- [ ] Lazy import.
- [ ] `from pycomfy.audio import empty_ace_step_15_latent_audio` works.
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: A new `pycomfy/audio.py` module is created as the single public namespace for all audio
  helper functions in this iteration.
- FR-2: Two new loader methods are added to `ModelManager` in `pycomfy/models.py`:
  `load_ltxv_audio_vae` and `load_ltxav_text_encoder`.
- FR-3: All audio functions use the lazy-import pattern — no `torch`, `comfy.*`, or
  `ensure_comfyui_on_path()` at module top level.
- FR-4: All path arguments use the `str | Path` type annotation consistent with the rest of the
  codebase.
- FR-5: `pycomfy/__init__.py` is NOT modified — consumers import directly from
  `pycomfy.audio` or `pycomfy.models`.
- FR-6: A new `tests/test_audio.py` file provides CPU-passing tests for all 7 use cases using
  mock objects (no real model files required).
- FR-7: All existing tests continue to pass without regression.
- FR-8: Mapped ComfyUI nodes: `LTXVAudioVAELoader`, `LTXVAudioVAEEncode`, `LTXVAudioVAEDecode`,
  `LTXVEmptyLatentAudio`, `LTXAVTextEncoderLoader`, `TextEncodeAceStepAudio1.5`,
  `EmptyAceStep1.5LatentAudio`.

## Non-Goals (Out of Scope)

- Stable Audio (`ConditioningStableAudio`, `AudioEncoderLoader`, `AudioEncoderEncode`,
  `EmptyLatentAudio`, `VAEEncodeAudio`, `VAEDecodeAudio`, `VAEDecodeAudioTiled`) — deferred.
- ACE Step 1.0 (`TextEncodeAceStepAudio`, `EmptyAceStepLatentAudio`) — deferred.
- Audio I/O utilities (`LoadAudio`, `SaveAudio`, torchaudio helpers) — handled by external
  libraries per project convention; not wrapped here.
- Any high-level pipeline abstraction combining sampling + audio.
- Changes to `pyproject.toml` extras or documentation beyond code comments.

## Open Questions — Resolved

**Q1: What exact signature does `TextEncodeAceStepAudio1.5` expect?**

Resolved via `comfy_extras/nodes_ace.py`. There is **no** `sample_rate` parameter. The full
parameter list for `encode_ace_step_15_audio` must be:

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `clip` | object | — | text encoder from `load_ltxav_text_encoder` |
| `tags` | str | — | genre / style description |
| `lyrics` | str | `""` | song lyrics |
| `seed` | int | `0` | |
| `bpm` | int | `120` | range 10–300 |
| `duration` | float | `120.0` | seconds |
| `timesignature` | str | `"4"` | one of `"2"`, `"3"`, `"4"`, `"6"` |
| `language` | str | `"en"` | ISO code, 23 options |
| `keyscale` | str | `"C major"` | `"<root> major\|minor"` |
| `generate_audio_codes` | bool | `True` | enables LLM audio code generation |
| `cfg_scale` | float | `2.0` | advanced |
| `temperature` | float | `0.85` | advanced |
| `top_p` | float | `0.9` | advanced |
| `top_k` | int | `0` | advanced |
| `min_p` | float | `0.0` | advanced |

Internally calls `clip.tokenize(tags, lyrics=lyrics, bpm=bpm, ...)` then
`clip.encode_from_tokens_scheduled(tokens)`.

**Q2: Does `LTXVAudioVAELoader` load from a standalone checkpoint or reuse the main LTXV checkpoint?**

Resolved via `comfy_extras/nodes_lt_audio.py`. It loads from a **standalone, dedicated audio VAE
checkpoint file** (`checkpoints/` folder). It is completely independent of the main LTXV video
checkpoint. The loader does:
```python
sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
return AudioVAE(sd, metadata)
```

**Q3: Does `LTXAVTextEncoderLoader` differ meaningfully from `ModelManager.load_clip()`?**

Yes — it requires **two separate files**: a text encoder file (from `text_encoders/`) and a
checkpoint file (from `checkpoints/`). It always uses `CLIPType.LTXV` and passes both paths to
`comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], ...)`. The existing
`ModelManager.load_clip()` only accepts a single path. A dedicated method
`load_ltxav_text_encoder(text_encoder_path, checkpoint_path)` is required — it cannot be an alias.

**Additional finding — `LTXVEmptyLatentAudio` signature:**

The node takes `frames_number: int`, `frame_rate: int`, and `audio_vae` (required) — **not**
`seconds`. It calls `audio_vae.num_of_latents_from_frames(frames_number, frame_rate)` to determine
latent dimensions. The pycomfy wrapper `ltxv_empty_latent_audio` should mirror this signature:
`ltxv_empty_latent_audio(audio_vae, frames_number: int, frame_rate: int = 25, batch_size: int = 1)`.

**Additional finding — `EmptyAceStep15LatentAudio` latent shape:**

Uses `seconds * 48000 / 1920` to compute length. Shape is `[batch, 64, length]` (3-D, unlike the
LTXV 4-D audio latent `[batch, z_channels, num_latents, freq_bins]`). No VAE dependency.

**Additional finding — `LTXVAudioVAEDecode` output format:**

Returns `{"waveform": tensor, "sample_rate": int}` (not `{"samples": ...}`). The
`ltxv_audio_vae_decode` function should return this same dict shape so callers can pass it to
torchaudio / external audio libs directly.

**Additional finding — `LTXVAudioVAEEncode` return format:**

Returns `{"samples": latents, "sample_rate": int, "type": "audio"}` — a latent dict that is
compatible with the `sampling.sample()` input convention.
