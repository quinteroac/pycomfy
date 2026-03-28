# comfy-diffusion Roadmap

## Intention

Build production-ready pipelines powered by the workflows catalogued in `comfyui_official_workflows/`.

Each pipeline wraps one or more official ComfyUI workflows into a clean Python interface, enabling programmatic execution, parameterization, and integration with the rest of the comfy-diffusion project.

## Pipeline Development Priority

Pipelines will be developed in the following order. API-based and LLM workflows are out of scope.

### 1. Video

Covers all local video generation and editing workflows under `comfyui_official_workflows/video/`:

- **LTX** — `ltx1/`, `ltx2/`, `ltx3/`
- **WAN** — `wan2.1/`, `wan2.2/`, `fun/`, `vace/`, `ati/`, `move/`, `infinitetalk/`, `scail/`
- **HunyuanVideo** — `hunyuan/`
- **Others** — `others/`, `utility/`

### 2. Image

Covers all local image generation and editing workflows under `comfyui_official_workflows/image/`:

- **Generation** — `flux/`, `flux2/`, `flux_klein/`, `hidream/`, `qwen/`, `sdxl/`, `chroma/`, `others/`
- **Editing** — `flux/`, `flux_kontext/`, `flux2/`, `flux_klein/`, `hidream/`, `qwen/`, `others/`
- **ControlNet** — `flux/`, `qwen/`, `sd3/`, `z_image/`
- **Reference & Utility**

### 3. Audio

Covers all local audio generation workflows under `comfyui_official_workflows/audio/`:

- **ACE Step** — `v1.5/`
- **Chatterbox**
- **Stable Audio**
- **Utility**

### 4. 3D

Covers all local 3D generation workflows under `comfyui_official_workflows/3d/`.

## Development Phases

### Phase 1 — Model Downloader Module [DONE]

Implement an automatic model download module (`comfy_diffusion/downloader.py`) that resolves and fetches all models required by a given pipeline before execution. This removes the manual step of pre-downloading checkpoints, VAEs, LoRAs, and other weights, and is a prerequisite for reliable pipeline execution across environments.

---

### Phase 2 — Expose LTX Core Nodes

Expose the three nodes that block all ltx2 and ltx3 pipelines.

| Node | Scope |
|---|---|
| `LTXAVTextEncoderLoader` | All ltx2 and ltx3 workflows |
| `LTXVAudioVAELoader` | All ltx2 and ltx3 workflows |
| `LTXVImgToVideoInplace` | i2v, lora, canny, pose, all ltx3 |

---

### Phase 3 — LTX2 / LTX3 Base Pipelines

- `ltx2/video_ltx2_t2v` — Text to Video (LTX 2.0)
- `ltx2/video_ltx2_t2v_distilled` — Text to Video distilled
- `ltx2/video_ltx2_i2v` — Image to Video
- `ltx2/video_ltx2_i2v_distilled` — Image to Video distilled
- `ltx2/video_ltx2_i2v_lora` — Image to Video with LoRA
- `ltx3/video_ltx2_3_t2v` — Text to Video (LTX 2.3)
- `ltx3/video_ltx2_3_i2v` — Image to Video (LTX 2.3)
- `ltx3/video_ltx2_3_flf2v` — First-Last-Frame to Video

---

### Phase 4 — Expose Secondary Nodes

**General utility:**

| Node | Workflows |
|---|---|
| `ResizeImageMaskNode` | i2v, canny, depth, pose, ltx3 |
| `ResizeImagesByLongerEdge` | i2v, lora, ltx3 |
| `EmptyImage` | t2v, i2v, lora (ltx2) |
| `ComfyMathExpression` | all ltx3 |
| `GetVideoComponents` | canny, depth, pose |

**ControlNet:**

| Node | Notes |
|---|---|
| `LTXVAddGuide` | Guide-frame injection for controlnet conditioning |
| `Canny` | Edge detection preprocessor |
| `DWPreprocessor` | Pose estimation preprocessor |
| `LotusConditioning` | Depth map conditioning (Lotus model) |
| `SetFirstSigma` | Depth workflow sampler adjustment |
| `ImageInvert` | Depth map post-processing |

**ltx3:**

| Node | Notes |
|---|---|
| `TextGenerateLTX2Prompt` | LLM-based prompt generation |

---

### Phase 5 — Remaining LTX Workflows

- `ltx2/video_ltx2_canny_to_video`
- `ltx2/video_ltx2_depth_to_video`
- `ltx2/video_ltx2_pose_to_video`
- `ltx3/video_ltx2_3_ia2v` — Image+Audio to Video

---

### Phase 6 — Expose LTX Audio-to-Video Nodes

Experimental LTX2 nodes and third-party integrations required by `ltx2/video_ltx_2_audio_to_video`.

| Node | Notes |
|---|---|
| `LTXVAddGuideMulti` | Multi-frame guide injection |
| `LTXVAudioVAEEncode` | Audio VAE encoding |
| `LTXVAudioVideoMask` | Audio-video mask generation |
| `LTXVChunkFeedForward` | Chunk-based feed-forward processing |
| `LTXVImgToVideoInplaceKJ` | KJ-suite variant of inplace i2v |
| `LTX2_NAG` | NAG sampling variant |
| `AudioCrop`, `AudioSeparation`, `TrimAudioDuration` | Audio preprocessing utilities |
| `VHS_VideoCombine`, `VAELoaderKJ`, `ImageResizeKJv2` | Third-party (VideoHelperSuite / KJ nodes) |

---

### Phase 7 — LTX Audio-to-Video Pipeline

- `ltx2/video_ltx_2_audio_to_video`
