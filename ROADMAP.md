# comfy-diffusion Roadmap

## Iteration Plan

| # | Module | Goal | Status |
|---|--------|-------|--------|
| 01 | `_runtime` / `check_runtime()` | Package foundation + ComfyUI vendoring | ✅ Done |
| 02 | `models` | Checkpoint loading (`ModelManager`, `CheckpointResult`) | ✅ Done |
| 03 | `conditioning` | Prompt encoding via `encode_prompt` | ✅ Done |
| 04 | `sampling` | KSampler wrapper via `sample()` | ✅ Done |
| 05 | `vae` | VAE decode latent→PIL via `vae_decode()` | ✅ Done |
| 06 | `lora` | LoRA loading and stacking via `apply_lora()` | ✅ Done |
| 07 | `vae` + `models` | VAE encode image→latent (`vae_encode`) + standalone loaders (`load_vae`, `load_clip`, `load_unet`) on `ModelManager` | ✅ Done |
| 08 | `vae` — tiled | `vae_decode_tiled`, `vae_encode_tiled` for large images without OOM | ⬜ Next |
| 09 | `vae` — batch/video | `vae_decode_batch`, `vae_encode_batch` for video frame sequences (WAN, LTX2) | ⬜ |
| 10 | `sampling` — advanced | `KSamplerAdvanced`, `SamplerCustomAdvanced`, guiders, schedulers, sigma manipulation | ⬜ |
| 11 | `audio` | Stable Audio, WAN sound-to-video, LTXV audio, ACE Step text-to-audio | ⬜ |
| — | **`v0.1.0-preview`** | **Preview release — ACE Step Studio minimum viable stack. Publish after it_11.** | ⬜ |
| 12 | `conditioning` — advanced | `ConditioningCombine`, `ConditioningSetMask`, `ConditioningSetTimestepRange`, Flux, WAN, LTXV | ⬜ |
| 13 | `controlnet` | `ControlNetLoader`, `ControlNetApplyAdvanced`, `SetUnionControlNetType` | ⬜ |
| 14 | `latent` — utilities | `EmptyLatentImage`, `LatentUpscale`, `LatentCrop`, `LatentComposite`, batch ops, video ops | ⬜ |
| 15 | `image` — utilities | `LoadImage`, `ImagePadForOutpaint`, `ImageUpscaleWithModel`, video I/O, WAN/LTXV img2vid | ⬜ |
| 16 | `mask` | `LoadImageMask`, `ImageToMask`, `MaskToImage`, `GrowMask`, `FeatherMask`, inpaint masks | ⬜ |
| 17 | `model` — patches | `ModelSamplingFlux`, `ModelSamplingSD3`, `ModelSamplingAuraFlow`, video CFG guidance | ⬜ |
| 18 | packaging | pip-installable, type stubs, DX polish, extras (`[video]`, `[audio]`, `[all]`) | ⬜ |

---

## Node Inventory

### Already covered (it_01–07)

| comfy-diffusion API | ComfyUI node |
|-------------|-------------|
| `vae_decode()` | VAEDecode |
| `vae_encode()` | VAEEncode |
| `sample()` | KSampler |
| `encode_prompt()` | CLIPTextEncode |
| `ModelManager.load_checkpoint()` | CheckpointLoaderSimple |
| `ModelManager.load_unet()` | UNETLoader |
| `ModelManager.load_vae()` | VAE loader (comfy.sd) |
| `ModelManager.load_clip()` | CLIP loader (comfy.sd) |
| `apply_lora()` | LoraLoader / LoraLoaderModelOnly |

---

### Roadmap nodes

**Latent**
`EmptyLatentImage`, `SetLatentNoiseMask`, `LatentUpscale`, `LatentUpscaleBy`, `LatentCrop`, `LatentFromBatch`, `RepeatLatentBatch`, `LatentConcat`, `LatentCutToBatch`, `ReplaceVideoLatentFrames`, `LatentComposite`, `LatentCompositeMasked`, `VAEEncodeForInpaint`, `InpaintModelConditioning`, `VAEDecodeTiled`, `VAEEncodeTiled`

**Image**
`LoadImage`, `ImagePadForOutpaint`, `ImageFromBatch`, `RepeatImageBatch`, `ImageCompositeMasked`, `ImageUpscaleWithModel`, `GetVideoComponents`, `CreateVideo`, `SaveWEBM`, `SaveVideo`, `LoadVideo`, `LTXVImgToVideo`, `LTXVPreprocess`, `WanImageToVideo`, `WanFirstLastFrameToVideo`

**Mask**
`LoadImageMask`, `ImageToMask`, `MaskToImage`, `SolidMask`, `InvertMask`, `GrowMask`, `FeatherMask`, `CropMask`

**Conditioning**
`ConditioningCombine`, `ConditioningSetMask`, `ConditioningSetTimestepRange`, `CLIPVisionEncode`, `StyleModelApply`, `CLIPTextEncodeFlux`, `FluxGuidance`, `WanImageToVideo`, `WanFirstLastFrameToVideo`, `WanFunInpaintToVideo`, `LTXVImgToVideo`, `LTXVConditioning`

**ControlNet**
`ControlNetLoader`, `DiffControlNetLoader`, `ControlNetApplyAdvanced`, `SetUnionControlNetType`

**Sampling**
`KSamplerAdvanced`, `SamplerCustomAdvanced`, `BasicGuider`, `CFGGuider`, `RandomNoise`, `DisableNoise`, `BasicScheduler`, `KarrasScheduler`, `AlignYourStepsScheduler`, `Flux2Scheduler`, `LTXVScheduler`, `SplitSigmas`, `SplitSigmasDenoise`, `KSamplerSelect`, `SamplerDPMPP_3M_SDE`, `SamplerDPMPP_2M_SDE`, `SamplerEulerAncestral`, `VideoLinearCFGGuidance`, `VideoTriangleCFGGuidance`

**Model**
`ModelSamplingFlux`, `ModelSamplingSD3`, `ModelSamplingAuraFlow`, `VideoLinearCFGGuidance`, `VideoTriangleCFGGuidance`

**Audio**
`VAEEncodeAudio`, `VAEDecodeAudio`, `VAEDecodeAudioTiled`, `EmptyLatentAudio`, `ConditioningStableAudio`, `AudioEncoderLoader`, `AudioEncoderEncode`, `LTXVAudioVAELoader`, `LTXVAudioVAEEncode`, `LTXVAudioVAEDecode`, `LTXVEmptyLatentAudio`, `LTXAVTextEncoderLoader`, `TextEncodeAceStepAudio`, `TextEncodeAceStepAudio1.5`, `EmptyAceStepLatentAudio`, `EmptyAceStep1.5LatentAudio`

---

### Nice-to-have nodes

**Latent**
`SaveLatent`, `LoadLatent`, `LatentFlip`, `LatentRotate`, `LatentBatchSeedBehavior`, `LatentCut`, `LatentBlend`, `LatentInterpolate`, `LatentAdd`, `LatentSubtract`, `LatentMultiply`, `LTXVLatentUpsampler`, `SVD_img2vid_Conditioning`

**Image**
`EmptyImage`, `ImageScaleToTotalPixels`, `ImageFlip`, `ImageRotate`, `ImageStitch`, `SplitImageToTileList`, `ImageMergeTileList`, `BatchImagesNode`, `ImageColorToMask`, `SaveAnimatedWEBP`, `SaveAnimatedPNG`, `WanVaceToVideo`, `WanCameraImageToVideo`, `WanPhantomSubjectToVideo`, `LTXVAddGuide`

**Mask**
`ImageColorToMask`, `ThresholdMask`, `MaskComposite`

**Conditioning**
`ConditioningAverage`, `ConditioningConcat`, `ConditioningSetArea`, `ConditioningSetAreaPercentage`, `ConditioningSetAreaStrength`, `ConditioningZeroOut`, `ControlNetInpaintingAliMamaApply`, `unCLIPConditioning`, `CLIPTextEncodeSD3`, `CLIPTextEncodeHunyuanDiT`, `WanVaceToVideo`, `WanCameraImageToVideo`, `WanPhantomSubjectToVideo`, `LTXVAddGuide`

**Sampling**
`SamplerCustom`, `DualCFGGuider`, `AddNoise`, `SDTurboScheduler`, `ExponentialScheduler`, `PolyexponentialScheduler`, `LaplaceScheduler`, `BetaSamplingScheduler`, `GITSScheduler`, `OptimalStepsScheduler`, `FlipSigmas`, `SetFirstSigma`, `SamplingPercentToSigma`, `SamplerDPMPP_SDE`, `SamplerDPMPP_2S_Ancestral`, `SamplerLMS`, `SamplerDPMAdaptative`, `SamplerER_SDE`, `SamplerSASolver`, `SamplerSEEDS2`, `SamplerEulerAncestralCFGPP`, `APG`, `TCFG`, `NAGuidance`

**Model**
`unCLIPCheckpointLoader`, `ImageOnlyCheckpointLoader`, `ModelSamplingDiscrete`, `ModelSamplingContinuousEDM`, `ModelSamplingContinuousV`, `RescaleCFG`, `ModelComputeDtype`, `ModelMergeSimple`, `ModelMergeBlocks`, `ModelMergeSDXL`, `ModelMergeFlux1`, `ModelMergeLTXV`

**Audio**
`EmptyAudio`, `ReferenceAudio`

---

### Discarded nodes

**Replaced by Python libraries (PIL, numpy, torch, cv2, torchaudio)**
- Image transforms: `ImageScale`, `ImageScaleBy`, `ImageScaleToMaxDimension`, `ImageCropV2`, `ResizeAndPadImage`, `ResizeImageMaskNode`, `GetImageSize`, `ImageFlip`, `ImageRotate`, `ImageBlend`, `ImageBlur`, `ImageSharpen`, `ImageInvert`, `ImageAddNoise`
- Mask ops: `InvertMask`, `SolidMask`, `CropMask`, `ThresholdMask`, `MaskComposite`
- Audio I/O: `LoadAudio`, `SaveAudio`, `SaveAudioMP3`, `SaveAudioOpus`, `PreviewAudio`, `RecordAudio`, `TrimAudioDuration`, `SplitAudioChannels`, `JoinAudioChannels`, `AudioConcat`, `AudioMerge`, `AudioAdjustVolume`, `AudioEqualizer3Band`
- Video I/O: handled by `opencv-python` / `imageio`

**UI-specific**
`SaveImage`, `PreviewImage`, `LoadImageOutput`, `WebcamCapture`, `MaskPreview`, `PreviewAudio`, `RecordAudio`

**Deprecated**
`CheckpointLoader`, `DiffusersLoader`, `ControlNetApply`, `ControlNetApplySD3`, `LatentBatch`, `ImageBatch`, `ImageCrop`

**Out of scope / niche**
`GLIGENTextBoxApply`, `unCLIPConditioning`, `SVD_img2vid_Conditioning`, `PerpNeg`, `PerpNegGuider`, `SamplerLCMUpscale`, `VPScheduler`, `ManualSigmas`, `LatentApplyOperation`, `LatentApplyOperationCFG`, `LatentOperationTonemapReinhard`, `LatentOperationSharpen`, `CLIPTextEncodePixArtAlpha`, `CLIPTextEncodeLumina2`, `CLIPTextEncodeHiDream`, `CLIPTextEncodeKandinsky5`, `StableCascade_StageB_Conditioning`, `LotusConditioning`, `PhotoMakerEncode`, `InstructPixToPixConditioning`, `TextEncodeQwenImageEdit`, `TextEncodeZImageOmni`

**Full modules discarded**
`nodes_train.py`, `nodes_dataset.py`, `nodes_hypernetwork.py`, `nodes_webcam.py`, `nodes_preview_any.py`, `nodes_nop.py`, `nodes_glsl.py`, `nodes_load_3d.py`, `nodes_stable3d.py`, `nodes_hunyuan3d.py`, `nodes_string.py`, `nodes_logic.py`, `nodes_primitive.py`, `nodes_textgen.py`, `nodes_cosmos.py`, `nodes_mochi.py`, `nodes_lotus.py`, `nodes_pixart.py`, `nodes_kandinsky5.py`, `nodes_stable_cascade.py`, `nodes_mahiro.py`, `nodes_fresca.py`, `nodes_model_patch.py`, `nodes_lora_debug.py`, `nodes_edit_model.py`, `nodes_lora_extract.py`, `nodes_model_merging_model_specific.py` (except Flux1, SDXL, LTXV in nice-to-have)

---

## Optional Dependencies

| Extra | Libraries | Use case |
|-------|-----------|----------|
| `comfy-diffusion[cuda]` | torch + CUDA | GPU inference |
| `comfy-diffusion[cpu]` | torch CPU | CPU inference |
| `comfy-diffusion[video]` | opencv-python, imageio | Video I/O, mask morphology |
| `comfy-diffusion[audio]` | torchaudio | Audio pipelines |
| `comfy-diffusion[all]` | all of the above | Full installation |