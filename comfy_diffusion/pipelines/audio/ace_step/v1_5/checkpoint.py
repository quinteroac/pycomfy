"""ACE Step 1.5 all-in-one checkpoint text-to-audio pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: checkpoint loading,
  AuraFlow model-sampling patch, latent creation, text conditioning,
  KSampler denoising, and VAE audio decoding.

This pipeline mirrors the ``audio_ace_step_1_5_checkpoint.json`` reference
workflow exactly: one checkpoint supplies the model, CLIP, and VAE in a single
file (``ace_step_1.5_turbo_aio.safetensors``).

Pattern
-------
Every pipeline module in ``comfy_diffusion/pipelines/`` follows this contract:

1. Define a ``manifest() -> list[ModelEntry]`` function that declares all
   required model files as typed ``ModelEntry`` dataclass instances.
2. Define a ``run(...)`` function that implements the full pipeline using the
   ``comfy_diffusion`` public API (lazy imports only — no top-level
   ``comfy.*`` or ``comfy_diffusion.*`` imports).
3. Keep the manifest as the single source of truth for model file paths —
   ``run()`` should derive its default paths from ``manifest()`` so the two
   stay in sync automatically.

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        tags="neo-soul, warm groove, live drums",
        lyrics="Late night glow on your skin\\nWindow cracked, city hums again",
    )
    audio = result["audio"]   # {"waveform": tensor, "sample_rate": int}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for ACE Step 1.5 all-in-one checkpoint
# ---------------------------------------------------------------------------

_HF_REPO = "Comfy-Org/ace_step_1.5_ComfyUI_files"

# Relative destination path (resolved against models_dir by download_models).
_CHECKPOINT_DEST = Path("checkpoints") / "ace_step_1.5_turbo_aio.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the ACE Step 1.5 checkpoint pipeline.

    Returns a single :class:`~comfy_diffusion.downloader.HFModelEntry` for the
    all-in-one checkpoint that bundles the diffusion model, CLIP text encoder,
    and VAE decoder.  Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="checkpoints/ace_step_1.5_turbo_aio.safetensors",
            dest=_CHECKPOINT_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    tags: str,
    lyrics: str = "",
    duration: float = 120.0,
    bpm: int = 120,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    seed: int = 0,
    steps: int = 8,
    cfg: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    checkpoint_filename: str | None = None,
) -> dict[str, Any]:
    """Run the ACE Step 1.5 all-in-one checkpoint text-to-audio pipeline.

    Mirrors the ``audio_ace_step_1_5_checkpoint.json`` reference workflow:
    ``CheckpointLoaderSimple`` → ``ModelSamplingAuraFlow(shift=3)`` →
    ``EmptyAceStep1.5LatentAudio`` → ``TextEncodeAceStepAudio1.5`` →
    ``ConditioningZeroOut`` → ``KSampler`` → ``VAEDecodeAudio``.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    tags : str
        Positive text description of the desired audio (genre tags, instruments,
        mood, etc.).
    lyrics : str, optional
        Song lyrics or spoken text to embed in the audio.  Default ``""``.
    duration : float, optional
        Target audio duration in seconds.  Default ``120.0``.
    bpm : int, optional
        Beats per minute for the generated audio.  Default ``120``.
    timesignature : str, optional
        Time signature numerator (e.g. ``"4"`` for 4/4).  Default ``"4"``.
    language : str, optional
        Language code for the lyrics (e.g. ``"en"``).  Default ``"en"``.
    keyscale : str, optional
        Key and scale descriptor (e.g. ``"C major"``, ``"E minor"``).
        Default ``"C major"``.
    seed : int, optional
        Random seed for reproducibility.  Used for both text encoding and
        KSampler noise.  Default ``0``.
    steps : int, optional
        Number of denoising steps.  Default ``8``.
    cfg : float, optional
        CFG guidance scale.  Default ``1.0``.
    sampler_name : str, optional
        Sampler algorithm name.  Default ``"euler"``.
    scheduler : str, optional
        Noise schedule name.  Default ``"simple"``.
    checkpoint_filename : str | None, optional
        Override the default checkpoint filename.  When ``None``, uses
        ``ace_step_1.5_turbo_aio.safetensors`` from the manifest.
        Default ``None``.

    Returns
    -------
    dict[str, Any]
        ``{"audio": {"waveform": tensor, "sample_rate": int}}``

        - ``audio`` — decoded audio as ``{"waveform": tensor, "sample_rate": int}``.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.audio import encode_ace_step_15_audio, empty_ace_step_15_latent_audio, vae_decode_audio
    from comfy_diffusion.conditioning import conditioning_zero_out
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve checkpoint path (allow caller override, fall back to manifest path).
    ckpt_path = models_dir / (checkpoint_filename or _CHECKPOINT_DEST)

    # Node #97 — CheckpointLoaderSimple
    result = mm.load_checkpoint(str(ckpt_path))
    model = result.model
    clip = result.clip
    vae = result.vae

    # Node #78 — ModelSamplingAuraFlow(shift=3)
    model = model_sampling_aura_flow(model, shift=3)

    # Node #98 — EmptyAceStep1.5LatentAudio(seconds=duration, batch_size=1)
    latent = empty_ace_step_15_latent_audio(duration, batch_size=1)

    # Node #94 — TextEncodeAceStepAudio1.5
    positive = encode_ace_step_15_audio(
        clip,
        tags,
        lyrics=lyrics,
        seed=seed,
        bpm=bpm,
        duration=duration,
        timesignature=timesignature,
        language=language,
        keyscale=keyscale,
    )

    # Node #47 — ConditioningZeroOut
    negative = conditioning_zero_out(positive)

    # Node #3 — KSampler(steps=8, cfg=1.0, sampler="euler", scheduler="simple")
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
    )

    # Node #18 — VAEDecodeAudio
    waveform = vae_decode_audio(vae, latent_out)
    sample_rate = getattr(vae, "audio_sample_rate", 44100)

    return {"audio": {"waveform": waveform, "sample_rate": sample_rate}}
