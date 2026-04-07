"""I/O helpers for saving pipeline outputs (images, video frames, audio).

All imports of torch, torchaudio, av, and PIL are deferred to call time to
keep the CLI module importable without heavy dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def save_image(images: list[Any], output: str) -> str:
    """Save the first PIL image to *output*; return the resolved path.

    Parameters
    ----------
    images:
        List of PIL Image objects returned by an image pipeline.
    output:
        Destination file path (e.g. ``"output.png"``).

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    if not images:
        raise ValueError("pipeline returned no images")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(out))
    return str(out.resolve())


def save_video_frames(
    frames: list[Any],
    output: str,
    fps: float = 24.0,
    audio: "dict[str, Any] | None" = None,
) -> str:
    """Encode *frames* (list of PIL images) to an MP4 file at *output*.

    Uses PyAV when available; falls back to saving individual PNG frames in a
    directory when PyAV is not installed.

    Parameters
    ----------
    frames:
        List of PIL Image objects (video frames).
    output:
        Destination MP4 file path.
    fps:
        Frame rate.
    audio:
        Optional ComfyUI AUDIO dict ``{"waveform": tensor, "sample_rate": int}``
        to mux into the MP4.  Ignored when PyAV is not available.

    Returns
    -------
    str
        Resolved path to the saved file (MP4 or directory).
    """
    if not frames:
        raise ValueError("pipeline returned no video frames")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import av  # type: ignore[import]
        from fractions import Fraction

        w, h = frames[0].size
        rate = Fraction(int(fps), 1) if fps == int(fps) else Fraction(round(fps * 1000), 1000)
        with av.open(str(out), "w") as container:
            vstream = container.add_stream("libx264", rate=rate, options={"crf": "18"})
            vstream.width = w
            vstream.height = h
            vstream.pix_fmt = "yuv420p"

            astream = None
            if audio is not None:
                import numpy as np
                sample_rate: int = audio["sample_rate"]
                waveform = audio["waveform"]  # [1, C, N] or [C, N]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)  # → [C, N]
                astream = container.add_stream("aac", rate=sample_rate)

            for i, pil_img in enumerate(frames):
                vframe = av.VideoFrame.from_image(pil_img)
                vframe.pts = i
                for packet in vstream.encode(vframe):
                    container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)

            if astream is not None:
                pcm = waveform.cpu().numpy()  # [C, N] float32
                # PyAV expects [C, N] float32 in fltp layout.
                aframe = av.AudioFrame.from_ndarray(pcm, format="fltp", layout="stereo" if pcm.shape[0] == 2 else "mono")
                aframe.sample_rate = sample_rate
                aframe.pts = 0
                for packet in astream.encode(aframe):
                    container.mux(packet)
                for packet in astream.encode():
                    container.mux(packet)

        return str(out.resolve())

    except ImportError:
        out_dir = out.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            img.save(out_dir / f"frame_{i:04d}.png")
        print(
            f"Warning: PyAV not available; saved {len(frames)} frames to {out_dir}/",
            file=sys.stderr,
        )
        return str(out_dir.resolve())


def save_audio(audio: dict[str, Any], output: str) -> str:
    """Save audio waveform tensor to *output* (WAV).

    Parameters
    ----------
    audio:
        Dict with keys ``"waveform"`` (tensor) and ``"sample_rate"`` (int).
    output:
        Destination WAV file path.

    Returns
    -------
    str
        Resolved path to the saved file.
    """
    waveform = audio["waveform"]
    sample_rate: int = audio["sample_rate"]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torchaudio  # type: ignore[import]

        torchaudio.save(str(out), waveform.cpu(), sample_rate)
    except ImportError:
        import numpy as np
        from scipy.io import wavfile  # type: ignore[import]

        wav_data = waveform.cpu().numpy() if hasattr(waveform, "numpy") else np.array(waveform)
        wav_data = (np.clip(wav_data, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(out), sample_rate, wav_data.T)

    return str(out.resolve())


def resolve_models_dir(models_dir: str | None) -> Path:
    """Resolve the models directory from the CLI flag or environment.

    Parameters
    ----------
    models_dir:
        Value from ``--models-dir`` option, or ``None`` to fall back to the
        ``PYCOMFY_MODELS_DIR`` environment variable.

    Returns
    -------
    Path
        Validated models directory path.

    Raises
    ------
    SystemExit
        If the resolved path is missing or not a directory.
    """
    import os

    resolved = models_dir or os.environ.get("PYCOMFY_MODELS_DIR")
    if not resolved or not Path(resolved).is_dir():
        print(
            "Error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(resolved)
