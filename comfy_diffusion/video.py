"""Video loading and saving helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _get_video_backend() -> tuple[str, Any]:
    """Return available video backend module (cv2 first, then imageio)."""
    try:
        import cv2

        return "cv2", cv2
    except ModuleNotFoundError:
        pass

    try:
        import imageio.v2 as imageio

        return "imageio", imageio
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Video I/O requires optional dependencies. Install with "
            "`comfy-diffusion[video]` to enable `load_video`, `save_video`, "
            "and `get_video_components`."
        ) from error


def _get_torch_module() -> Any | None:
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _normalize_to_rgb_uint8(frame: Any) -> Any:
    """Normalize a single frame to HWC uint8 RGB."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError("each frame must be HWC, HW, or PIL image")

    channels = array.shape[2]
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels != 3:
        raise ValueError("frame channel count must be 1, 3, or 4")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max(initial=0.0))
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _coerce_frames_to_rgb_uint8(frames: Any) -> list[Any]:
    """Coerce supported frame inputs into a non-empty list of RGB uint8 arrays."""
    numpy_module: Any | None = None
    try:
        import numpy as np

        numpy_module = np
    except ModuleNotFoundError:
        pass

    if hasattr(frames, "detach") and hasattr(frames, "cpu"):
        tensor = frames.detach().cpu()
        if getattr(tensor, "ndim", None) == 3:
            tensor = tensor.unsqueeze(0)
        if getattr(tensor, "ndim", None) != 4:
            raise ValueError("tensor frames must be BHWC or HWC")

        array = tensor.numpy()
        return [_normalize_to_rgb_uint8(frame) for frame in array]

    if numpy_module is not None and isinstance(frames, numpy_module.ndarray):
        if frames.ndim == 3:
            return [_normalize_to_rgb_uint8(frames)]
        if frames.ndim == 4:
            return [_normalize_to_rgb_uint8(frame) for frame in frames]
        raise ValueError("numpy frames must be BHWC or HWC")

    if not isinstance(frames, Iterable) or isinstance(frames, (str, bytes)):
        raise TypeError("frames must be a BHWC tensor, numpy array, or iterable of frames")

    normalized_frames = [_normalize_to_rgb_uint8(frame) for frame in frames]
    if not normalized_frames:
        raise ValueError("frames is empty")
    return normalized_frames


def _read_frames_cv2(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video file: {path}")

    frames: list[Any] = []
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def _read_frames_imageio(imageio: Any, path: Path) -> list[Any]:
    reader = imageio.get_reader(str(path))
    try:
        frames = [_normalize_to_rgb_uint8(frame) for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def _frames_to_output(frames: list[Any]) -> Any:
    """Return BHWC float tensor when torch is available; otherwise PIL image list."""
    torch = _get_torch_module()
    if torch is not None:
        import numpy as np

        stacked = np.stack(frames)
        return torch.from_numpy(stacked).to(torch.float32) / 255.0

    from PIL import Image

    return [Image.fromarray(frame, mode="RGB") for frame in frames]


def load_video(path: str | Path) -> Any:
    """Load a video file into a BHWC tensor, or PIL frames when torch is unavailable."""
    video_path = Path(path)
    backend_name, backend = _get_video_backend()
    if backend_name == "cv2":
        frames = _read_frames_cv2(backend, video_path)
    else:
        frames = _read_frames_imageio(backend, video_path)
    return _frames_to_output(frames)


def save_video(frames: Any, path: str | Path, fps: float) -> None:
    """Save frames to a video file."""
    output_path = Path(path)
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    backend_name, backend = _get_video_backend()
    normalized_frames = _coerce_frames_to_rgb_uint8(frames)

    if backend_name == "cv2":
        _EXTENSION_TO_FOURCC = {".webm": "VP80", ".mp4": "mp4v"}
        fourcc_str = _EXTENSION_TO_FOURCC.get(output_path.suffix.lower(), "mp4v")
        height, width = normalized_frames[0].shape[:2]
        writer = backend.VideoWriter(
            str(output_path),
            backend.VideoWriter_fourcc(*fourcc_str),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"unable to open video writer for path: {output_path}")

        try:
            for frame in normalized_frames:
                if frame.shape[:2] != (height, width):
                    raise ValueError("all frames must have identical width and height")
                writer.write(backend.cvtColor(frame, backend.COLOR_RGB2BGR))
        finally:
            writer.release()
        return

    writer = backend.get_writer(str(output_path), fps=float(fps))
    try:
        for frame in normalized_frames:
            writer.append_data(frame)
    finally:
        writer.close()


def get_video_components(video_path: str | Path) -> dict[str, int | float]:
    """Return frame count, fps, width, and height for a video."""
    path = Path(video_path)
    backend_name, backend = _get_video_backend()

    if backend_name == "cv2":
        capture = backend.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"unable to open video file: {path}")

        try:
            frame_count = int(capture.get(backend.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(backend.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(backend.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(backend.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        }

    reader = backend.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0.0))

        size = metadata.get("size")
        if size is None:
            frame0 = _normalize_to_rgb_uint8(reader.get_data(0))
            height, width = frame0.shape[:2]
        else:
            width, height = int(size[0]), int(size[1])

        frame_count = int(metadata.get("nframes", 0) or 0)
        if frame_count <= 0:
            frame_count = reader.count_frames()
    finally:
        reader.close()

    return {
        "frame_count": int(frame_count),
        "fps": fps,
        "width": int(width),
        "height": int(height),
    }


def ltxv_img_to_video_inplace(
    vae: Any,
    image: Any,
    latent: dict[str, Any],
    strength: float = 1.0,
    bypass: bool = False,
) -> dict[str, Any]:
    """Inject an image frame into a latent for LTX2/LTX3 image-to-video pipelines.

    Mirrors ``LTXVImgToVideoInplace.execute()`` from ComfyUI.
    """
    if bypass:
        return latent

    import comfy.utils  # type: ignore[import-untyped]
    import torch

    samples = latent["samples"]
    _, height_scale_factor, width_scale_factor = vae.downscale_index_formula

    batch, _, latent_frames, latent_height, latent_width = samples.shape
    width = latent_width * width_scale_factor
    height = latent_height * height_scale_factor

    if image.shape[1] != height or image.shape[2] != width:
        pixels = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    else:
        pixels = image

    encode_pixels = pixels[:, :, :, :3]
    t = vae.encode(encode_pixels)

    samples[:, :, : t.shape[2]] = t

    conditioning_latent_frames_mask = torch.ones(
        (batch, 1, latent_frames, 1, 1),
        dtype=torch.float32,
        device=samples.device,
    )
    conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength

    return {"samples": samples, "noise_mask": conditioning_latent_frames_mask}


__all__ = [
    "load_video",
    "save_video",
    "get_video_components",
    "ltxv_img_to_video_inplace",
]
