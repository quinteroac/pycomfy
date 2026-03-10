"""Tests for video I/O helpers."""

from __future__ import annotations

import builtins
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import comfy_diffusion
import comfy_diffusion.video as video_module
from comfy_diffusion.video import get_video_components, load_video, save_video


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"

    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=_repo_root(),
    )


def test_video_module_exports_expected_entrypoints() -> None:
    assert video_module.__all__ == ["load_video", "save_video", "get_video_components"]


def test_load_video_signature_matches_contract() -> None:
    signature = inspect.signature(load_video)
    assert str(signature) == "(path: 'str | Path') -> 'Any'"


def test_save_video_signature_matches_contract() -> None:
    signature = inspect.signature(save_video)
    assert str(signature) == "(frames: 'Any', path: 'str | Path', fps: 'float') -> 'None'"


def test_get_video_components_signature_matches_contract() -> None:
    signature = inspect.signature(get_video_components)
    assert str(signature) == "(video_path: 'str | Path') -> 'dict[str, int | float]'"


def test_video_helpers_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_video")
    assert not hasattr(comfy_diffusion, "save_video")
    assert not hasattr(comfy_diffusion, "get_video_components")


def test_load_video_reads_frames_with_backend_and_returns_bhwc_tensor_or_pil(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"not-a-real-video")

    expected_frames = [object()]
    expected_output = object()
    calls: dict[str, Any] = {}

    def fake_read_frames(_cv2: Any, path: Path) -> list[Any]:
        calls["path"] = path
        return expected_frames

    def fake_frames_to_output(frames: list[Any]) -> Any:
        calls["frames"] = frames
        return expected_output

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", object()))
    monkeypatch.setattr(video_module, "_read_frames_cv2", fake_read_frames)
    monkeypatch.setattr(video_module, "_frames_to_output", fake_frames_to_output)

    result = load_video(video_path)

    assert result is expected_output
    assert calls["path"] == video_path
    assert calls["frames"] == expected_frames


def test_save_video_writes_frames_with_cv2_backend(monkeypatch: Any, tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    frame_a = SimpleNamespace(shape=(8, 12, 3), name="a")
    frame_b = SimpleNamespace(shape=(8, 12, 3), name="b")
    written: list[Any] = []

    class FakeWriter:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: Any) -> None:
            written.append(frame)

        def release(self) -> None:
            written.append("released")

    class FakeCv2:
        COLOR_RGB2BGR = "COLOR_RGB2BGR"

        def __init__(self) -> None:
            self.writer_calls: list[tuple[str, Any, float, tuple[int, int]]] = []

        def VideoWriter_fourcc(self, *_args: Any) -> str:
            return "mp4v"

        def VideoWriter(
            self, path: str, fourcc: Any, fps: float, size: tuple[int, int]
        ) -> FakeWriter:
            self.writer_calls.append((path, fourcc, fps, size))
            return FakeWriter()

        def cvtColor(self, frame: Any, conversion: Any) -> tuple[Any, Any, str]:
            return (frame, conversion, "converted")

    fake_cv2 = FakeCv2()

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", fake_cv2))
    monkeypatch.setattr(
        video_module,
        "_coerce_frames_to_rgb_uint8",
        lambda _frames: [frame_a, frame_b],
    )

    save_video(frames=object(), path=output_path, fps=24.0)

    assert fake_cv2.writer_calls == [(str(output_path), "mp4v", 24.0, (12, 8))]
    assert written == [
        (frame_a, "COLOR_RGB2BGR", "converted"),
        (frame_b, "COLOR_RGB2BGR", "converted"),
        "released",
    ]


def test_get_video_components_returns_frame_count_fps_width_height(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "meta.mp4"

    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, prop: float) -> float:
            return {
                1.0: 300.0,
                2.0: 30.0,
                3.0: 640.0,
                4.0: 360.0,
            }[float(prop)]

        def release(self) -> None:
            self.released = True

    fake_capture = FakeCapture()

    class FakeCv2:
        CAP_PROP_FRAME_COUNT = 1.0
        CAP_PROP_FPS = 2.0
        CAP_PROP_FRAME_WIDTH = 3.0
        CAP_PROP_FRAME_HEIGHT = 4.0

        def VideoCapture(self, path: str) -> FakeCapture:
            assert path == str(video_path)
            return fake_capture

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", FakeCv2()))

    metadata = get_video_components(video_path)

    assert metadata == {"frame_count": 300, "fps": 30.0, "width": 640, "height": 360}
    assert fake_capture.released is True


def test_get_video_backend_raises_clear_error_when_video_extra_is_missing(
    monkeypatch: Any,
) -> None:
    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2" or name == "imageio" or name.startswith("imageio."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    with pytest.raises(ModuleNotFoundError, match=r"comfy-diffusion\[video\]"):
        video_module._get_video_backend()


def test_import_comfy_diffusion_video_has_no_cv2_or_imageio_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.video import load_video, save_video, get_video_components\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'load_video': load_video.__name__,\n"
        "  'save_video': save_video.__name__,\n"
        "  'get_video_components': get_video_components.__name__,\n"
        "  'cv2_loaded': 'cv2' in sys.modules,\n"
        "  'imageio_loaded': any(m.startswith('imageio') for m in sys.modules),\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["load_video"] == "load_video"
    assert payload["save_video"] == "save_video"
    assert payload["get_video_components"] == "get_video_components"
    assert payload["cv2_loaded"] is False
    assert payload["imageio_loaded"] is False
    heavy = [m for m in payload["new_modules"] if m == "cv2" or m.startswith("imageio")]
    assert heavy == []
