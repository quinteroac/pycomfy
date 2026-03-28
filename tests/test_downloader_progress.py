"""Tests for US-003 — Progress Reporting in comfy_diffusion.downloader.

All tests run without network access; urllib.request.urlopen and
huggingface_hub are patched where needed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comfy_diffusion.downloader import (
    CivitAIModelEntry,
    HFModelEntry,
    URLModelEntry,
    download_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cm(data: bytes) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.__enter__ = lambda s: s  # type: ignore[assignment]
    buf.__exit__ = lambda s, *a: None  # type: ignore[assignment]
    return buf


def _json_bytes(obj: object) -> bytes:
    return json.dumps(obj).encode()


def _make_tqdm_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (tqdm_module, tqdm_class, tqdm_instance) mocks."""
    tqdm_instance = MagicMock()
    tqdm_instance.__enter__ = MagicMock(return_value=tqdm_instance)
    tqdm_instance.__exit__ = MagicMock(return_value=False)
    tqdm_class = MagicMock(return_value=tqdm_instance)
    tqdm_module = MagicMock()
    tqdm_module.tqdm = tqdm_class
    return tqdm_module, tqdm_class, tqdm_instance


# ---------------------------------------------------------------------------
# AC01 — HFModelEntry: progress handled natively by huggingface_hub
# ---------------------------------------------------------------------------


class TestHFProgress:
    def test_disable_progress_bars_called_when_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quiet=True causes disable_progress_bars() then enable_progress_bars()."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(b"weights")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached)

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry], quiet=True)

        hf_mock.disable_progress_bars.assert_called_once()
        hf_mock.enable_progress_bars.assert_called_once()

    def test_progress_bars_not_disabled_when_not_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quiet=False leaves HF progress bars as-is."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(b"weights")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached)

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry], quiet=False)

        hf_mock.disable_progress_bars.assert_not_called()

    def test_enable_progress_bars_called_even_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enable_progress_bars() is always called in the finally block."""
        monkeypatch.setenv("HF_TOKEN", "tok")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.side_effect = Exception("network timeout")

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            with pytest.raises(RuntimeError):
                download_models([entry], quiet=True)

        hf_mock.enable_progress_bars.assert_called_once()


# ---------------------------------------------------------------------------
# AC02 — URLModelEntry: tqdm progress bar when tqdm is installed
# ---------------------------------------------------------------------------


class TestURLProgress:
    def test_tqdm_used_when_available(self, tmp_path: Path) -> None:
        """tqdm is used when installed and quiet=False."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            resp = _make_cm(b"weights")
            resp.headers = {"Content-Length": "7"}  # type: ignore[attr-defined]
            return resp

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry])

        tqdm_class.assert_called_once()

    def test_tqdm_desc_is_dest_filename(self, tmp_path: Path) -> None:
        """Progress bar description equals the destination filename."""
        entry = URLModelEntry(
            url="https://example.com/some_model.safetensors",
            dest=tmp_path / "some_model.safetensors",
        )
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            resp = _make_cm(b"data")
            resp.headers = {}  # type: ignore[attr-defined]
            return resp

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry])

        _, kwargs = tqdm_class.call_args
        assert kwargs.get("desc") == "some_model.safetensors"

    def test_tqdm_not_used_when_quiet(self, tmp_path: Path) -> None:
        """quiet=True skips tqdm even when tqdm is installed."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        with (
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(b"w")),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry], quiet=True)

        tqdm_class.assert_not_called()

    def test_fallback_to_copyfileobj_when_tqdm_missing(self, tmp_path: Path) -> None:
        """Download still completes when tqdm is not installed."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with (
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(b"weights")),
            patch.dict("sys.modules", {"tqdm": None}),
        ):
            download_models([entry])

        assert (tmp_path / "model.safetensors").read_bytes() == b"weights"

    def test_content_length_forwarded_to_tqdm(self, tmp_path: Path) -> None:
        """Content-Length header is passed as total to tqdm."""
        entry = URLModelEntry(
            url="https://example.com/big.safetensors",
            dest=tmp_path / "big.safetensors",
        )
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            resp = _make_cm(b"x" * 1024)
            resp.headers = {"Content-Length": "1024"}  # type: ignore[attr-defined]
            return resp

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry])

        _, kwargs = tqdm_class.call_args
        assert kwargs.get("total") == 1024


# ---------------------------------------------------------------------------
# AC03 — CivitAIModelEntry: progress via available mechanism
# ---------------------------------------------------------------------------


class TestCivitAIProgress:
    def test_progress_shown_when_not_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CivitAI download shows tqdm progress when quiet=False."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "lora.safetensors"
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "model-versions" in url:
                return _make_cm(_json_bytes({"files": [{"name": "lora.safetensors"}]}))
            resp = _make_cm(b"weights")
            resp.headers = {}  # type: ignore[attr-defined]
            return resp

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([CivitAIModelEntry(model_id=1, dest=dest, version_id=42)])

        tqdm_class.assert_called_once()

    def test_progress_suppressed_when_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quiet=True suppresses CivitAI progress."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "lora.safetensors"
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "model-versions" in url:
                return _make_cm(_json_bytes({"files": [{"name": "lora.safetensors"}]}))
            return _make_cm(b"weights")

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models(
                [CivitAIModelEntry(model_id=1, dest=dest, version_id=42)], quiet=True
            )

        tqdm_class.assert_not_called()


# ---------------------------------------------------------------------------
# AC04 — quiet=True suppresses all progress across all entry types
# ---------------------------------------------------------------------------


class TestQuietSuppressesAll:
    def test_hf_quiet_disables_bars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quiet=True calls disable_progress_bars on huggingface_hub."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cached = tmp_path / "c.safetensors"
        cached.write_bytes(b"w")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached)

        entry = HFModelEntry(
            repo_id="a/b",
            filename="x.safetensors",
            dest=tmp_path / "x.safetensors",
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry], quiet=True)

        hf_mock.disable_progress_bars.assert_called_once()

    def test_url_quiet_skips_tqdm(self, tmp_path: Path) -> None:
        """quiet=True does not invoke tqdm for URL downloads."""
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest=tmp_path / "m.safetensors",
        )

        with (
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(b"w")),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry], quiet=True)

        tqdm_class.assert_not_called()

    def test_civitai_quiet_skips_tqdm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quiet=True does not invoke tqdm for CivitAI downloads."""
        monkeypatch.setenv("CIVITAI_API_KEY", "k")
        dest = tmp_path / "m.safetensors"
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "model-versions" in url:
                return _make_cm(_json_bytes({"files": [{"name": "m.safetensors"}]}))
            return _make_cm(b"w")

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models(
                [CivitAIModelEntry(model_id=5, dest=dest, version_id=1)], quiet=True
            )

        tqdm_class.assert_not_called()
