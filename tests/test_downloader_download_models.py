"""Tests for US-002 — download_models() in comfy_diffusion.downloader.

All tests run without network access; urllib.request.urlopen and
huggingface_hub are patched where needed.
"""

from __future__ import annotations

import io
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comfy_diffusion.downloader import (
    CivitAIModelEntry,
    HFModelEntry,
    ModelEntry,
    URLModelEntry,
    download_models,
)


# ---------------------------------------------------------------------------
# AC-09 — import
# ---------------------------------------------------------------------------


class TestImport:
    def test_download_models_importable(self) -> None:
        from comfy_diffusion.downloader import download_models as dm  # noqa: F401

        assert callable(dm)


# ---------------------------------------------------------------------------
# AC-01 — signature
# ---------------------------------------------------------------------------


class TestSignature:
    def test_accepts_empty_manifest(self, tmp_path: Path) -> None:
        """Empty manifest completes without error."""
        download_models([], models_dir=tmp_path)

    def test_accepts_quiet_flag(self, tmp_path: Path) -> None:
        download_models([], models_dir=tmp_path, quiet=True)

    def test_models_dir_defaults_to_none(self) -> None:
        """models_dir=None is valid when manifest is empty."""
        download_models([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_urlopen(url_or_req: object, *args: object, **kwargs: object) -> io.BytesIO:
    """Fake urlopen that returns 8 bytes of data."""
    buf = io.BytesIO(b"fakebody")
    buf.read = buf.read  # type: ignore[assignment]
    # Make it usable as a context manager
    buf.__enter__ = lambda s: s  # type: ignore[assignment]
    buf.__exit__ = lambda s, *a: None  # type: ignore[assignment]
    return buf


# ---------------------------------------------------------------------------
# URLModelEntry tests (AC-02, AC-06, AC-07, AC-08)
# ---------------------------------------------------------------------------


class TestURLEntry:
    def test_downloads_file_to_dest(self, tmp_path: Path) -> None:
        """AC-02: URLModelEntry dispatched and file written to dest."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])
        assert (tmp_path / "model.safetensors").exists()

    def test_resolves_relative_dest_against_models_dir(self, tmp_path: Path) -> None:
        """AC-02 + AC-07: relative dest resolved against models_dir."""
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest="checkpoints/m.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry], models_dir=tmp_path)
        assert (tmp_path / "checkpoints" / "m.safetensors").exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """AC-07: dest directories are created with parents=True."""
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest=tmp_path / "a" / "b" / "c" / "m.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])
        assert (tmp_path / "a" / "b" / "c" / "m.safetensors").exists()

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        """AC-06: files already present are not re-downloaded."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"original")

        urlopen_mock = MagicMock()
        entry = URLModelEntry(url="https://example.com/model.safetensors", dest=dest)
        with patch("urllib.request.urlopen", urlopen_mock):
            download_models([entry])

        urlopen_mock.assert_not_called()
        assert dest.read_bytes() == b"original"

    def test_dest_no_extension_appends_url_basename(self, tmp_path: Path) -> None:
        """dest without extension → URL basename is appended."""
        entry = URLModelEntry(
            url="https://example.com/flux.safetensors",
            dest=tmp_path / "checkpoints",
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])
        assert (tmp_path / "checkpoints" / "flux.safetensors").exists()

    def test_download_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        """AC-08: download failure raises RuntimeError with source identifier."""
        entry = URLModelEntry(
            url="https://example.com/missing.safetensors",
            dest=tmp_path / "missing.safetensors",
        )

        def fail_urlopen(*args: object, **kwargs: object) -> None:
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=fail_urlopen):
            with pytest.raises(RuntimeError, match="https://example.com/missing.safetensors"):
                download_models([entry])

    def test_relative_dest_without_models_dir_raises(self) -> None:
        """RuntimeError when dest is relative and models_dir is None."""
        entry = URLModelEntry(url="https://example.com/m.safetensors", dest="checkpoints/m.safetensors")
        with pytest.raises(RuntimeError, match="models_dir must be provided"):
            download_models([entry])


# ---------------------------------------------------------------------------
# HFModelEntry tests (AC-02, AC-03, AC-06, AC-07, AC-08)
# ---------------------------------------------------------------------------


class TestHFEntry:
    def test_downloads_hf_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-02: HFModelEntry dispatched to huggingface_hub."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(b"weights")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "checkpoints" / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        assert (tmp_path / "checkpoints" / "model.safetensors").exists()
        hf_mock.hf_hub_download.assert_called_once_with(
            repo_id="org/model",
            filename="model.safetensors",
            token=None,
        )

    def test_passes_hf_token_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF_TOKEN env var is forwarded to hf_hub_download as token."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret")

        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(b"weights")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)

        entry = HFModelEntry(
            repo_id="org/gated",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        _, kwargs = hf_mock.hf_hub_download.call_args
        assert kwargs["token"] == "hf_secret"

    def test_gated_model_without_token_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-03: gated model + no HF_TOKEN → RuntimeError with instructions."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.side_effect = Exception("401 Unauthorized gated repo")

        entry = HFModelEntry(
            repo_id="org/gated-model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                download_models([entry])

    def test_hf_download_failure_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-08: non-auth download failure raises RuntimeError with source."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.side_effect = Exception("network timeout")

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            with pytest.raises(RuntimeError, match="org/model"):
                download_models([entry])

    def test_skips_existing_hf_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-06: existing file is not re-downloaded."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"original")

        hf_mock = MagicMock()
        entry = HFModelEntry(repo_id="org/model", filename="model.safetensors", dest=dest)

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        hf_mock.hf_hub_download.assert_not_called()

    def test_dest_no_extension_appends_filename_basename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dest without extension → basename of entry.filename is appended."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(b"weights")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)

        entry = HFModelEntry(
            repo_id="org/model",
            filename="subfolder/flux.safetensors",
            dest=tmp_path / "checkpoints",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        assert (tmp_path / "checkpoints" / "flux.safetensors").exists()

    def test_token_not_in_error_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-05: token value is not exposed in error messages."""
        monkeypatch.setenv("HF_TOKEN", "super_secret_token_abc123")

        hf_mock = MagicMock()
        hf_mock.hf_hub_download.side_effect = Exception("download failed")

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            with pytest.raises(RuntimeError) as exc_info:
                download_models([entry])

        assert "super_secret_token_abc123" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# CivitAIModelEntry tests (AC-02, AC-04, AC-05, AC-06, AC-07, AC-08)
# ---------------------------------------------------------------------------


class TestCivitAIEntry:
    def test_raises_without_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-04: missing CIVITAI_API_KEY raises RuntimeError with instructions."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)

        entry = CivitAIModelEntry(model_id=12345, dest=tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="CIVITAI_API_KEY"):
            download_models([entry])

    def test_raises_without_api_key_mentions_set_instructions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-04: error message contains actionable instructions."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)

        entry = CivitAIModelEntry(model_id=99, dest=tmp_path / "m.safetensors")
        with pytest.raises(RuntimeError) as exc_info:
            download_models([entry])
        msg = str(exc_info.value)
        assert "CIVITAI_API_KEY" in msg
        assert "99" in msg  # model_id in message

    def test_skips_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-06: existing file skipped — no API or download call made."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")

        dest = tmp_path / "lora.safetensors"
        dest.write_bytes(b"original")

        entry = CivitAIModelEntry(model_id=1, dest=dest, version_id=42)

        with patch("urllib.request.urlopen") as urlopen_mock:
            download_models([entry])

        urlopen_mock.assert_not_called()
        assert dest.read_bytes() == b"original"

    def test_downloads_with_version_id_and_explicit_dest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-02: CivitAIModelEntry with version_id + full dest path downloads file."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")

        dest = tmp_path / "lora.safetensors"

        def fake_urlopen(req: object, *args: object, **kwargs: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "model-versions" in url:
                data = json_bytes({"files": [{"name": "lora.safetensors"}]})
                return _make_cm(data)
            # download call
            return _make_cm(b"weights_data")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            download_models([CivitAIModelEntry(model_id=1, dest=dest, version_id=42)])

        assert dest.exists()

    def test_token_not_in_file_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-05: API key not embedded in destination path."""
        secret = "my_secret_civitai_key"
        monkeypatch.setenv("CIVITAI_API_KEY", secret)

        dest = tmp_path / "model.safetensors"

        def fake_urlopen(req: object, *args: object, **kwargs: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "model-versions" in url:
                return _make_cm(json_bytes({"files": [{"name": "model.safetensors"}]}))
            return _make_cm(b"data")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            download_models([CivitAIModelEntry(model_id=1, dest=dest, version_id=5)])

        assert dest.exists()
        # token must not appear anywhere in the saved path string
        assert secret not in str(dest)

    def test_download_failure_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-08: download failure raises RuntimeError with source identifier."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")

        dest = tmp_path / "model.safetensors"

        call_count = 0

        def fail_on_download(req: object, *args: object, **kwargs: object) -> io.BytesIO:
            nonlocal call_count
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            call_count += 1
            if "model-versions" in url:
                return _make_cm(json_bytes({"files": [{"name": "model.safetensors"}]}))
            raise urllib.error.URLError("connection reset")

        with patch("urllib.request.urlopen", side_effect=fail_on_download):
            with pytest.raises(RuntimeError, match="12345"):
                download_models([CivitAIModelEntry(model_id=12345, dest=dest, version_id=7)])


# ---------------------------------------------------------------------------
# AC-02 — dispatch by type (mixed manifest)
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_mixed_manifest_dispatches_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-02: each entry type is dispatched to the correct handler."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")

        # Pre-create all dest files so no actual downloads are attempted
        hf_dest = tmp_path / "hf_model.safetensors"
        url_dest = tmp_path / "url_model.safetensors"
        civ_dest = tmp_path / "civ_model.safetensors"
        for p in (hf_dest, url_dest, civ_dest):
            p.write_bytes(b"existing")

        manifest: list[ModelEntry] = [
            HFModelEntry(repo_id="org/a", filename="a.safetensors", dest=hf_dest),
            URLModelEntry(url="https://example.com/b.safetensors", dest=url_dest),
            CivitAIModelEntry(model_id=1, dest=civ_dest, version_id=1),
        ]

        urlopen_mock = MagicMock()
        hf_mock = MagicMock()

        with (
            patch("urllib.request.urlopen", urlopen_mock),
            patch.dict("sys.modules", {"huggingface_hub": hf_mock}),
        ):
            download_models(manifest, models_dir=tmp_path)

        # All files already exist — no network calls should occur
        urlopen_mock.assert_not_called()
        hf_mock.hf_hub_download.assert_not_called()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402


def json_bytes(obj: object) -> bytes:
    return _json.dumps(obj).encode()


def _make_cm(data: bytes) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.__enter__ = lambda s: s  # type: ignore[assignment]
    buf.__exit__ = lambda s, *a: None  # type: ignore[assignment]
    return buf
