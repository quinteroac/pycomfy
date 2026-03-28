"""Tests for US-004 — SHA256 integrity verification in comfy_diffusion.downloader.

All tests run without network access; urllib.request.urlopen and
huggingface_hub are patched where needed.
"""

from __future__ import annotations

import hashlib
import io
import json as _json
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
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_BODY = b"fakebody"
CORRECT_SHA256 = hashlib.sha256(FAKE_BODY).hexdigest()
WRONG_SHA256 = "a" * 64


def _make_cm(data: bytes) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.__enter__ = lambda s: s  # type: ignore[assignment]
    buf.__exit__ = lambda s, *a: None  # type: ignore[assignment]
    return buf


def _fake_urlopen(url_or_req: object, *args: object, **kwargs: object) -> io.BytesIO:
    return _make_cm(FAKE_BODY)


# ---------------------------------------------------------------------------
# URLModelEntry SHA-256 tests
# ---------------------------------------------------------------------------


class TestURLEntrySHA256:
    def test_ac01_correct_hash_after_download(self, tmp_path: Path) -> None:
        """AC01: sha256 is verified after a fresh download and passes when correct."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=CORRECT_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])  # must not raise
        assert (tmp_path / "model.safetensors").exists()

    def test_ac03_fresh_download_wrong_hash_raises_value_error(self, tmp_path: Path) -> None:
        """AC03: wrong hash on freshly downloaded file → ValueError."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=WRONG_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                download_models([entry])

    def test_ac03_fresh_download_error_mentions_filename(self, tmp_path: Path) -> None:
        """AC03: ValueError message contains the filename."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=WRONG_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(ValueError) as exc_info:
                download_models([entry])
        assert "model.safetensors" in str(exc_info.value)

    def test_ac03_file_deleted_after_hash_failure(self, tmp_path: Path) -> None:
        """AC03: dest file is removed when fresh-download hash check fails."""
        dest = tmp_path / "model.safetensors"
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=WRONG_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(ValueError):
                download_models([entry])
        assert not dest.exists()

    def test_ac02_existing_file_correct_hash_skips_download(self, tmp_path: Path) -> None:
        """AC02: already-present file with correct hash is not re-downloaded."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(FAKE_BODY)

        urlopen_mock = MagicMock()
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch("urllib.request.urlopen", urlopen_mock):
            download_models([entry])

        urlopen_mock.assert_not_called()
        assert dest.exists()

    def test_ac04_existing_file_wrong_hash_deleted_and_redownloaded(
        self, tmp_path: Path
    ) -> None:
        """AC04: already-present file with wrong hash is deleted then re-downloaded."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"stale_data")  # wrong hash

        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])

        assert dest.exists()
        assert dest.read_bytes() == FAKE_BODY

    def test_ac04_existing_file_wrong_hash_redownload_still_wrong_raises(
        self, tmp_path: Path
    ) -> None:
        """AC04 + AC03: re-downloaded file also has wrong hash → ValueError."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"stale_data")

        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=WRONG_SHA256,  # nothing will ever match this
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                download_models([entry])

    def test_ac05_no_sha256_skips_verification(self, tmp_path: Path) -> None:
        """AC05: sha256=None → no verification, download succeeds silently."""
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=None,
        )
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            download_models([entry])  # must not raise
        assert (tmp_path / "model.safetensors").exists()

    def test_ac05_existing_no_sha256_skips_verification(self, tmp_path: Path) -> None:
        """AC05: existing file with sha256=None → skips check, returns without download."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"anything")

        urlopen_mock = MagicMock()
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=None,
        )
        with patch("urllib.request.urlopen", urlopen_mock):
            download_models([entry])

        urlopen_mock.assert_not_called()


# ---------------------------------------------------------------------------
# HFModelEntry SHA-256 tests
# ---------------------------------------------------------------------------


class TestHFEntrySHA256:
    def _make_hf_mock(self, tmp_path: Path, body: bytes = FAKE_BODY) -> MagicMock:
        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(body)
        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)
        return hf_mock

    def test_ac01_correct_hash_after_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC01: correct hash after HF download passes without error."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        hf_mock = self._make_hf_mock(tmp_path)
        dest = tmp_path / "out" / "model.safetensors"
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])
        assert dest.exists()

    def test_ac03_fresh_download_wrong_hash_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC03: wrong hash on freshly downloaded HF file → ValueError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        hf_mock = self._make_hf_mock(tmp_path)
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "out" / "model.safetensors",
            sha256=WRONG_SHA256,
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                download_models([entry])

    def test_ac02_existing_file_correct_hash_skips_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC02: already-present HF file with correct hash is not re-downloaded."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(FAKE_BODY)

        hf_mock = MagicMock()
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        hf_mock.hf_hub_download.assert_not_called()

    def test_ac04_existing_file_wrong_hash_triggers_redownload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC04: already-present HF file with wrong hash is deleted and re-downloaded."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"stale_data")

        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(FAKE_BODY)
        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)

        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        hf_mock.hf_hub_download.assert_called_once()
        assert dest.read_bytes() == FAKE_BODY

    def test_ac05_no_sha256_skips_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC05: sha256=None → no verification for HF entry."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        hf_mock = self._make_hf_mock(tmp_path)
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=tmp_path / "out" / "model.safetensors",
            sha256=None,
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])  # must not raise


# ---------------------------------------------------------------------------
# CivitAIModelEntry SHA-256 tests
# ---------------------------------------------------------------------------


def _json_bytes(obj: object) -> bytes:
    return _json.dumps(obj).encode()


def _make_civitai_urlopen(body: bytes = FAKE_BODY):  # type: ignore[return]
    def fake_urlopen(req: object, *args: object, **kwargs: object) -> io.BytesIO:
        url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
        if "/api/v1/" in url:
            return _make_cm(_json_bytes({"files": [{"name": "model.safetensors"}]}))
        return _make_cm(body)

    return fake_urlopen


class TestCivitAIEntrySHA256:
    def test_ac01_correct_hash_after_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC01: correct hash after CivitAI download passes without error."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"
        entry = CivitAIModelEntry(
            model_id=1, dest=dest, version_id=42, sha256=CORRECT_SHA256
        )
        with patch("urllib.request.urlopen", side_effect=_make_civitai_urlopen(FAKE_BODY)):
            download_models([entry])
        assert dest.exists()

    def test_ac03_fresh_download_wrong_hash_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC03: wrong hash on freshly downloaded CivitAI file → ValueError."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"
        entry = CivitAIModelEntry(
            model_id=1, dest=dest, version_id=42, sha256=WRONG_SHA256
        )
        with patch("urllib.request.urlopen", side_effect=_make_civitai_urlopen(FAKE_BODY)):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                download_models([entry])

    def test_ac02_existing_file_correct_hash_skips_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC02: already-present CivitAI file with correct hash is not re-downloaded."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(FAKE_BODY)

        entry = CivitAIModelEntry(
            model_id=1, dest=dest, version_id=42, sha256=CORRECT_SHA256
        )
        urlopen_mock = MagicMock()
        with patch("urllib.request.urlopen", urlopen_mock):
            download_models([entry])

        urlopen_mock.assert_not_called()

    def test_ac04_existing_file_wrong_hash_triggers_redownload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC04: already-present CivitAI file with wrong hash is deleted and re-downloaded."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"stale_data")

        entry = CivitAIModelEntry(
            model_id=1, dest=dest, version_id=42, sha256=CORRECT_SHA256
        )
        with patch("urllib.request.urlopen", side_effect=_make_civitai_urlopen(FAKE_BODY)):
            download_models([entry])

        assert dest.read_bytes() == FAKE_BODY

    def test_ac05_no_sha256_skips_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC05: sha256=None → no verification for CivitAI entry."""
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"
        entry = CivitAIModelEntry(model_id=1, dest=dest, version_id=42, sha256=None)
        with patch("urllib.request.urlopen", side_effect=_make_civitai_urlopen(FAKE_BODY)):
            download_models([entry])  # must not raise
        assert dest.exists()
