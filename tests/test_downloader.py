"""Comprehensive tests for comfy_diffusion.downloader.

Covers entry types, download dispatch, idempotency, SHA256 integrity
verification, progress/quiet flag, and error handling.

All tests run without network access — urllib.request.urlopen and
huggingface_hub are patched where needed.
"""

from __future__ import annotations

import hashlib
import io
import json
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
# Helpers
# ---------------------------------------------------------------------------

FAKE_BODY = b"fakebody_for_testing"
CORRECT_SHA256 = hashlib.sha256(FAKE_BODY).hexdigest()
WRONG_SHA256 = "0" * 64


def _make_cm(data: bytes) -> io.BytesIO:
    """Wrap bytes in a context-manager–compatible BytesIO."""
    buf = io.BytesIO(data)
    buf.__enter__ = lambda s: s  # type: ignore[assignment]
    buf.__exit__ = lambda s, *a: None  # type: ignore[assignment]
    return buf


def _json_bytes(obj: object) -> bytes:
    return json.dumps(obj).encode()


def _civitai_urlopen(body: bytes = FAKE_BODY):  # type: ignore[return]
    """Fake urlopen that returns CivitAI API metadata then download body."""

    def fake(req: object, *args: object, **kwargs: object) -> io.BytesIO:
        url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
        if "/api/v1/" in url:
            return _make_cm(_json_bytes({"files": [{"name": "model.safetensors"}]}))
        return _make_cm(body)

    return fake


def _make_tqdm_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    tqdm_instance = MagicMock()
    tqdm_instance.__enter__ = MagicMock(return_value=tqdm_instance)
    tqdm_instance.__exit__ = MagicMock(return_value=False)
    tqdm_class = MagicMock(return_value=tqdm_instance)
    tqdm_module = MagicMock()
    tqdm_module.tqdm = tqdm_class
    return tqdm_module, tqdm_class, tqdm_instance


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------


class TestHFModelEntry:
    def test_required_fields(self) -> None:
        entry = HFModelEntry(
            repo_id="black-forest-labs/FLUX.1-schnell",
            filename="flux1-schnell.safetensors",
            dest="checkpoints",
        )
        assert entry.repo_id == "black-forest-labs/FLUX.1-schnell"
        assert entry.filename == "flux1-schnell.safetensors"
        assert entry.dest == "checkpoints"
        assert entry.sha256 is None

    def test_sha256_can_be_set(self) -> None:
        entry = HFModelEntry(
            repo_id="org/m", filename="m.safetensors", dest="cp", sha256="a" * 64
        )
        assert entry.sha256 == "a" * 64

    def test_dest_accepts_path_object(self) -> None:
        entry = HFModelEntry(
            repo_id="org/m", filename="m.safetensors", dest=Path("checkpoints")
        )
        assert isinstance(entry.dest, Path)


class TestCivitAIModelEntry:
    def test_required_fields(self) -> None:
        entry = CivitAIModelEntry(model_id=12345, dest="loras")
        assert entry.model_id == 12345
        assert entry.dest == "loras"
        assert entry.version_id is None
        assert entry.sha256 is None

    def test_optional_fields_settable(self) -> None:
        entry = CivitAIModelEntry(model_id=1, dest="loras", version_id=99, sha256="b" * 64)
        assert entry.version_id == 99
        assert entry.sha256 == "b" * 64


class TestURLModelEntry:
    def test_required_fields(self) -> None:
        entry = URLModelEntry(url="https://example.com/m.safetensors", dest="checkpoints")
        assert entry.url == "https://example.com/m.safetensors"
        assert entry.dest == "checkpoints"
        assert entry.sha256 is None

    def test_sha256_can_be_set(self) -> None:
        entry = URLModelEntry(
            url="https://example.com/m.pt", dest="cp", sha256="c" * 64
        )
        assert entry.sha256 == "c" * 64


class TestModelEntryUnion:
    def test_all_types_are_valid_model_entries(self) -> None:
        manifest: list[ModelEntry] = [
            HFModelEntry(repo_id="org/a", filename="a.safetensors", dest="checkpoints"),
            CivitAIModelEntry(model_id=99, dest="loras"),
            URLModelEntry(url="https://example.com/b.pt", dest="unet"),
        ]
        assert len(manifest) == 3


# ---------------------------------------------------------------------------
# download_models — signature
# ---------------------------------------------------------------------------


class TestSignature:
    def test_accepts_empty_manifest(self, tmp_path: Path) -> None:
        download_models([], models_dir=tmp_path)

    def test_accepts_quiet_flag(self, tmp_path: Path) -> None:
        download_models([], models_dir=tmp_path, quiet=True)

    def test_models_dir_optional_for_empty_manifest(self) -> None:
        download_models([])


# ---------------------------------------------------------------------------
# URLModelEntry — dispatch, idempotency, dest resolution, errors
# ---------------------------------------------------------------------------


class TestURLEntry:
    def test_downloads_file(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert (tmp_path / "model.safetensors").exists()

    def test_resolves_relative_dest_against_models_dir(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest="checkpoints/m.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry], models_dir=tmp_path)
        assert (tmp_path / "checkpoints" / "m.safetensors").exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest=tmp_path / "a" / "b" / "c" / "m.safetensors",
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert (tmp_path / "a" / "b" / "c" / "m.safetensors").exists()

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"original")
        urlopen_mock = MagicMock()
        entry = URLModelEntry(url="https://example.com/model.safetensors", dest=dest)
        with patch("urllib.request.urlopen", urlopen_mock):
            download_models([entry])
        urlopen_mock.assert_not_called()
        assert dest.read_bytes() == b"original"

    def test_no_extension_appends_url_basename(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/flux.safetensors",
            dest=tmp_path / "checkpoints",
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert (tmp_path / "checkpoints" / "flux.safetensors").exists()

    def test_download_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/missing.safetensors",
            dest=tmp_path / "missing.safetensors",
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="https://example.com/missing.safetensors"):
                download_models([entry])

    def test_relative_dest_without_models_dir_raises(self) -> None:
        entry = URLModelEntry(
            url="https://example.com/m.safetensors", dest="checkpoints/m.safetensors"
        )
        with pytest.raises(RuntimeError, match="models_dir must be provided"):
            download_models([entry])


# ---------------------------------------------------------------------------
# HFModelEntry — dispatch, idempotency, auth, errors
# ---------------------------------------------------------------------------


class TestHFEntry:
    def test_downloads_hf_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(FAKE_BODY)
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
            repo_id="org/model", filename="model.safetensors", token=None
        )

    def test_passes_hf_token_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        cached_file = tmp_path / "cached.safetensors"
        cached_file.write_bytes(FAKE_BODY)
        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached_file)

        entry = HFModelEntry(
            repo_id="org/gated", filename="model.safetensors", dest=tmp_path / "model.safetensors"
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])

        _, kwargs = hf_mock.hf_hub_download.call_args
        assert kwargs["token"] == "hf_secret"

    def test_gated_model_without_token_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_download_failure_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"original")
        hf_mock = MagicMock()
        entry = HFModelEntry(repo_id="org/model", filename="model.safetensors", dest=dest)
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry])
        hf_mock.hf_hub_download.assert_not_called()

    def test_token_not_in_error_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
# CivitAIModelEntry — dispatch, idempotency, auth, errors
# ---------------------------------------------------------------------------


class TestCivitAIEntry:
    def test_raises_without_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        entry = CivitAIModelEntry(model_id=12345, dest=tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="CIVITAI_API_KEY"):
            download_models([entry])

    def test_error_mentions_model_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        entry = CivitAIModelEntry(model_id=99, dest=tmp_path / "m.safetensors")
        with pytest.raises(RuntimeError) as exc_info:
            download_models([entry])
        assert "99" in str(exc_info.value)

    def test_skips_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "lora.safetensors"
        dest.write_bytes(b"original")
        entry = CivitAIModelEntry(model_id=1, dest=dest, version_id=42)
        with patch("urllib.request.urlopen") as urlopen_mock:
            download_models([entry])
        urlopen_mock.assert_not_called()

    def test_downloads_with_version_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "lora.safetensors"
        with patch("urllib.request.urlopen", side_effect=_civitai_urlopen(FAKE_BODY)):
            download_models([CivitAIModelEntry(model_id=1, dest=dest, version_id=42)])
        assert dest.exists()

    def test_download_failure_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")
        dest = tmp_path / "model.safetensors"

        def fail_on_download(req: object, *args: object, **kwargs: object) -> io.BytesIO:
            url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[union-attr]
            if "/api/v1/" in url:
                return _make_cm(_json_bytes({"files": [{"name": "model.safetensors"}]}))
            raise urllib.error.URLError("connection reset")

        with patch("urllib.request.urlopen", side_effect=fail_on_download):
            with pytest.raises(RuntimeError, match="12345"):
                download_models([CivitAIModelEntry(model_id=12345, dest=dest, version_id=7)])

    def test_api_key_not_in_dest_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "my_secret_civitai_key"
        monkeypatch.setenv("CIVITAI_API_KEY", secret)
        dest = tmp_path / "model.safetensors"
        with patch("urllib.request.urlopen", side_effect=_civitai_urlopen(FAKE_BODY)):
            download_models([CivitAIModelEntry(model_id=1, dest=dest, version_id=5)])
        assert secret not in str(dest)


# ---------------------------------------------------------------------------
# Mixed manifest dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_mixed_manifest_all_skipped_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("CIVITAI_API_KEY", "fake_key")

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

        urlopen_mock.assert_not_called()
        hf_mock.hf_hub_download.assert_not_called()


# ---------------------------------------------------------------------------
# SHA256 integrity verification
# ---------------------------------------------------------------------------


class TestSHA256:
    def test_correct_hash_passes(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=CORRECT_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert (tmp_path / "model.safetensors").exists()

    def test_wrong_hash_raises_value_error(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=WRONG_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                download_models([entry])

    def test_hash_failure_deletes_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=WRONG_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            with pytest.raises(ValueError):
                download_models([entry])
        assert not dest.exists()

    def test_existing_file_correct_hash_skips_download(self, tmp_path: Path) -> None:
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

    def test_existing_file_wrong_hash_triggers_redownload(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"stale_data")
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=dest,
            sha256=CORRECT_SHA256,
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert dest.read_bytes() == FAKE_BODY

    def test_no_sha256_skips_verification(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
            sha256=None,
        )
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)):
            download_models([entry])
        assert (tmp_path / "model.safetensors").exists()


# ---------------------------------------------------------------------------
# Progress / quiet flag
# ---------------------------------------------------------------------------


class TestQuiet:
    def test_url_quiet_skips_tqdm(self, tmp_path: Path) -> None:
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()
        entry = URLModelEntry(
            url="https://example.com/m.safetensors",
            dest=tmp_path / "m.safetensors",
        )
        with (
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry], quiet=True)
        tqdm_class.assert_not_called()

    def test_url_not_quiet_uses_tqdm_when_available(self, tmp_path: Path) -> None:
        tqdm_module, tqdm_class, _ = _make_tqdm_mocks()
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )

        def fake_urlopen(req: object, *a: object, **kw: object) -> io.BytesIO:
            resp = _make_cm(FAKE_BODY)
            resp.headers = {}  # type: ignore[attr-defined]
            return resp

        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("sys.modules", {"tqdm": tqdm_module}),
        ):
            download_models([entry], quiet=False)
        tqdm_class.assert_called_once()

    def test_hf_quiet_disables_progress_bars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cached = tmp_path / "cached.safetensors"
        cached.write_bytes(FAKE_BODY)
        hf_mock = MagicMock()
        hf_mock.hf_hub_download.return_value = str(cached)
        entry = HFModelEntry(
            repo_id="a/b", filename="x.safetensors", dest=tmp_path / "x.safetensors"
        )
        with patch.dict("sys.modules", {"huggingface_hub": hf_mock}):
            download_models([entry], quiet=True)
        hf_mock.disable_progress_bars.assert_called_once()
        hf_mock.enable_progress_bars.assert_called_once()

    def test_hf_enable_progress_bars_called_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_fallback_without_tqdm(self, tmp_path: Path) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest=tmp_path / "model.safetensors",
        )
        with (
            patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _make_cm(FAKE_BODY)),
            patch.dict("sys.modules", {"tqdm": None}),
        ):
            download_models([entry])
        assert (tmp_path / "model.safetensors").read_bytes() == FAKE_BODY
