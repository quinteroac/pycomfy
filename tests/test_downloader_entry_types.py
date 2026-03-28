"""Tests for US-001 — Model Entry Types in comfy_diffusion.downloader."""

from __future__ import annotations

from pathlib import Path

from comfy_diffusion.downloader import (
    CivitAIModelEntry,
    HFModelEntry,
    ModelEntry,
    URLModelEntry,
)

# ---------------------------------------------------------------------------
# HFModelEntry
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

    def test_sha256_defaults_to_none(self) -> None:
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest="checkpoints",
        )
        assert entry.sha256 is None

    def test_sha256_can_be_set(self) -> None:
        digest = "a" * 64
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest="checkpoints",
            sha256=digest,
        )
        assert entry.sha256 == digest

    def test_dest_accepts_path_object(self) -> None:
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=Path("checkpoints/subfolder"),
        )
        assert isinstance(entry.dest, Path)

    def test_dest_accepts_absolute_path(self) -> None:
        abs_path = Path("/models/checkpoints")
        entry = HFModelEntry(
            repo_id="org/model",
            filename="model.safetensors",
            dest=abs_path,
        )
        assert entry.dest == abs_path


# ---------------------------------------------------------------------------
# CivitAIModelEntry
# ---------------------------------------------------------------------------


class TestCivitAIModelEntry:
    def test_required_fields(self) -> None:
        entry = CivitAIModelEntry(model_id=12345, dest="loras")
        assert entry.model_id == 12345
        assert entry.dest == "loras"

    def test_optional_fields_default_to_none(self) -> None:
        entry = CivitAIModelEntry(model_id=1, dest="loras")
        assert entry.version_id is None
        assert entry.sha256 is None

    def test_version_id_can_be_set(self) -> None:
        entry = CivitAIModelEntry(model_id=1, dest="loras", version_id=999)
        assert entry.version_id == 999

    def test_sha256_can_be_set(self) -> None:
        digest = "b" * 64
        entry = CivitAIModelEntry(model_id=1, dest="loras", sha256=digest)
        assert entry.sha256 == digest

    def test_dest_accepts_path_object(self) -> None:
        entry = CivitAIModelEntry(model_id=1, dest=Path("loras/subfolder"))
        assert isinstance(entry.dest, Path)


# ---------------------------------------------------------------------------
# URLModelEntry
# ---------------------------------------------------------------------------


class TestURLModelEntry:
    def test_required_fields(self) -> None:
        entry = URLModelEntry(
            url="https://example.com/model.safetensors",
            dest="checkpoints",
        )
        assert entry.url == "https://example.com/model.safetensors"
        assert entry.dest == "checkpoints"

    def test_sha256_defaults_to_none(self) -> None:
        entry = URLModelEntry(url="https://example.com/m.pt", dest="checkpoints")
        assert entry.sha256 is None

    def test_sha256_can_be_set(self) -> None:
        digest = "c" * 64
        entry = URLModelEntry(
            url="https://example.com/m.pt",
            dest="checkpoints",
            sha256=digest,
        )
        assert entry.sha256 == digest

    def test_dest_accepts_path_object(self) -> None:
        entry = URLModelEntry(url="https://example.com/m.pt", dest=Path("checkpoints"))
        assert isinstance(entry.dest, Path)

    def test_dest_accepts_absolute_path(self) -> None:
        abs_path = Path("/models/checkpoints/m.pt")
        entry = URLModelEntry(url="https://example.com/m.pt", dest=abs_path)
        assert entry.dest == abs_path


# ---------------------------------------------------------------------------
# ModelEntry union
# ---------------------------------------------------------------------------


class TestModelEntryUnion:
    def test_hf_entry_is_model_entry(self) -> None:
        entry: ModelEntry = HFModelEntry(
            repo_id="org/model", filename="m.safetensors", dest="checkpoints"
        )
        assert isinstance(entry, HFModelEntry)

    def test_civitai_entry_is_model_entry(self) -> None:
        entry: ModelEntry = CivitAIModelEntry(model_id=1, dest="loras")
        assert isinstance(entry, CivitAIModelEntry)

    def test_url_entry_is_model_entry(self) -> None:
        entry: ModelEntry = URLModelEntry(
            url="https://example.com/m.pt", dest="checkpoints"
        )
        assert isinstance(entry, URLModelEntry)

    def test_manifest_list_accepts_mixed_entry_types(self) -> None:
        manifest: list[ModelEntry] = [
            HFModelEntry(repo_id="org/a", filename="a.safetensors", dest="checkpoints"),
            CivitAIModelEntry(model_id=99, dest="loras"),
            URLModelEntry(url="https://example.com/b.pt", dest="unet"),
        ]
        assert len(manifest) == 3
