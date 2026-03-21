"""Internal runtime bootstrap for comfy_diffusion.

Path insertion is intentionally lightweight and import-safe: this module must not
import torch or comfy internals just to make ComfyUI discoverable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COMFYUI_PINNED_TAG = "v0.18.0"
COMFYUI_PINNED_ARCHIVE_URL = (
    "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/"
    f"{COMFYUI_PINNED_TAG}.zip"
)


def _comfyui_root() -> Path:
    """Return the absolute path to the vendored ComfyUI directory."""
    package_dir = Path(__file__).resolve().parent

    # Preferred layout: repo_root/vendor/ComfyUI (vendored git submodule).
    repo_vendor = package_dir.parent / "vendor" / "ComfyUI"
    if repo_vendor.exists():
        return repo_vendor

    # Back-compat layout (older iterations): comfy_diffusion/vendor/ComfyUI.
    package_vendor = package_dir / "vendor" / "ComfyUI"
    return package_vendor


def _has_comfyui_runtime(comfyui_root: Path) -> bool:
    """Return True if the ComfyUI runtime directory looks initialized."""
    return comfyui_root.is_dir() and (comfyui_root / "comfy").is_dir()


def _download_and_extract_pinned_comfyui(comfyui_root: Path) -> None:
    """Download and extract the pinned ComfyUI release into vendor/ComfyUI."""
    vendor_dir = comfyui_root.parent
    vendor_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="comfyui-download-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "comfyui.zip"

        urllib.request.urlretrieve(COMFYUI_PINNED_ARCHIVE_URL, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_dir)

        extracted_candidates = list(tmp_dir.glob("ComfyUI-*"))
        if not extracted_candidates:
            raise RuntimeError("Downloaded ComfyUI archive had unexpected structure.")

        extracted_root = extracted_candidates[0]
        if comfyui_root.exists():
            shutil.rmtree(comfyui_root)
        shutil.move(str(extracted_root), str(comfyui_root))

    if not _has_comfyui_runtime(comfyui_root):
        raise RuntimeError("ComfyUI runtime download completed but content is invalid.")


def ensure_comfyui_available() -> Path:
    """Ensure vendored ComfyUI exists; download pinned release if missing."""
    comfyui_root = _comfyui_root()

    if not _has_comfyui_runtime(comfyui_root):
        _download_and_extract_pinned_comfyui(comfyui_root)

    return comfyui_root


def ensure_comfyui_on_path() -> Path:
    """Ensure vendored ComfyUI is available and importable; return the inserted path."""
    comfyui_root = ensure_comfyui_available()
    comfyui_root_str = str(comfyui_root)

    if comfyui_root_str not in sys.path:
        sys.path.insert(0, comfyui_root_str)

    return comfyui_root
