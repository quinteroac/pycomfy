"""Manifest entry types and download utilities for comfy-diffusion.

Public API
----------
- HFModelEntry      — model file hosted on Hugging Face Hub
- CivitAIModelEntry — model file hosted on CivitAI
- URLModelEntry     — model file at an arbitrary HTTPS/HTTP URL
- ModelEntry        — union alias for the three entry types
- download_models   — fetch all entries in a manifest to local paths

``dest`` is always resolved relative to ``models_dir`` when it is a
relative path.  Absolute ``dest`` values are used as-is.  If ``dest``
has no file extension it is treated as a directory and the appropriate
filename is appended automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "HFModelEntry",
    "CivitAIModelEntry",
    "URLModelEntry",
    "ModelEntry",
    "download_models",
]


@dataclass
class HFModelEntry:
    """A model file downloaded from Hugging Face Hub.

    Parameters
    ----------
    repo_id:
        Repository identifier, e.g. ``"black-forest-labs/FLUX.1-schnell"``.
    filename:
        File path within the repository, e.g. ``"flux1-schnell.safetensors"``.
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
        If no file extension is present the basename of ``filename`` is appended.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    repo_id: str
    filename: str
    dest: str | Path
    sha256: str | None = field(default=None)


@dataclass
class CivitAIModelEntry:
    """A model file downloaded from CivitAI.

    Parameters
    ----------
    model_id:
        Numeric CivitAI model ID.
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
        If no file extension is present the filename from the CivitAI API is appended.
    version_id:
        Specific model-version ID.  ``None`` selects the latest version.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    model_id: int
    dest: str | Path
    version_id: int | None = field(default=None)
    sha256: str | None = field(default=None)


@dataclass
class URLModelEntry:
    """A model file downloaded from an arbitrary URL.

    Parameters
    ----------
    url:
        Direct download URL (``https://`` or ``http://``).
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
        If no file extension is present the URL basename is appended.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    url: str
    dest: str | Path
    sha256: str | None = field(default=None)


# Union type alias — use as the annotation type for manifest lists.
ModelEntry = HFModelEntry | CivitAIModelEntry | URLModelEntry


# ---------------------------------------------------------------------------
# Public download entry point
# ---------------------------------------------------------------------------


def download_models(
    manifest: list[ModelEntry],
    *,
    models_dir: str | Path | None = None,
    quiet: bool = False,
) -> None:
    """Download every entry in *manifest* to its resolved destination path.

    Parameters
    ----------
    manifest:
        List of :class:`ModelEntry` instances describing the files to fetch.
    models_dir:
        Base directory used to resolve relative ``dest`` values.  Required
        when any entry has a relative ``dest``; ignored for absolute paths.
    quiet:
        Suppress informational output (reserved for progress support in a
        future iteration; accepted now to stabilise the public signature).

    Raises
    ------
    RuntimeError
        If a required environment variable is missing, if a download fails,
        or if ``models_dir`` is ``None`` for a relative ``dest``.
    """
    resolved_dir: Path | None = Path(models_dir) if models_dir is not None else None

    for entry in manifest:
        if isinstance(entry, HFModelEntry):
            _download_hf_entry(entry, resolved_dir, quiet=quiet)
        elif isinstance(entry, CivitAIModelEntry):
            _download_civitai_entry(entry, resolved_dir, quiet=quiet)
        elif isinstance(entry, URLModelEntry):
            _download_url_entry(entry, resolved_dir, quiet=quiet)
        else:
            raise RuntimeError(f"Unknown ModelEntry type: {type(entry)!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hash(path: Path, expected: str, *, fresh: bool) -> bool:
    """Verify *path* against *expected* SHA-256 hex digest.

    Returns ``True`` when the digest matches.  When the digest does not match:

    - *fresh* ``True``  (just downloaded): deletes the file and raises
      :class:`ValueError`.
    - *fresh* ``False`` (already present): deletes the file and returns
      ``False`` so the caller can re-download.
    """
    actual = _compute_sha256(path)
    if actual == expected.lower():
        return True
    path.unlink(missing_ok=True)
    if fresh:
        raise ValueError(
            f"SHA-256 mismatch for freshly downloaded {path.name}: "
            f"expected {expected.lower()}, got {actual}"
        )
    return False


def _resolve_dest(dest: str | Path, models_dir: Path | None, fallback_name: str) -> Path:
    """Return the absolute destination *file* path for an entry.

    If *dest* has no file extension it is treated as a directory and
    *fallback_name* is appended as the filename.
    """
    path = Path(dest)
    if not path.is_absolute():
        if models_dir is None:
            raise RuntimeError(
                "models_dir must be provided when dest is a relative path; "
                f"got dest={dest!r}"
            )
        path = models_dir / path
    if not path.suffix:
        path = path / fallback_name
    return path


def _stream_to_file(
    response: Any,
    dest: Path,
    *,
    progress_label: str | None = None,
    quiet: bool = False,
) -> None:
    """Write *response* body to *dest* atomically via a temporary file.

    When *progress_label* is provided and *quiet* is ``False``, a tqdm
    progress bar is shown if tqdm is installed; otherwise falls back to a
    plain copy.
    """
    total: int | None = None
    if not quiet and progress_label is not None:
        try:
            headers = getattr(response, "headers", None)
            raw_len = headers.get("Content-Length") if headers is not None else None
            total = int(raw_len) if raw_len is not None else None
        except (TypeError, ValueError):
            total = None

    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=dest.parent)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            if not quiet and progress_label is not None:
                try:
                    from tqdm import tqdm  # lazy — optional dep

                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=progress_label,
                        leave=True,
                    ) as pbar:
                        for chunk in iter(lambda: response.read(8192), b""):
                            fh.write(chunk)
                            pbar.update(len(chunk))
                except ImportError:
                    shutil.copyfileobj(response, fh)
            else:
                shutil.copyfileobj(response, fh)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_hf_entry(
    entry: HFModelEntry,
    models_dir: Path | None,
    *,
    quiet: bool,
) -> None:
    fallback_name = Path(entry.filename).name
    dest_path = _resolve_dest(entry.dest, models_dir, fallback_name)

    if dest_path.exists():
        if entry.sha256 is None or _verify_hash(dest_path, entry.sha256, fresh=False):
            return  # present and hash matches (or no hash required)
        # hash mismatch — file was deleted; fall through to re-download

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import huggingface_hub  # lazy import — optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download Hugging Face models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    token: str | None = os.environ.get("HF_TOKEN")

    if not quiet:
        print(f"downloading {entry.repo_id}/{entry.filename} → {dest_path}", flush=True)

    if quiet:
        huggingface_hub.disable_progress_bars()
    try:
        cached: str = huggingface_hub.hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.filename,
            token=token,
        )
    except Exception as exc:
        err_lower = str(exc).lower()
        is_auth_error = any(
            kw in err_lower
            for kw in ("gated", "401", "403", "unauthorized", "access", "forbidden")
        )
        if is_auth_error and not token:
            raise RuntimeError(
                f"{entry.repo_id}/{entry.filename} requires authentication. "
                "Set the HF_TOKEN environment variable to your Hugging Face "
                "access token. You can create one at "
                "https://huggingface.co/settings/tokens"
            ) from exc
        raise RuntimeError(
            f"Failed to download {entry.repo_id}/{entry.filename}: {exc}"
        ) from exc
    finally:
        if quiet:
            huggingface_hub.enable_progress_bars()

    if not quiet:
        print(f"copying to {dest_path}", flush=True)
    shutil.copy2(cached, dest_path)

    if entry.sha256 is not None:
        _verify_hash(dest_path, entry.sha256, fresh=True)


def _download_civitai_entry(
    entry: CivitAIModelEntry,
    models_dir: Path | None,
    *,
    quiet: bool,
) -> None:
    api_key: str | None = os.environ.get("CIVITAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"CIVITAI_API_KEY is not set. "
            f"A CivitAI API key is required to download model_id={entry.model_id}. "
            "Set the CIVITAI_API_KEY environment variable to your CivitAI API key. "
            "You can generate one at https://civitai.com/user/account"
        )

    version_id: int | None = entry.version_id
    api_filename: str | None = None

    # Determine version_id and/or filename via CivitAI REST API when needed.
    needs_api_call = version_id is None or not Path(entry.dest).suffix

    if needs_api_call:
        if version_id is not None:
            info_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
        else:
            info_url = f"https://civitai.com/api/v1/models/{entry.model_id}"

        req = urllib.request.Request(
            info_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data: dict[str, Any] = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to fetch CivitAI info for model_id={entry.model_id}: {exc}"
            ) from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"Unexpected CivitAI API response for model_id={entry.model_id}: {exc}"
            ) from exc

        if version_id is None:
            try:
                latest = data["modelVersions"][0]
                version_id = int(latest["id"])
                api_filename = str(latest["files"][0]["name"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Could not parse CivitAI model info for model_id={entry.model_id}: {exc}"
                ) from exc
        else:
            try:
                api_filename = str(data["files"][0]["name"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Could not parse CivitAI version info for "
                    f"model_id={entry.model_id} version_id={version_id}: {exc}"
                ) from exc

    fallback_name = api_filename or f"civitai_{entry.model_id}_{version_id}.bin"
    dest_path = _resolve_dest(entry.dest, models_dir, fallback_name)

    if dest_path.exists():
        if entry.sha256 is None or _verify_hash(dest_path, entry.sha256, fresh=False):
            return  # present and hash matches (or no hash required)
        # hash mismatch — file was deleted; fall through to re-download

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    download_url = f"https://civitai.com/api/download/models/{version_id}"
    dl_req = urllib.request.Request(
        download_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(dl_req) as resp:
            _stream_to_file(resp, dest_path, progress_label=dest_path.name, quiet=quiet)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to download CivitAI model_id={entry.model_id} "
            f"version_id={version_id}: {exc}"
        ) from exc

    if entry.sha256 is not None:
        _verify_hash(dest_path, entry.sha256, fresh=True)


def _download_url_entry(
    entry: URLModelEntry,
    models_dir: Path | None,
    *,
    quiet: bool,
) -> None:
    url_path = urllib.parse.urlparse(entry.url).path
    fallback_name = Path(url_path).name or "model.bin"

    dest_path = _resolve_dest(entry.dest, models_dir, fallback_name)

    if dest_path.exists():
        if entry.sha256 is None or _verify_hash(dest_path, entry.sha256, fresh=False):
            return  # present and hash matches (or no hash required)
        # hash mismatch — file was deleted; fall through to re-download

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        entry.url,
        headers={"User-Agent": "comfy-diffusion/1.0"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            _stream_to_file(resp, dest_path, progress_label=dest_path.name, quiet=quiet)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {entry.url}: {exc}") from exc

    if entry.sha256 is not None:
        _verify_hash(dest_path, entry.sha256, fresh=True)
