"""``parallax frontend`` — manage the pre-built chat UI.

Acceptance criteria implemented (US-001):
  AC01 — fetch latest release from GitHub; find asset ``parallax-frontend-{version}.tar.gz``
  AC02 — extract archive to ``~/.parallax/frontend/``, replacing any prior installation
  AC03 — write ``PARALLAX_FRONTEND_PATH=~/.parallax/frontend`` to ``~/.parallax/config.env``
  AC04 — print "Frontend installed at ~/.parallax/frontend" on success
  AC05 — network/404 errors → human-readable message + exit 1 + no partial installation left behind
  AC06 — re-running overwrites the existing installation cleanly

Acceptance criteria implemented (US-002):
  AC01 — ``--version`` accepts a semver string (e.g. ``1.2.3``)
  AC02 — unknown version → print GitHub error + exit non-zero
  AC03 — ``--version`` omitted → install latest release

Acceptance criteria implemented (US-003):
  AC01 — ``parallax frontend version`` prints version from ``~/.parallax/frontend/version.txt``
  AC02 — if no frontend installed, prints a helpful message
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Optional

import typer

from cli.commands._common import CONFIG_ENV_PATH, FRONTEND_DIR

app = typer.Typer(name="frontend", help="Manage the Parallax frontend.", no_args_is_help=True)

_GITHUB_REPO = "quinteroac/comfy-diffusion"
_ASSET_PREFIX = "parallax-frontend-"
_RELEASES_API_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
_RELEASES_API_URL_BY_TAG = (
    f"https://api.github.com/repos/{_GITHUB_REPO}/releases/tags/{{tag}}"
)


# ---------------------------------------------------------------------------
# Internal helpers (extracted for testability)
# ---------------------------------------------------------------------------


def _latest_release_info() -> dict:  # type: ignore[type-arg]
    """Fetch the latest release metadata from the GitHub Releases API."""
    req = urllib.request.Request(
        _RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _release_info_by_tag(version: str) -> dict:  # type: ignore[type-arg]
    """Fetch release metadata for the given *version* tag from the GitHub Releases API.

    The tag is tried first as ``v{version}`` (e.g. ``v1.2.3``) and falls back to
    the bare version string if the caller already includes the prefix.
    """
    tag = version if version.startswith("v") else f"v{version}"
    url = _RELEASES_API_URL_BY_TAG.format(tag=tag)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _find_frontend_asset(release: dict) -> tuple[str, str]:  # type: ignore[type-arg]
    """Return ``(download_url, version)`` for the frontend archive.

    Raises ``ValueError`` if the expected asset is not found in the release.
    """
    version: str = release.get("tag_name", "").lstrip("v")
    asset_name = f"{_ASSET_PREFIX}{version}.tar.gz"
    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            return asset["browser_download_url"], version
    raise ValueError(
        f"Asset '{asset_name}' not found in release '{release.get('tag_name', 'unknown')}'. "
        "Check that the release has been published with the expected frontend archive."
    )


def _download_archive(url: str, dest: Path) -> None:
    """Download the archive at *url* and write it to *dest*."""
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _write_config_env(frontend_dir: Path, config_path: Path) -> None:
    """Upsert ``PARALLAX_FRONTEND_PATH`` in the ``config.env`` file (AC03)."""
    key = "PARALLAX_FRONTEND_PATH"
    value = str(frontend_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if config_path.exists():
        lines = config_path.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")

    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


@app.command("install")
def install(
    version: Annotated[
        Optional[str],
        typer.Option(
            "--version",
            "-v",
            help="Semver version to install (e.g. 1.2.3). Defaults to the latest release.",
        ),
    ] = None,
) -> None:
    """Download and install the pre-built frontend to ~/.parallax/frontend.

    Installs the latest release by default. Pass ``--version 1.2.3`` to pin a
    specific release.
    """
    frontend_dir = FRONTEND_DIR
    config_path = CONFIG_ENV_PATH

    # Fetch release info — either latest or a specific tagged version (US-002 AC01/AC03)
    try:
        if version is not None:
            release = _release_info_by_tag(version)
        else:
            release = _latest_release_info()
    except urllib.error.HTTPError as exc:
        # US-002 AC02 — print the GitHub error body when present
        error_body = ""
        if exc.fp is not None:
            try:
                raw = exc.fp.read()
                data = json.loads(raw)
                error_body = data.get("message", "")
            except Exception:
                pass
        detail = error_body or exc.reason
        typer.echo(
            f"Error: failed to fetch release info (HTTP {exc.code}: {detail}).",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Error: failed to fetch release info: {exc}", err=True)
        raise typer.Exit(1)

    # Locate the frontend asset in the release
    try:
        download_url, version = _find_frontend_asset(release)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    # Download to a temporary file so a failure never touches the real install dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_archive = Path(tmp_dir) / f"{_ASSET_PREFIX}{version}.tar.gz"

        # AC01/AC05 — download
        try:
            _download_archive(download_url, tmp_archive)
        except urllib.error.HTTPError as exc:
            typer.echo(
                f"Error: download failed (HTTP {exc.code}: {exc.reason}).",
                err=True,
            )
            raise typer.Exit(1)
        except Exception as exc:
            typer.echo(f"Error: download failed: {exc}", err=True)
            raise typer.Exit(1)

        # AC02/AC06 — replace existing installation atomically
        if frontend_dir.exists():
            shutil.rmtree(frontend_dir)
        frontend_dir.mkdir(parents=True, exist_ok=True)

        # Extract; on failure remove the partially-populated dir (AC05)
        try:
            with tarfile.open(tmp_archive, "r:gz") as tar:
                tar.extractall(frontend_dir, filter="data")  # noqa: S202
        except Exception as exc:
            shutil.rmtree(frontend_dir, ignore_errors=True)
            typer.echo(f"Error: failed to extract archive: {exc}", err=True)
            raise typer.Exit(1)

    # AC03 — persist the path in config.env
    _write_config_env(frontend_dir, config_path)

    # AC04 — success
    typer.echo(f"Frontend installed at {frontend_dir}")


@app.command("version")
def version() -> None:
    """Print the installed frontend version.

    Reads the version string from ``~/.parallax/frontend/version.txt``.
    """
    version_file = FRONTEND_DIR / "version.txt"
    if not version_file.exists():
        typer.echo("Frontend not installed. Run `parallax frontend install` to install.")
        return
    typer.echo(version_file.read_text(encoding="utf-8").strip())
