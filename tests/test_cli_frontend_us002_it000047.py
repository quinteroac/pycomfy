"""Tests for US-002 it_000047 — parallax frontend install --version <semver>.

Covers:
  AC01 — ``--version`` accepts a semver string (e.g. ``1.2.3``)
  AC02 — unknown version → print GitHub error + exit non-zero
  AC03 — ``--version`` omitted → latest release is installed
"""
from __future__ import annotations

import io
import json
import tarfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from cli.commands.frontend import (
    _release_info_by_tag,
    app,
)

runner = CliRunner()

_cli = typer.Typer()
_cli.add_typer(app, name="frontend")

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_RELEASE_LATEST = {
    "tag_name": "v2.0.0",
    "assets": [
        {
            "name": "parallax-frontend-2.0.0.tar.gz",
            "browser_download_url": "https://example.com/parallax-frontend-2.0.0.tar.gz",
        }
    ],
}

_FAKE_RELEASE_PINNED = {
    "tag_name": "v1.2.3",
    "assets": [
        {
            "name": "parallax-frontend-1.2.3.tar.gz",
            "browser_download_url": "https://example.com/parallax-frontend-1.2.3.tar.gz",
        }
    ],
}


def _make_tar_gz_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


_FAKE_ARCHIVE = _make_tar_gz_bytes({"index.html": "<html>Parallax</html>"})


def _make_urlopen(latest_release: dict = _FAKE_RELEASE_LATEST, pinned_release: dict = _FAKE_RELEASE_PINNED):
    """Return a fake urlopen that routes to the correct release JSON or archive bytes."""

    def _fake_urlopen(req_or_url, timeout=None):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        resp = MagicMock()
        if "releases/latest" in url:
            resp.read.return_value = json.dumps(latest_release).encode()
        elif "releases/tags/" in url:
            resp.read.return_value = json.dumps(pinned_release).encode()
        else:
            resp.read.return_value = _FAKE_ARCHIVE
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    return _fake_urlopen


def _run(tmp_path: Path, args: list[str], urlopen=None):
    frontend_dir = tmp_path / "frontend"
    config_path = tmp_path / "config.env"
    if urlopen is None:
        urlopen = _make_urlopen()
    with (
        patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir),
        patch("cli.commands.frontend.CONFIG_ENV_PATH", config_path),
        patch("urllib.request.urlopen", urlopen),
    ):
        result = runner.invoke(_cli, args)
    return result, frontend_dir, config_path


# ---------------------------------------------------------------------------
# AC01 — --version accepts a semver string
# ---------------------------------------------------------------------------


class TestAC01VersionOption:
    def test_version_option_accepted(self, tmp_path: Path) -> None:
        result, _, _ = _run(tmp_path, ["frontend", "install", "--version", "1.2.3"])
        assert result.exit_code == 0

    def test_version_option_short_flag(self, tmp_path: Path) -> None:
        result, _, _ = _run(tmp_path, ["frontend", "install", "-v", "1.2.3"])
        assert result.exit_code == 0

    def test_pinned_version_archive_downloaded(self, tmp_path: Path) -> None:
        """When --version is given, the pinned-version archive is downloaded."""
        downloaded_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            downloaded_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        _run(tmp_path, ["frontend", "install", "--version", "1.2.3"], urlopen=tracking_urlopen)
        assert any("parallax-frontend-1.2.3.tar.gz" in u for u in downloaded_urls)

    def test_versioned_tag_api_called(self, tmp_path: Path) -> None:
        """GitHub releases/tags/<tag> endpoint is called when --version is given."""
        called_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        _run(tmp_path, ["frontend", "install", "--version", "1.2.3"], urlopen=tracking_urlopen)
        assert any("releases/tags/v1.2.3" in u for u in called_urls)

    def test_version_without_v_prefix_resolves_tag(self, tmp_path: Path) -> None:
        """Bare semver (``1.2.3``) is auto-prefixed to ``v1.2.3`` for the API call."""
        called_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        _run(tmp_path, ["frontend", "install", "--version", "1.2.3"], urlopen=tracking_urlopen)
        assert any("v1.2.3" in u for u in called_urls), (
            "Expected the API call to include the v-prefixed tag"
        )

    def test_install_succeeds_with_semver(self, tmp_path: Path) -> None:
        result, frontend_dir, _ = _run(tmp_path, ["frontend", "install", "--version", "1.2.3"])
        assert result.exit_code == 0
        assert frontend_dir.is_dir()
        assert (frontend_dir / "index.html").exists()


# ---------------------------------------------------------------------------
# AC02 — unknown version → print GitHub error + exit non-zero
# ---------------------------------------------------------------------------


class TestAC02UnknownVersion:
    def _make_not_found_urlopen(self, error_message: str = "Not Found"):
        """urlopen that raises 404 with a JSON body for tag lookups."""

        def _urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            if "releases/tags/" in url:
                fp = io.BytesIO(json.dumps({"message": error_message}).encode())
                raise urllib.error.HTTPError(
                    url=url, code=404, msg="Not Found", hdrs=MagicMock(), fp=fp  # type: ignore[arg-type]
                )
            # Should not reach archive download
            resp = MagicMock()
            resp.read.return_value = _FAKE_ARCHIVE
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        return _urlopen

    def test_exits_nonzero_for_unknown_version(self, tmp_path: Path) -> None:
        result, _, _ = _run(
            tmp_path,
            ["frontend", "install", "--version", "9.9.9"],
            urlopen=self._make_not_found_urlopen(),
        )
        assert result.exit_code != 0

    def test_prints_github_error_message(self, tmp_path: Path) -> None:
        result, _, _ = _run(
            tmp_path,
            ["frontend", "install", "--version", "9.9.9"],
            urlopen=self._make_not_found_urlopen("Not Found"),
        )
        assert result.exit_code != 0
        assert "Not Found" in result.output or "404" in result.output

    def test_no_partial_install_on_404(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        result, _, _ = _run(
            tmp_path,
            ["frontend", "install", "--version", "9.9.9"],
            urlopen=self._make_not_found_urlopen(),
        )
        assert result.exit_code != 0
        assert not frontend_dir.exists() or not list(frontend_dir.iterdir())

    def test_error_output_contains_error_keyword(self, tmp_path: Path) -> None:
        result, _, _ = _run(
            tmp_path,
            ["frontend", "install", "--version", "9.9.9"],
            urlopen=self._make_not_found_urlopen(),
        )
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# AC03 — --version omitted → latest release installed
# ---------------------------------------------------------------------------


class TestAC03LatestWhenVersionOmitted:
    def test_latest_release_api_called(self, tmp_path: Path) -> None:
        called_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        result, _, _ = _run(tmp_path, ["frontend", "install"], urlopen=tracking_urlopen)
        assert result.exit_code == 0
        assert any("releases/latest" in u for u in called_urls)

    def test_tagged_api_not_called_without_version(self, tmp_path: Path) -> None:
        called_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        _run(tmp_path, ["frontend", "install"], urlopen=tracking_urlopen)
        assert not any("releases/tags/" in u for u in called_urls)

    def test_installs_latest_archive(self, tmp_path: Path) -> None:
        downloaded_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            downloaded_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        result, _, _ = _run(tmp_path, ["frontend", "install"], urlopen=tracking_urlopen)
        assert result.exit_code == 0
        assert any("parallax-frontend-2.0.0.tar.gz" in u for u in downloaded_urls)

    def test_latest_install_creates_frontend_dir(self, tmp_path: Path) -> None:
        result, frontend_dir, _ = _run(tmp_path, ["frontend", "install"])
        assert result.exit_code == 0
        assert frontend_dir.is_dir()


# ---------------------------------------------------------------------------
# Unit test — _release_info_by_tag helper
# ---------------------------------------------------------------------------


class TestReleaseInfoByTag:
    def test_prepends_v_prefix(self) -> None:
        called_urls: list[str] = []

        def fake_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            resp = MagicMock()
            resp.read.return_value = json.dumps(_FAKE_RELEASE_PINNED).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            _release_info_by_tag("1.2.3")

        assert any("v1.2.3" in u for u in called_urls)

    def test_does_not_double_prefix(self) -> None:
        """If caller passes ``v1.2.3`` it should not become ``vv1.2.3``."""
        called_urls: list[str] = []

        def fake_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            resp = MagicMock()
            resp.read.return_value = json.dumps(_FAKE_RELEASE_PINNED).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            _release_info_by_tag("v1.2.3")

        assert all("vv" not in u for u in called_urls)
        assert any("v1.2.3" in u for u in called_urls)

    def test_returns_release_dict(self) -> None:
        def fake_urlopen(req_or_url, timeout=None):
            resp = MagicMock()
            resp.read.return_value = json.dumps(_FAKE_RELEASE_PINNED).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            result = _release_info_by_tag("1.2.3")

        assert result["tag_name"] == "v1.2.3"
