"""Tests for US-001 it_000047 — parallax frontend install CLI command.

Covers:
  AC01 — downloads latest pre-built frontend archive from GitHub Releases
  AC02 — extracts to ~/.parallax/frontend/, replacing any previous installation
  AC03 — writes PARALLAX_FRONTEND_PATH=~/.parallax/frontend to config.env
  AC04 — prints "Frontend installed at <path>" on success
  AC05 — on network/404 error: human-readable message + exit 1 + no partial install
  AC06 — re-running overwrites existing installation cleanly
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from cli.commands.frontend import (
    _find_frontend_asset,
    _write_config_env,
    app,
)

runner = CliRunner()

# Wrap the frontend sub-app in a parent group so `["frontend", "install"]` routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="frontend")

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_RELEASE = {
    "tag_name": "v1.3.4",
    "assets": [
        {
            "name": "parallax-frontend-1.3.4.tar.gz",
            "browser_download_url": "https://example.com/parallax-frontend-1.3.4.tar.gz",
        }
    ],
}


def _make_tar_gz_bytes(files: dict[str, str]) -> bytes:
    """Return an in-memory .tar.gz archive containing *files* (name → content)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


_FAKE_ARCHIVE_BYTES = _make_tar_gz_bytes(
    {"index.html": "<html><body>Parallax UI</body></html>", "assets/main.js": "console.log(1)"}
)


def _make_urlopen(archive: bytes = _FAKE_ARCHIVE_BYTES):
    """Return a fake urlopen callable that serves the release JSON and archive bytes."""

    def _fake_urlopen(req_or_url, timeout=None):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        resp = MagicMock()
        resp.read.return_value = (
            json.dumps(_FAKE_RELEASE).encode() if "releases/latest" in url else archive
        )
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    return _fake_urlopen


def _run(tmp_path: Path, urlopen=None, archive: bytes = _FAKE_ARCHIVE_BYTES):
    """Invoke ``parallax frontend install`` with path and network mocks."""
    frontend_dir = tmp_path / "frontend"
    config_path = tmp_path / "config.env"
    if urlopen is None:
        urlopen = _make_urlopen(archive)
    with (
        patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir),
        patch("cli.commands.frontend.CONFIG_ENV_PATH", config_path),
        patch("urllib.request.urlopen", urlopen),
    ):
        result = runner.invoke(_cli, ["frontend", "install"])
    return result, frontend_dir, config_path


# ---------------------------------------------------------------------------
# AC01 — downloads from GitHub Releases
# ---------------------------------------------------------------------------


class TestAC01Downloads:
    def test_install_calls_github_api(self, tmp_path: Path) -> None:
        called_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            called_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        result, _, _ = _run(tmp_path, urlopen=tracking_urlopen)
        assert result.exit_code == 0
        assert any("releases/latest" in u for u in called_urls), "GitHub API was not called"

    def test_install_fetches_versioned_asset(self, tmp_path: Path) -> None:
        downloaded_urls: list[str] = []

        def tracking_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            downloaded_urls.append(url)
            return _make_urlopen()(req_or_url, timeout=timeout)

        _run(tmp_path, urlopen=tracking_urlopen)
        assert any("parallax-frontend-1.3.4.tar.gz" in u for u in downloaded_urls)


# ---------------------------------------------------------------------------
# AC02 — extracts to frontend_dir
# ---------------------------------------------------------------------------


class TestAC02Extraction:
    def test_frontend_dir_created(self, tmp_path: Path) -> None:
        result, frontend_dir, _ = _run(tmp_path)
        assert result.exit_code == 0
        assert frontend_dir.is_dir()

    def test_index_html_extracted(self, tmp_path: Path) -> None:
        _, frontend_dir, _ = _run(tmp_path)
        assert (frontend_dir / "index.html").exists()
        assert "Parallax UI" in (frontend_dir / "index.html").read_text()

    def test_existing_install_replaced(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir(parents=True)
        (frontend_dir / "old_file.txt").write_text("stale")

        _run(tmp_path)

        assert not (frontend_dir / "old_file.txt").exists(), "stale file should be removed"


# ---------------------------------------------------------------------------
# AC03 — config.env written
# ---------------------------------------------------------------------------


class TestAC03ConfigEnv:
    def test_config_env_created(self, tmp_path: Path) -> None:
        _, _, config_path = _run(tmp_path)
        assert config_path.exists()

    def test_config_env_contains_key(self, tmp_path: Path) -> None:
        _, _, config_path = _run(tmp_path)
        assert "PARALLAX_FRONTEND_PATH=" in config_path.read_text(encoding="utf-8")

    def test_config_env_value_is_frontend_dir(self, tmp_path: Path) -> None:
        _, frontend_dir, config_path = _run(tmp_path)
        content = config_path.read_text(encoding="utf-8")
        assert f"PARALLAX_FRONTEND_PATH={frontend_dir}" in content

    def test_config_env_preserves_existing_keys(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        config_path = tmp_path / "config.env"
        config_path.write_text("OTHER_KEY=other_value\n", encoding="utf-8")

        _run(tmp_path)

        assert "OTHER_KEY=other_value" in config_path.read_text(encoding="utf-8")

    def test_config_env_updates_existing_key(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        config_path = tmp_path / "config.env"
        config_path.write_text("PARALLAX_FRONTEND_PATH=/old/path\n", encoding="utf-8")

        _run(tmp_path)

        content = config_path.read_text(encoding="utf-8")
        assert "/old/path" not in content
        assert f"PARALLAX_FRONTEND_PATH={frontend_dir}" in content


# ---------------------------------------------------------------------------
# AC04 — success message
# ---------------------------------------------------------------------------


class TestAC04SuccessMessage:
    def test_success_message_printed(self, tmp_path: Path) -> None:
        result, frontend_dir, _ = _run(tmp_path)
        assert "Frontend installed at" in result.output
        assert str(frontend_dir) in result.output

    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        result, _, _ = _run(tmp_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC05 — error handling + no partial installation
# ---------------------------------------------------------------------------


class TestAC05ErrorHandling:
    def test_http_error_on_release_fetch(self, tmp_path: Path) -> None:
        import urllib.error

        def _failing_urlopen(req_or_url, timeout=None):
            raise urllib.error.HTTPError(
                url="https://api.github.com/...", code=404, msg="Not Found",
                hdrs=MagicMock(), fp=None,  # type: ignore[arg-type]
            )

        result, _, _ = _run(tmp_path, urlopen=_failing_urlopen)
        assert result.exit_code != 0

    def test_network_error_exits_nonzero(self, tmp_path: Path) -> None:
        import urllib.error

        def _failing_urlopen(req_or_url, timeout=None):
            raise urllib.error.URLError("Network unreachable")

        result, _, _ = _run(tmp_path, urlopen=_failing_urlopen)
        assert result.exit_code != 0

    def test_no_partial_install_on_download_failure(self, tmp_path: Path) -> None:
        import urllib.error

        frontend_dir = tmp_path / "frontend"
        config_path = tmp_path / "config.env"

        def _partial_urlopen(req_or_url, timeout=None):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            if "releases/latest" in url:
                resp = MagicMock()
                resp.read.return_value = json.dumps(_FAKE_RELEASE).encode()
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            raise urllib.error.HTTPError(
                url=url, code=404, msg="Not Found", hdrs=MagicMock(), fp=None  # type: ignore[arg-type]
            )

        with (
            patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir),
            patch("cli.commands.frontend.CONFIG_ENV_PATH", config_path),
            patch("urllib.request.urlopen", _partial_urlopen),
        ):
            result = runner.invoke(_cli, ["frontend", "install"])

        assert result.exit_code != 0
        assert not frontend_dir.exists() or not list(frontend_dir.iterdir())

    def test_error_message_is_human_readable(self, tmp_path: Path) -> None:
        import urllib.error

        def _failing_urlopen(req_or_url, timeout=None):
            raise urllib.error.HTTPError(
                url="https://api.github.com/...", code=503, msg="Service Unavailable",
                hdrs=MagicMock(), fp=None,  # type: ignore[arg-type]
            )

        result, _, _ = _run(tmp_path, urlopen=_failing_urlopen)
        assert result.exit_code != 0
        assert "Error" in result.output
        assert "503" in result.output or "Service Unavailable" in result.output


# ---------------------------------------------------------------------------
# AC06 — re-install overwrites cleanly
# ---------------------------------------------------------------------------


class TestAC06Reinstall:
    def test_second_install_succeeds(self, tmp_path: Path) -> None:
        result1, _, _ = _run(tmp_path)
        result2, _, _ = _run(tmp_path)
        assert result1.exit_code == 0
        assert result2.exit_code == 0

    def test_second_install_overwrites_files(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        config_path = tmp_path / "config.env"

        first_archive = _make_tar_gz_bytes({"index.html": "first version"})
        second_archive = _make_tar_gz_bytes({"index.html": "second version"})

        with (
            patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir),
            patch("cli.commands.frontend.CONFIG_ENV_PATH", config_path),
            patch("urllib.request.urlopen", _make_urlopen(first_archive)),
        ):
            runner.invoke(_cli, ["frontend", "install"])

        assert "first version" in (frontend_dir / "index.html").read_text()

        with (
            patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir),
            patch("cli.commands.frontend.CONFIG_ENV_PATH", config_path),
            patch("urllib.request.urlopen", _make_urlopen(second_archive)),
        ):
            runner.invoke(_cli, ["frontend", "install"])

        assert "second version" in (frontend_dir / "index.html").read_text()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestFindFrontendAsset:
    def test_finds_expected_asset(self) -> None:
        url, version = _find_frontend_asset(_FAKE_RELEASE)
        assert version == "1.3.4"
        assert "parallax-frontend-1.3.4.tar.gz" in url

    def test_raises_when_asset_missing(self) -> None:
        release = {"tag_name": "v2.0.0", "assets": []}
        with pytest.raises(ValueError, match="parallax-frontend-2.0.0.tar.gz"):
            _find_frontend_asset(release)

    def test_strips_v_prefix_from_tag(self) -> None:
        release = {
            "tag_name": "v99.0.0",
            "assets": [
                {
                    "name": "parallax-frontend-99.0.0.tar.gz",
                    "browser_download_url": "https://example.com/parallax-frontend-99.0.0.tar.gz",
                }
            ],
        }
        _, version = _find_frontend_asset(release)
        assert version == "99.0.0"


class TestWriteConfigEnv:
    def test_creates_file_if_absent(self, tmp_path: Path) -> None:
        config = tmp_path / "config.env"
        _write_config_env(tmp_path / "frontend", config)
        assert config.exists()

    def test_writes_key_value(self, tmp_path: Path) -> None:
        frontend_dir = tmp_path / "frontend"
        config = tmp_path / "config.env"
        _write_config_env(frontend_dir, config)
        assert f"PARALLAX_FRONTEND_PATH={frontend_dir}" in config.read_text()

    def test_upserts_existing_key(self, tmp_path: Path) -> None:
        config = tmp_path / "config.env"
        config.write_text("PARALLAX_FRONTEND_PATH=/old\nOTHER=1\n")
        new_dir = tmp_path / "new_frontend"
        _write_config_env(new_dir, config)
        content = config.read_text()
        assert "/old" not in content
        assert f"PARALLAX_FRONTEND_PATH={new_dir}" in content
        assert "OTHER=1" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        config = tmp_path / "deep" / "nested" / "config.env"
        _write_config_env(tmp_path / "frontend", config)
        assert config.exists()
