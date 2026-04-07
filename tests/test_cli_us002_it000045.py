"""Tests for US-002 (it_000045) — ``parallax mcp install`` command.

AC01: checks ~/.parallax/env + parallax-mcp script; exits 1 if missing.
AC02: locates Claude Desktop config at platform-appropriate path.
AC03: adds/updates mcpServers.parallax-mcp with absolute script path.
AC04: creates config file (with only mcpServers) when absent.
AC05: never overwrites keys other than mcpServers.parallax-mcp.
AC06: prints "MCP server registered. Restart Claude Desktop to apply."
AC07: already registered → prints "Already registered." and exits 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app():
    from cli.main import app
    return app


def _runner():
    return CliRunner()


def _invoke(extra_args: list[str] | None = None, env_exists: bool = True, script_exists: bool = True):
    """Invoke ``parallax mcp install`` with env/script existence controlled via mocks."""
    runner = _runner()
    app = _app()

    with patch("cli.commands.mcp._ENV_DIR", new_callable=lambda: _make_path_mock(env_exists)):
        with patch("cli.commands.mcp._mcp_script_path") as mock_script:
            mock_script.return_value = _FakePath(script_exists)
            result = runner.invoke(app, ["mcp", "install"] + (extra_args or []))
    return result


class _FakePath:
    """Minimal Path-like for testing script existence."""

    def __init__(self, exists: bool, path: str = "/home/user/.parallax/env/bin/parallax-mcp"):
        self._exists = exists
        self._path = path

    def exists(self):
        return self._exists

    def __str__(self):
        return self._path

    def __eq__(self, other):
        return str(self) == str(other)


def _make_path_mock(exists: bool):
    """Return a class whose instances have the given exists() value."""
    class FakeDir(Path):
        _flavour = Path(".")._flavour  # type: ignore[attr-defined]

        def exists(self):
            return exists

    # Use a simpler approach: just return a Path subclass instance factory
    # Actually, let's just use patch.object approach in tests directly.
    return type("FakeEnvDir", (), {"exists": lambda self: exists})


# ---------------------------------------------------------------------------
# AC01 — prerequisite check
# ---------------------------------------------------------------------------


class TestPrerequisiteCheck:
    """AC01: exit 1 with message if parallax install not done."""

    def test_exits_1_when_env_dir_missing(self, tmp_path):
        """Exits 1 and prints guidance when ~/.parallax/env is absent (AC01)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env_missing"  # does not exist

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path") as mock_script:
                mock_script.return_value = env_dir / "bin" / "parallax-mcp"
                result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 1
        assert "parallax install" in result.output

    def test_exits_1_when_script_missing(self, tmp_path):
        """Exits 1 when env dir exists but parallax-mcp script is absent (AC01)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        # Script does NOT exist inside

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path") as mock_script:
                mock_script.return_value = env_dir / "bin" / "parallax-mcp"
                result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 1
        assert "parallax install" in result.output

    def test_proceeds_when_env_and_script_exist(self, tmp_path):
        """Proceeds past AC01 check when env and script both exist."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC02 — platform-appropriate config path
# ---------------------------------------------------------------------------


class TestConfigPath:
    """AC02: correct Claude Desktop config path per platform."""

    def test_macos_path(self):
        """Returns ~/Library/Application Support/Claude/... on macOS (AC02)."""
        from cli.commands.mcp import _claude_config_path

        with patch("platform.system", return_value="Darwin"):
            path = _claude_config_path()

        assert "Library" in str(path)
        assert "Application Support" in str(path)
        assert "Claude" in str(path)
        assert path.name == "claude_desktop_config.json"

    def test_windows_path(self):
        """Returns %APPDATA%/Claude/... on Windows (AC02)."""
        from cli.commands.mcp import _claude_config_path

        with patch("platform.system", return_value="Windows"):
            with patch.dict("os.environ", {"APPDATA": "C:\\Users\\test\\AppData\\Roaming"}):
                path = _claude_config_path()

        assert "Claude" in str(path)
        assert path.name == "claude_desktop_config.json"

    def test_linux_path(self):
        """Returns ~/.config/claude/... on Linux (AC02)."""
        from cli.commands.mcp import _claude_config_path

        with patch("platform.system", return_value="Linux"):
            path = _claude_config_path()

        assert ".config" in str(path)
        assert "claude" in str(path)
        assert path.name == "claude_desktop_config.json"


# ---------------------------------------------------------------------------
# AC03 — adds/updates mcpServers.parallax-mcp
# ---------------------------------------------------------------------------


class TestMcpServersEntry:
    """AC03: adds parallax-mcp entry with absolute script path."""

    def test_sets_command_to_absolute_script_path(self, tmp_path):
        """mcpServers.parallax-mcp.command equals the absolute parallax-mcp path (AC03)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        config = json.loads(config_path.read_text())
        assert config["mcpServers"]["parallax-mcp"]["command"] == str(script)

    def test_updates_existing_entry(self, tmp_path):
        """Overwrites an outdated parallax-mcp command with the new path (AC03)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"
        # Pre-existing config with outdated path
        config_path.write_text(json.dumps({
            "mcpServers": {"parallax-mcp": {"command": "/old/path/parallax-mcp"}}
        }))

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        config = json.loads(config_path.read_text())
        assert config["mcpServers"]["parallax-mcp"]["command"] == str(script)


# ---------------------------------------------------------------------------
# AC04 — creates config file if absent
# ---------------------------------------------------------------------------


class TestConfigCreation:
    """AC04: creates claude_desktop_config.json when it does not exist."""

    def test_creates_config_file(self, tmp_path):
        """Config file is created with mcpServers key when absent (AC04)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "nested" / "claude_desktop_config.json"
        assert not config_path.exists()

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert "mcpServers" in config

    def test_created_config_has_only_mcp_servers(self, tmp_path):
        """Newly created config contains only the mcpServers key (AC04)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    runner.invoke(app, ["mcp", "install"])

        config = json.loads(config_path.read_text())
        assert set(config.keys()) == {"mcpServers"}


# ---------------------------------------------------------------------------
# AC05 — never overwrites other keys
# ---------------------------------------------------------------------------


class TestNoKeyOverwrite:
    """AC05: existing config keys are preserved."""

    def test_preserves_existing_top_level_keys(self, tmp_path):
        """Other top-level keys survive the merge (AC05)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"
        original_config = {
            "theme": "dark",
            "mcpServers": {
                "other-tool": {"command": "/usr/local/bin/other-tool"},
            },
        }
        config_path.write_text(json.dumps(original_config))

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        config = json.loads(config_path.read_text())
        assert config["theme"] == "dark"
        assert config["mcpServers"]["other-tool"]["command"] == "/usr/local/bin/other-tool"
        assert "parallax-mcp" in config["mcpServers"]


# ---------------------------------------------------------------------------
# AC06 — success message
# ---------------------------------------------------------------------------


class TestSuccessMessage:
    """AC06: prints success message on registration."""

    def test_prints_success_message(self, tmp_path):
        """Prints 'MCP server registered. Restart Claude Desktop to apply.' (AC06)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        assert "MCP server registered" in result.output
        assert "Restart Claude Desktop to apply" in result.output


# ---------------------------------------------------------------------------
# AC07 — already registered
# ---------------------------------------------------------------------------


class TestAlreadyRegistered:
    """AC07: re-running when already registered prints message and exits 0."""

    def test_already_registered_prints_message(self, tmp_path):
        """Prints 'Already registered.' when entry already matches (AC07)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"
        existing_config = {
            "mcpServers": {
                "parallax-mcp": {"command": str(script)},
            }
        }
        config_path.write_text(json.dumps(existing_config))

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
        assert "Already registered" in result.output

    def test_already_registered_does_not_modify_file(self, tmp_path):
        """Config file is not modified when already registered (AC07)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"
        existing_config = {
            "mcpServers": {
                "parallax-mcp": {"command": str(script)},
            }
        }
        original_text = json.dumps(existing_config)
        config_path.write_text(original_text)

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    runner.invoke(app, ["mcp", "install"])

        assert config_path.read_text() == original_text

    def test_already_registered_exits_0(self, tmp_path):
        """Exit code is 0 when already registered (AC07)."""
        runner = _runner()
        app = _app()

        env_dir = tmp_path / "env"
        bin_dir = env_dir / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "parallax-mcp"
        script.write_text("#!/bin/sh\n")

        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text(json.dumps({
            "mcpServers": {"parallax-mcp": {"command": str(script)}}
        }))

        with patch("cli.commands.mcp._ENV_DIR", env_dir):
            with patch("cli.commands.mcp._mcp_script_path", return_value=script):
                with patch("cli.commands.mcp._claude_config_path", return_value=config_path):
                    result = runner.invoke(app, ["mcp", "install"])

        assert result.exit_code == 0
