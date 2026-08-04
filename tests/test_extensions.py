"""Tests for the cothis extensions system (install + discover + CLI).

All uv/subprocess calls are mocked — hermetic, no network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---- sanitization ----


def test_sanitize_name_strips_version() -> None:
    from cothis.extensions import _sanitize_name

    assert _sanitize_name("rich>=13") == "rich"
    assert _sanitize_name("httpx[http2]==0.27") == "httpx"


def test_sanitize_name_rejects_traversal() -> None:
    from cothis.extensions import _sanitize_name

    with pytest.raises(ValueError):
        _sanitize_name("../x")
    with pytest.raises(ValueError):
        _sanitize_name("a/b")


def test_sanitize_name_rejects_non_pypi() -> None:
    from cothis.extensions import _sanitize_name

    with pytest.raises(ValueError):
        _sanitize_name("git+https://github.com/x/y")


# ---- install (mocked uv) ----


def test_install_creates_venv_and_manifest(tmp_path: Path) -> None:
    from cothis.extensions import ExtensionManager

    mgr = ExtensionManager(tmp_path)
    with patch("cothis.extensions.subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="rich 13.7.0\n", stderr=""),
        ]
        ext = mgr.install("rich>=13")
    assert ext.name == "rich"
    assert ext.version == "13.7.0"
    manifest = json.loads((tmp_path / "extensions" / "rich" / "extension.json").read_text())
    assert manifest["name"] == "rich"
    assert manifest["version"] == "13.7.0"
    assert manifest["cothis_extension_api"] == 1


def test_install_uv_missing(tmp_path: Path) -> None:
    from cothis.extensions import ExtensionError, ExtensionManager

    mgr = ExtensionManager(tmp_path)
    with patch("cothis.extensions.shutil.which", return_value=None):
        with pytest.raises(ExtensionError, match="uv not found"):
            mgr.install("rich")


# ---- discover ----


def test_discover_empty(tmp_path: Path) -> None:
    from cothis.extensions import ExtensionManager

    assert ExtensionManager(tmp_path).discover() == []


def test_discover_lists_installed(tmp_path: Path) -> None:
    from cothis.extensions import ExtensionManager

    ext_dir = tmp_path / "extensions" / "rich"
    ext_dir.mkdir(parents=True)
    (ext_dir / "extension.json").write_text(
        json.dumps(
            {
                "name": "rich",
                "spec": "rich",
                "version": "13.0",
                "venv_path": str(ext_dir / "venv"),
                "cothis_extension_api": 1,
            }
        )
    )
    exts = ExtensionManager(tmp_path).discover()
    assert len(exts) == 1
    assert exts[0].name == "rich"


# ---- CLI ----


def test_cli_install_invokes_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from cothis.cli import app

    mock_mgr = MagicMock()
    mock_ext = MagicMock()
    mock_ext.name = "rich"
    mock_ext.version = "13.0"
    mock_mgr.install.return_value = mock_ext
    monkeypatch.setattr("cothis.extensions.ExtensionManager", lambda *a, **kw: mock_mgr)
    runner = CliRunner()
    result = runner.invoke(app, ["install", "rich"])
    assert result.exit_code == 0
    assert "installed" in result.stdout.lower()
    mock_mgr.install.assert_called_once_with("rich")


def test_cli_extensions_list_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from cothis.cli import app

    mock_mgr = MagicMock()
    mock_mgr.discover.return_value = []
    monkeypatch.setattr("cothis.extensions.ExtensionManager", lambda *a, **kw: mock_mgr)
    runner = CliRunner()
    result = runner.invoke(app, ["extensions"])
    assert result.exit_code == 0
    assert "no extensions installed" in result.stdout
