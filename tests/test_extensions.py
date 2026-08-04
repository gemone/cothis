"""Tests for the cothis extensions system (install + discover + CLI).

All uv/subprocess calls are mocked — hermetic, no network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_extract_name_pypi() -> None:
    from cothis.extensions import _extract_name

    assert _extract_name("rich>=13") == "rich"
    assert _extract_name("httpx[http2]==0.27") == "httpx"


def test_extract_name_git_url() -> None:
    from cothis.extensions import _extract_name

    assert _extract_name("git+https://github.com/x/mypkg.git") == "mypkg"
    assert _extract_name("https://github.com/x/cool-pkg.git") == "cool-pkg"


def test_install_into_shared_venv(tmp_path):
    from cothis.extensions import ExtensionManager

    mgr = ExtensionManager(tmp_path)
    with patch("cothis.extensions.subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="rich 13.7.0\n", stderr=""),
            MagicMock(returncode=0, stdout="httpx 0.27.0\n", stderr=""),
        ]
        exts = mgr.install(["rich", "httpx"])
    assert len(exts) == 2
    assert exts[0].name == "rich"
    assert exts[0].version == "13.7.0"
    manifest = json.loads((tmp_path / "extensions" / "extensions.json").read_text())
    assert len(manifest["extensions"]) == 2


def test_install_uv_missing(tmp_path):
    from cothis.extensions import ExtensionError, ExtensionManager

    mgr = ExtensionManager(tmp_path)
    with patch("cothis.extensions.shutil.which", return_value=None):
        with pytest.raises(ExtensionError, match="uv not found"):
            mgr.install(["rich"])


def test_discover_empty(tmp_path):
    from cothis.extensions import ExtensionManager

    assert ExtensionManager(tmp_path).discover() == []


def test_discover_from_manifest(tmp_path):
    from cothis.extensions import ExtensionManager

    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(parents=True)
    (ext_dir / "extensions.json").write_text(
        json.dumps({"extensions": [{"name": "rich", "spec": "rich", "version": "13.0"}], "cothis_extension_api": 1})
    )
    exts = ExtensionManager(tmp_path).discover()
    assert len(exts) == 1
    assert exts[0].name == "rich"


def test_cli_install_invokes_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from cothis.cli import app

    mock_mgr = MagicMock()
    mock_ext = MagicMock()
    mock_ext.name = "rich"
    mock_ext.version = "13.0"
    mock_mgr.install.return_value = [mock_ext]
    monkeypatch.setattr("cothis.extensions.ExtensionManager", lambda *a, **kw: mock_mgr)
    runner = CliRunner()
    result = runner.invoke(app, ["install", "rich"])
    assert result.exit_code == 0
    assert "installed" in result.stdout.lower()
    mock_mgr.install.assert_called_once_with(["rich"])


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
