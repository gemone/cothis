"""Tests for ``cothis.tools.format.format_tool_output`` Unicode handling (#108).

The JSON path previously used ``json.dumps(result)`` with the default
``ensure_ascii=True``, escaping every non-ASCII codepoint to
``\\uXXXX``. The YAML path already used ``allow_unicode=True``. #108
brings the JSON path in line + applies the same fix to the CSV/TSV
cell encoder for nested non-string values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cothis.tools.format import format_tool_output

if TYPE_CHECKING:
    import pytest


def test_json_path_keeps_cjk_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default JSON path emits native CJK instead of ``\\uXXXX`` escapes."""
    monkeypatch.delenv("COTHIS_TOOL_OUTPUT_FORMAT", raising=False)
    result = {"file": "笔记.md", "preview": "你好，世界！"}
    out = format_tool_output(result)
    assert "笔记" in out
    assert "你好，世界！" in out
    # No \u escapes for CJK codepoints.
    assert "\\u" not in out


def test_json_path_keeps_emoji_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emoji survive the JSON path without ``\\u`` escapes."""
    monkeypatch.delenv("COTHIS_TOOL_OUTPUT_FORMAT", raising=False)
    out = format_tool_output({"reaction": "👋🎉"})
    assert "👋🎉" in out
    assert "\\u" not in out


def test_json_path_ascii_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASCII content is unaffected by ``ensure_ascii=False``."""
    monkeypatch.delenv("COTHIS_TOOL_OUTPUT_FORMAT", raising=False)
    out = format_tool_output({"path": "/tmp/file.txt", "bytes": 42})
    assert out == '{"path": "/tmp/file.txt", "bytes": 42}'


def test_csv_path_keeps_cjk_in_nested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV cells with nested non-string values keep CJK native.

    The CSV cell encoder (``format.py:72``) uses ``json.dumps`` for
    nested values; pre-fix, those escaped CJK to ``\\uXXXX``.
    """
    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "csv")
    out = format_tool_output(
        [{"name": "笔记", "meta": {"tag": "中文"}}]
    )
    assert "笔记" in out
    assert "中文" in out
    assert "\\u" not in out


def test_yaml_path_still_keeps_cjk_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML path was already correct; regression-guard it stays that way."""
    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "yaml")
    out: Any = format_tool_output({"file": "笔记.md"})
    assert "笔记" in out


def test_format_module_import_does_not_load_yaml() -> None:
    """AC #279: ``import cothis.tools.format`` must not import ``yaml``.

    ``yaml`` is ~18ms cold. The default json/csv/tsv paths never touch
    it, so importing it eagerly at module top taxes every cold ``cothis``
    invocation. The lazy import inside ``format_tool_output``'s YAML
    branch means a fresh process that just imports the module pays $0.

    Verified via subprocess so the test starts from a clean module
    cache (the parent test process has already imported yaml via other
    tests; checking ``sys.modules`` here would always show it).
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, cothis.tools.format; "
            "print('yaml' in sys.modules)",
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False", (
        f"import cothis.tools.format should NOT import yaml; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
