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
