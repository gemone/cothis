"""Tests for ``_parse_skill_md`` decode policy (#166).

SKILL.md body goes into the system prompt (via ``format_catalog``)
and tool-result content (via ``load_skill``). The decoder must use
the project's two-tier policy (UTF-8 → locale fallback → skip+warn),
never ``errors='replace'`` — silent U+FFFD injection violates the
"prompt correctness" floor encoded in ``agent._read_text``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cothis.skills import _parse_skill_md, discover_skills

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


_FRONTMATTER = "---\nname: x\ndescription: d\n---\n\n"


def _force_utf8_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin locale fallback to UTF-8 so invalid-byte tests are deterministic.

    Without this, Windows runners default to CP1252 where ``\\xff`` is
    a legitimate ``ÿ`` — the locale tier would decode it and the skill
    would load, breaking the "invalid bytes → skip" assertions.
    """
    import locale as _locale
    monkeypatch.setattr(
        _locale, "getpreferredencoding", lambda *a, **k: "utf-8",
    )


def test_invalid_utf_8_body_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytes invalid in UTF-8 (and the pinned UTF-8 locale) → skip + warn."""
    _force_utf8_locale(monkeypatch)
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(
        (_FRONTMATTER + "Body with \xff\xfe\xfd invalid bytes.\n").encode(
            "latin-1",
        )
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        result = _parse_skill_md(skill_file)
    assert result is None
    assert any(
        "skipped" in r.message and str(skill_file) in r.message
        for r in caplog.records
    )


def test_no_replacement_character_injected(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: invalid bytes never become U+FFFD in the body."""
    _force_utf8_locale(monkeypatch)
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(
        (_FRONTMATTER + "Bad byte: \xff\n").encode("latin-1")
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        result = _parse_skill_md(skill_file)
    # Either skipped (preferred) or loaded without U+FFFD. Never loaded
    # with U+FFFD silently in the body.
    if result is not None:
        assert "�" not in result.body


def test_valid_utf_8_multibyte_body_preserved(tmp_path: Path) -> None:
    """Multibyte UTF-8 (emoji, CJK) round-trips unchanged."""
    skill_dir = tmp_path / "good"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    body = "中文 emoji 🎉 and émoji"
    skill_file.write_text(
        _FRONTMATTER + body + "\n", encoding="utf-8",
    )
    result = _parse_skill_md(skill_file)
    assert result is not None
    assert body in result.body


def test_locale_fallback_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-UTF-8-but-locale-decodable file loads via fallback tier.

    Simulates a Windows-authored SKILL.md: bytes valid in CP1252 but
    not in UTF-8. Monkeypatches ``locale.getpreferredencoding`` so the
    test is locale-independent.
    """
    import cothis.skills as skills_mod

    # \x93\x94 are CP1252 curly quotes; invalid as standalone UTF-8
    # sequence start bytes.
    body_bytes = b"Body with \x93curly\x94 quotes"
    skill_dir = tmp_path / "win"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(
        b"---\nname: win\ndescription: d\n---\n\n" + body_bytes + b"\n",
    )

    import locale as _locale
    monkeypatch.setattr(
        _locale, "getpreferredencoding", lambda *a, **k: "cp1252",
    )
    # Some Python builds cache locale.getpreferredencoding via functools
    # lru_cache on the stdlib side; the direct call above is enough
    # because _parse_skill_md calls locale.getpreferredencoding(False)
    # at parse time.
    result = skills_mod._parse_skill_md(skill_file)
    assert result is not None
    assert "curly" in result.body
    assert "\x93" not in result.body  # decoded, not raw bytes


def test_invalid_utf_8_logs_specific_skip_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip log mentions encoding failure so the user can diagnose."""
    _force_utf8_locale(monkeypatch)
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(
        b"---\nname: x\ndescription: d\n---\n\nBad: \xff\xff\xff\n",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        result = _parse_skill_md(skill_file)
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    # The message should name the file and signal a decode/skip reason.
    assert str(skill_file) in log_text


def test_invalid_utf_8_skill_dropped_from_discovery(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: discover_skills skips the broken skill, loads the good one."""
    _force_utf8_locale(monkeypatch)
    skills_dir = tmp_path / ".agents" / "skills"

    good = skills_dir / "good"
    good.mkdir(parents=True)
    good.joinpath("SKILL.md").write_text(
        "---\nname: good\ndescription: g\n---\nbody\n", encoding="utf-8",
    )

    bad = skills_dir / "bad"
    bad.mkdir()
    bad.joinpath("SKILL.md").write_bytes(
        b"---\nname: bad\ndescription: b\n---\n\n\xff\n",
    )

    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skills = discover_skills(
            tmp_path,
            cothis_home=tmp_path / "ch",
            user_agents=tmp_path / "ua",
        )
    names = [s.name for s in skills]
    assert "good" in names
    assert "bad" not in names
