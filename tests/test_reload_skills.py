"""Tests for ``/reload-skills`` slash command (#172).

Verifies the handler re-runs ``discover_skills`` and returns a
summary. Slash-command dispatch wiring itself is #67's; these
tests exercise the handler through the slash framework directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import cothis.slash as slash_mod
from cothis.skills import Skill, register_slash_commands
from cothis.slash import SlashContext, dispatch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Clear the module-level slash registry between tests."""
    slash_mod._entries.clear()


@pytest.fixture
def _registered() -> None:
    """Register the skills slash commands (idempotent on the registry)."""
    register_slash_commands()


@pytest.mark.asyncio
async def test_reload_returns_summary(
    tmp_path: Path, _registered: None,
) -> None:
    """``/reload-skills`` returns a non-None summary string."""
    ctx = SlashContext()
    result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None
    assert "skill" in result.lower()


@pytest.mark.asyncio
async def test_reload_lists_discovered_skills(
    tmp_path: Path, _registered: None,
) -> None:
    """Summary names discovered skills."""
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "alpha").mkdir()
    (skills_dir / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: a\n---\nbody\n", encoding="utf-8",
    )

    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )

    ctx = SlashContext(session=s)
    result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None
    assert "alpha" in result
    s.close()


@pytest.mark.asyncio
async def test_reload_reports_zero_when_no_skills(
    tmp_path: Path, _registered: None,
) -> None:
    """Empty discovery → summary mentions zero."""
    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    ctx = SlashContext(session=s)
    result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None
    # Look for a number; "0" or "zero" or "no skills".
    assert any(token in result.lower() for token in ("0", "no skill", "zero"))
    s.close()


@pytest.mark.asyncio
async def test_reload_picks_up_newly_installed_skill(
    tmp_path: Path, _registered: None,
) -> None:
    """End-to-end: a skill installed after session start appears in reload."""
    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )

    skills_dir = tmp_path / ".agents" / "skills"

    # First reload: empty.
    ctx = SlashContext(session=s)
    first = await dispatch("/reload-skills", ctx=ctx)
    assert first is not None
    assert "beta" not in first

    # Install a skill.
    beta = skills_dir / "beta"
    beta.mkdir(parents=True)
    beta.joinpath("SKILL.md").write_text(
        "---\nname: beta\ndescription: b\n---\nbody\n", encoding="utf-8",
    )

    # Second reload: now includes beta.
    second = await dispatch("/reload-skills", ctx=ctx)
    assert second is not None
    assert "beta" in second
    s.close()


@pytest.mark.asyncio
async def test_reload_works_without_session(
    tmp_path: Path, _registered: None,
) -> None:
    """No session attached → handler still returns a summary."""
    ctx = SlashContext(session=None)
    result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None


@pytest.mark.asyncio
async def test_reload_summary_mentions_count(
    tmp_path: Path, _registered: None,
) -> None:
    """Summary includes the discovered-skill count."""
    skills_dir = tmp_path / ".agents" / "skills"
    for name in ("a", "b", "c"):
        d = skills_dir / name
        d.mkdir(parents=True)
        d.joinpath("SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}-desc\n---\nbody\n",
            encoding="utf-8",
        )

    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    ctx = SlashContext(session=s)
    result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None
    assert "3" in result
    s.close()


@pytest.mark.asyncio
async def test_reload_logs_and_continues_on_discovery_error(
    tmp_path: Path, _registered: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """Discovery raises → handler returns an error summary, doesn't crash."""
    import logging as _logging

    def _boom(*a, **k):
        raise RuntimeError("disk on fire")

    with patch("cothis.skills.discover_skills", side_effect=_boom):
        with caplog.at_level(_logging.WARNING, logger="cothis.skills"):
            ctx = SlashContext()
            result = await dispatch("/reload-skills", ctx=ctx)
    assert result is not None
    # Surfaces failure to the user; doesn't silently return None.
    assert any(
        "disk on fire" in r.message or "fail" in r.message.lower()
        or "error" in r.message.lower()
        for r in caplog.records
    )


def test_register_slash_commands_is_idempotent(_registered: None) -> None:
    """Calling register twice doesn't error (registry just overwrites)."""
    # The autouse fixture already called it once; call again.
    register_slash_commands()
    assert "reload-skills" in slash_mod.names()


def test_register_does_not_run_at_import() -> None:
    """Importing ``cothis.skills`` must not auto-register slash commands.

    Side-effect-free import: cli.py (or whoever) explicitly calls
    ``register_slash_commands`` when wiring up the REPL.
    """
    # Re-import a fresh copy in isolation. The slash registry should
    # not contain reload-skills unless the test fixture registered it.
    import importlib
    slash_mod._entries.clear()
    import cothis.skills as _skills
    importlib.reload(_skills)
    assert "reload-skills" not in slash_mod.names()


@pytest.mark.asyncio
async def test_reload_handler_directly(
    tmp_path: Path, _registered: None,
) -> None:
    """Call the handler directly (bypassing dispatch) for unit coverage."""
    from cothis.skills import reload_skills_handler

    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "x").mkdir()
    (skills_dir / "x" / "SKILL.md").write_text(
        "---\nname: x\ndescription: x\n---\nbody\n", encoding="utf-8",
    )

    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    ctx = SlashContext(session=s)
    result = await reload_skills_handler(ctx, "")
    assert result is not None
    assert "x" in result
    s.close()
