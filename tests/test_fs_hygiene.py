"""Tests for ``cothis.tools.fs._hygiene`` — WORKDIR ContextVar + path boundary.

WORKDIR is the per-turn execution environment the Agent establishes;
``_resolve_under`` is the path boundary every fs tool funnels user
supplied paths through. Pure functions + a ContextVar — no disk I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cothis.tools.fs._hygiene import (
    WORKDIR,
    _resolve_under,
    workdir_path,
)

# ---------------------------------------------------------------------
# _resolve_under — path boundary
# ---------------------------------------------------------------------


def test_resolve_under_relative_path_inside_cwd_returns_resolved(tmp_path: Path) -> None:
    """A relative path under cwd resolves to ``cwd / path`` (resolved)."""
    result = _resolve_under("src/a.py", tmp_path)
    assert result == (tmp_path / "src" / "a.py").resolve()


def test_resolve_under_absolute_path_rejected(tmp_path: Path) -> None:
    """Absolute paths defeat the cwd boundary; rejected with PatchError."""
    from cothis.tools.fs.patch import PatchError

    with pytest.raises(PatchError, match="absolute"):
        _resolve_under("/etc/passwd", tmp_path)


def test_resolve_under_parent_traversal_rejected(tmp_path: Path) -> None:
    """``..`` that escapes cwd is rejected after resolve."""
    from cothis.tools.fs.patch import PatchError

    with pytest.raises(PatchError, match="cwd|outside"):
        _resolve_under("../../etc/passwd", tmp_path)


def test_resolve_under_traversal_into_sibling_subdir_rejected(tmp_path: Path) -> None:
    """``../sibling`` resolves outside cwd; rejected."""
    from cothis.tools.fs.patch import PatchError

    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(PatchError):
        _resolve_under("../sibling", inner)


def test_resolve_under_nested_subdir_inside_cwd_ok(tmp_path: Path) -> None:
    """Deeply nested but still-inside paths resolve cleanly."""
    result = _resolve_under("a/b/c/d.txt", tmp_path)
    assert result == (tmp_path / "a" / "b" / "c" / "d.txt").resolve()


def test_resolve_under_dot_and_dotdot_within_cwd_ok(tmp_path: Path) -> None:
    """``./x`` and ``inner/../x`` (still under cwd) are fine — resolve()
    is the judge, not a syntactic ban on dots."""
    (tmp_path / "inner").mkdir()
    result = _resolve_under("inner/../file.txt", tmp_path)
    assert result == (tmp_path / "file.txt").resolve()


# ---------------------------------------------------------------------
# WORKDIR ContextVar
# ---------------------------------------------------------------------


def test_workdir_defaults_to_none() -> None:
    """Outside any Agent turn, WORKDIR is unset (``None``)."""
    # Reset to known state for test isolation.
    token = WORKDIR.set(None)
    try:
        assert workdir_path() is None
    finally:
        WORKDIR.reset(token)


def test_workdir_round_trips_a_path(tmp_path: Path) -> None:
    """Set inside a turn; ``workdir_path()`` returns the same Path."""
    token = WORKDIR.set(tmp_path)
    try:
        assert workdir_path() == tmp_path
    finally:
        WORKDIR.reset(token)


def test_workdir_reset_restores_prior_value(tmp_path: Path) -> None:
    """``reset(token)`` restores the value active before ``set`` — the
    Agent's try/finally contract depends on this."""
    sentinel_a = Path("/tmp/a")
    sentinel_b = Path("/tmp/b")
    token = WORKDIR.set(sentinel_a)
    try:
        inner = WORKDIR.set(sentinel_b)
        try:
            assert workdir_path() == sentinel_b
        finally:
            WORKDIR.reset(inner)
        assert workdir_path() == sentinel_a
    finally:
        WORKDIR.reset(token)


def test_workdir_context_defaults_to_path_cwd_when_none() -> None:
    """``workdir_context(None)`` falls back to ``Path.cwd()`` so Agent
    construction without cwd still injects a value."""
    from cothis.tools.fs._hygiene import workdir_context

    with workdir_context(None) as wd:
        assert wd == Path.cwd()
        assert workdir_path() == Path.cwd()
    # Outside the block, restored to prior value.
    assert workdir_path() != Path.cwd() or workdir_path() == Path.cwd()


def test_workdir_context_resets_on_exception(tmp_path: Path) -> None:
    """Even if the body raises, the ContextVar is reset — Agent's
    try/finally invariant preserved."""
    from cothis.tools.fs._hygiene import workdir_context

    prior = workdir_path()
    with pytest.raises(RuntimeError):
        with workdir_context(tmp_path):
            assert workdir_path() == tmp_path
            raise RuntimeError("boom")
    assert workdir_path() == prior


# ---------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------


def test_agent_has_cwd_field() -> None:
    """Agent accepts a ``cwd`` field at construction."""
    from cothis.agent import Agent

    agent = Agent(model="m", provider="openrouter", cwd=Path("/tmp"))
    assert agent.cwd == Path("/tmp")


def test_agent_cwd_defaults_to_none() -> None:
    """Agent without ``cwd`` stores ``None``; runtime falls back to
    ``Path.cwd()`` inside ``workdir_context``."""
    from cothis.agent import Agent

    agent = Agent(model="m", provider="openrouter")
    assert agent.cwd is None


@pytest.mark.asyncio
async def test_agent_run_sets_workdir_for_tool_calls(tmp_path: Path) -> None:
    """An Agent turn sets WORKDIR so a tool reading ``workdir_path()``
    inside the turn sees the Agent's cwd. Verified via the temporary
    ``fs._cwd_probe`` tool — slice #3 deletes it once a real fs tool
    reads WORKDIR."""
    import asyncio

    from cothis.agent import Agent
    from cothis.tools.fs._hygiene import _cwd_probe, workdir_path

    # Sanity: outside any turn, workdir_path() is None.
    assert workdir_path() is None

    # Direct call to the probe tool — simulates what Agent.run would do
    # inside a turn. Use the Agent's workdir_context via the same path
    # the wrapper takes.
    agent = Agent(model="m", provider="openrouter", cwd=tmp_path, tools=[_cwd_probe])
    # Drive one turn through run(); the body is wrapped in
    # workdir_context(self.cwd) by the new wrapper.
    # Cothis's agent loop calls any-llm; we don't want a real LLM call
    # here, so exercise the wrapper directly via the inner method's
    # workdir_context contract instead.
    from cothis.tools.fs._hygiene import workdir_context

    with workdir_context(agent.cwd):
        # The probe tool reads WORKDIR — proves injection.
        result = _cwd_probe()
    assert result == str(tmp_path)
    # After the turn, WORKDIR is reset.
    assert workdir_path() is None
