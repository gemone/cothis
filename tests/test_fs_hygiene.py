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
    """Absolute paths defeat the cwd boundary; rejected with PathBoundaryError."""
    from cothis.tools.fs._hygiene import PathBoundaryError

    with pytest.raises(PathBoundaryError, match="absolute"):
        _resolve_under("/etc/passwd", tmp_path)


def test_resolve_under_parent_traversal_rejected(tmp_path: Path) -> None:
    """``..`` that escapes cwd is rejected after resolve."""
    from cothis.tools.fs._hygiene import PathBoundaryError

    with pytest.raises(PathBoundaryError, match="cwd|outside"):
        _resolve_under("../../etc/passwd", tmp_path)


def test_resolve_under_traversal_into_sibling_subdir_rejected(tmp_path: Path) -> None:
    """``../sibling`` resolves outside cwd; rejected."""
    from cothis.tools.fs._hygiene import PathBoundaryError

    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(PathBoundaryError):
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
    token = WORKDIR.set(None)
    try:
        assert workdir_path() is None
    finally:
        WORKDIR.reset(token)


def test_workdir_set_get_reset_round_trip(tmp_path: Path) -> None:
    """``set`` stores, ``get`` reads, ``reset(token)`` restores the prior
    value — the Agent's try/finally contract depends on this. Covers both
    the single-level case and nested-set restoration."""
    sentinel_a = Path("/tmp/a")
    sentinel_b = Path("/tmp/b")

    # Single level.
    token = WORKDIR.set(tmp_path)
    try:
        assert workdir_path() == tmp_path
    finally:
        WORKDIR.reset(token)

    # Nested: inner set/reset restores outer's value.
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
    """Agent's Pydantic model declares a ``cwd`` field (Path | None).

    Avoids Agent construction (which triggers any-llm's API-key check);
    field introspection is enough to pin the contract.
    """
    from cothis.agent import Agent

    assert "cwd" in Agent.model_fields
    assert Agent.model_fields["cwd"].default is None


def test_agent_run_body_wraps_workdir_context() -> None:
    """Agent.run delegates to ``_run_inner`` inside ``workdir_context``.

    Verified by source inspection — Agent.run is a thin wrapper, not a
    duplicate of the loop. (Construction-time check would need any-llm
    mocked; this stays offline.)
    """
    import inspect

    from cothis.agent import Agent

    src = inspect.getsource(Agent.run)
    assert "workdir_context(self.cwd)" in src
    assert "self._run_inner(user_input)" in src


def test_agent_run_stream_body_wraps_workdir_context() -> None:
    """Agent.run_stream delegates to ``_run_stream_inner`` inside
    ``workdir_context``."""
    import inspect

    from cothis.agent import Agent

    src = inspect.getsource(Agent.run_stream)
    assert "workdir_context(self.cwd)" in src
    assert "_run_stream_inner" in src


@pytest.mark.asyncio
async def test_workdir_injection_through_probe_tool(tmp_path: Path) -> None:
    """The injection chain Agent → workdir_context → WORKDIR → tool works.

    Drives the same path Agent.run takes (workdir_context wrap + a tool
    that reads WORKDIR) without constructing a real Agent (which would
    require an any-llm API key).
    """
    from cothis.tools.fs._hygiene import _cwd_probe, workdir_context, workdir_path

    assert workdir_path() is None

    with workdir_context(tmp_path):
        result = _cwd_probe()
        assert result == str(tmp_path)
        assert workdir_path() == tmp_path

    assert workdir_path() is None
