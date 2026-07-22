"""Tests for ``load_skill`` tool + active_skills session state (#158)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cothis.skills import _SKILL_FILE, load_skill

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING:
    from cothis.session import Session


class _FakeSession:
    """Minimal Session stand-in for load_skill tests."""

    def __init__(self) -> None:
        self._active: set[str] = set()

    def is_skill_active(self, name: str) -> bool:
        return name in self._active

    def _activate_skill(self, name: str) -> bool:
        if name in self._active:
            return False
        self._active.add(name)
        return True


def _make_skill_on_disk(
    skills_dir: Path, name: str, *, body: str = "Instructions.", resources: list[str] | None = None,
) -> None:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    d.joinpath(_SKILL_FILE).write_text(
        f"---\nname: {name}\ndescription: Test skill.\n---\n{body}\n",
        encoding="utf-8",
    )
    for r in resources or []:
        (d / r).parent.mkdir(parents=True, exist_ok=True)
        (d / r).write_text("resource", encoding="utf-8")


def test_load_skill_returns_wrapped_body(tmp_path: Path, monkeypatch: Any) -> None:
    """load_skill returns body wrapped in <skill_content> tags."""
    _make_skill_on_disk(tmp_path / ".agents" / "skills", "deploy", body="Deploy steps.")
    monkeypatch.chdir(tmp_path)

    session = _FakeSession()
    result = load_skill(name="deploy", _session=session)

    assert "<skill_content" in result
    assert "Deploy steps." in result
    assert "</skill_content>" in result


def test_load_skill_includes_resources(tmp_path: Path, monkeypatch: Any) -> None:
    """load_skill lists resource files in <skill_resources>."""
    _make_skill_on_disk(
        tmp_path / ".agents" / "skills", "guide",
        body="Guide body.",
        resources=["refs/arch.md", "refs/deploy.md"],
    )
    monkeypatch.chdir(tmp_path)

    result = load_skill(name="guide", _session=_FakeSession())

    assert "<skill_resources>" in result
    assert "refs/arch.md" in result
    assert "refs/deploy.md" in result


def test_load_skill_unknown_returns_error(tmp_path: Path, monkeypatch: Any) -> None:
    """Unknown skill name returns an error string."""
    _make_skill_on_disk(tmp_path / ".agents" / "skills", "exists")
    monkeypatch.chdir(tmp_path)

    result = load_skill(name="nope", _session=_FakeSession())
    assert "unknown skill" in result.lower()
    assert "nope" in result


def test_repeated_load_returns_already_active(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Loading an already-active skill returns a notice, not the body."""
    _make_skill_on_disk(tmp_path / ".agents" / "skills", "deploy", body="Steps.")
    monkeypatch.chdir(tmp_path)

    session = _FakeSession()
    first = load_skill(name="deploy", _session=session)
    assert "<skill_content" in first

    second = load_skill(name="deploy", _session=session)
    assert "already active" in second.lower()
    assert "<skill_content" not in second


def test_load_skill_activates_in_session(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """load_skill adds the skill name to the session's active set."""
    _make_skill_on_disk(tmp_path / ".agents" / "skills", "deploy")
    monkeypatch.chdir(tmp_path)

    session = _FakeSession()
    assert not session.is_skill_active("deploy")
    load_skill(name="deploy", _session=session)
    assert session.is_skill_active("deploy")


def test_load_skill_with_none_session(tmp_path: Path, monkeypatch: Any) -> None:
    """load_skill works without a session (no activation tracking)."""
    _make_skill_on_disk(tmp_path / ".agents" / "skills", "deploy", body="Body.")
    monkeypatch.chdir(tmp_path)

    result = load_skill(name="deploy", _session=None)
    assert "<skill_content" in result
