"""Extensions system — install and discover cothis extensions.

Extensions are PyPI packages installed into a single shared uv venv under
``$COTHIS_HOME/extensions/venv/``. Loading extensions into the agent loop
(out-of-process) is a follow-up iteration; this module provides install +
discover now.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("cothis.extensions")


class ExtensionError(Exception):
    """An extension install/discovery failure (uv missing, install failed, …)."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _cothis_home() -> Path:
    """Resolve ``$COTHIS_HOME`` (default ``~/.cothis``). Mirrors cli.py's helper."""
    return Path(os.environ.get("COTHIS_HOME") or (Path.home() / ".cothis")).expanduser()


def _extensions_dir(cothis_home: Path | None = None) -> Path:
    return (cothis_home or _cothis_home()) / "extensions"


def _venv_path(cothis_home: Path | None = None) -> Path:
    return _extensions_dir(cothis_home) / "venv"


def _manifest_path(cothis_home: Path | None = None) -> Path:
    return _extensions_dir(cothis_home) / "extensions.json"


# ---------------------------------------------------------------------------
# Name sanitisation (PyPI-name extraction + path-traversal guard)
# ---------------------------------------------------------------------------

def _extract_name(spec: str) -> str:
    """Extract a display name from a PyPI spec or a git/HTTP URL.

    ``"rich>=13"`` -> ``"rich"``; ``"git+https://github.com/x/mypkg.git"``
    -> ``"mypkg"``. Used for the manifest + display only; the actual install
    delegates to ``uv pip install <spec>`` which handles all source types.
    """
    stripped = spec.strip()
    if stripped.startswith(("git+", "http://", "https://")) or ".git" in stripped:
        url = stripped.split("git+", 1)[-1]
        url = url.split("@", 1)[-1]
        url = url.split("?", 1)[0].split("#", 1)[0]
        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name.lower()
    for sep in ("[", "<", ">", "=", "!", ";", " @ "):
        stripped = stripped.split(sep, 1)[0]
    return stripped.strip().lower()


# ---------------------------------------------------------------------------
# uv discovery
# ---------------------------------------------------------------------------


def _find_uv() -> str:
    """Locate the ``uv`` binary on ``PATH`` or raise ``ExtensionError``."""
    uv = shutil.which("uv")
    if uv is None:
        raise ExtensionError(
            "uv not found on PATH; install uv first: https://docs.astral.sh/uv/"
        )
    return uv


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Extension:
    """A discovered/installed extension."""

    name: str
    spec: str
    version: str | None


# ---------------------------------------------------------------------------
# Manager (single shared venv)
# ---------------------------------------------------------------------------


class ExtensionManager:
    """Install and discover cothis extensions in a single shared uv venv.

    All extensions share one venv at ``$COTHIS_HOME/extensions/venv/``.
    ``cothis install rich httpx`` installs both into that venv in one
    ``uv pip install`` call; versions can be specified per-spec.
    """

    def __init__(self, cothis_home: Path | None = None) -> None:
        self._home = cothis_home or _cothis_home()

    # ------------------------------------------------------------------ install

    def install(self, specs: list[str]) -> list[Extension]:
        """Install *specs* into the shared extensions venv (idempotent venv).

        Returns one :class:`Extension` per spec (with read-back version).
        """
        uv = _find_uv()
        venv = _venv_path(self._home)
        venv.parent.mkdir(parents=True, exist_ok=True)

        # 1. Ensure the shared venv exists (uv venv is idempotent).
        self._run_uv(uv, ["venv", str(venv), "--python", "3.14"])
        # 2. Install all specs at once into the shared venv.
        self._run_uv(uv, ["pip", "install", *specs, "--python", str(venv)])
        # 3. Read back versions.
        results = [
            Extension(name=_extract_name(s), spec=s, version=self._read_version(uv, _extract_name(s), venv))
            for s in specs
        ]
        # 4. Update manifest.
        self._update_manifest(results)
        return results

    # --------------------------------------------------------------- discover

    def discover(self) -> list[Extension]:
        """List installed extensions (from the manifest)."""
        mpath = _manifest_path(self._home)
        if not mpath.is_file():
            return []
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
            return [
                Extension(
                    name=e["name"],
                    spec=e.get("spec", e["name"]),
                    version=e.get("version"),
                )
                for e in data.get("extensions", [])
            ]
        except Exception:
            logger.warning("skipping corrupt extensions manifest: %s", mpath)
            return []

    # --------------------------------------------------------------- internal

    def _run_uv(self, uv: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run ``uv <args>``; raise ``ExtensionError`` with stderr on failure."""
        proc = subprocess.run(
            [uv, *args], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            raise ExtensionError(
                f"uv {' '.join(args[:2])} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()[-300:]}"
            )
        return proc

    def _read_version(self, uv: str, name: str, venv: Path) -> str | None:
        """Read back the installed version of *name* from ``uv pip list``."""
        proc = self._run_uv(uv, ["pip", "list", "--python", str(venv)])
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == name:
                return parts[1]
        return None

    def _update_manifest(self, new_exts: list[Extension]) -> None:
        """Merge *new_exts* into the manifest (dedup by name)."""
        existing = {e.name: e for e in self.discover()}
        for ext in new_exts:
            existing[ext.name] = ext
        all_exts = sorted(existing.values(), key=lambda e: e.name)
        data = {
            "extensions": [
                {"name": e.name, "spec": e.spec, "version": e.version}
                for e in all_exts
            ],
            "venv_path": str(_venv_path(self._home)),
            "updated_at": datetime.now(UTC).isoformat(),
            "cothis_extension_api": 1,
        }
        mpath = _manifest_path(self._home)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(mpath, json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Future loading (stub — documented, NOT wired)
# ---------------------------------------------------------------------------


class ExtensionLoader:
    """Loads an installed extension's tools into the agent (future).

    Each extension's package may expose a console-script entry point that
    speaks MCP over stdio. cothis spawns it from the extensions venv and
    consumes it via the existing MCP-stdio tool source.
    """

    def __init__(self, ext: Extension) -> None:
        self._ext = ext

    def load(self) -> None:
        raise NotImplementedError(
            "extension loading into the agent loop is a follow-up."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, content: str) -> None:
    """Write *content* to *path* atomically (tmp + os.replace)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


__all__ = [
    "Extension",
    "ExtensionError",
    "ExtensionLoader",
    "ExtensionManager",
]
