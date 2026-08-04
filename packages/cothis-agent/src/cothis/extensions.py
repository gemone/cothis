"""Extensions system — install, discover, and (future) load per-extension isolated venvs.

Mirrors pi's discover→install→resolve flow on clean-room principles: each
extension is a PyPI package installed into its own uv-built venv under
``$COTHIS_HOME/extensions/<name>/``, tracked by an ``extension.json`` manifest.
Loading extensions into the agent loop (out-of-process via MCP-stdio) is a
follow-up iteration; this module provides install + discover now.

No pi TypeScript is copied; the architecture is reimplemented in idiomatic Python.
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

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


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


def _manifest_path(cothis_home: Path, name: str) -> Path:
    return _extensions_dir(cothis_home) / name / "extension.json"


# ---------------------------------------------------------------------------
# Name sanitisation (PyPI-name extraction + path-traversal guard)
# ---------------------------------------------------------------------------

_PYPI_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _sanitize_name(spec: str) -> str:
    """Extract the bare PyPI project name from a spec, rejecting non-PyPI sources.

    ``"rich>=13"``, ``"httpx[http2]==0.27"``, ``"package-name>=1.0"`` → the bare
    name.  Rejects git URLs, local paths, and anything with path separators
    (defence-in-depth against traversal out of ``extensions/``).
    """
    stripped = spec.strip()
    # Reject anything that looks like a path / URL / VCS spec (I8 = PyPI only).
    if "/" in stripped or "\\" in stripped or stripped.startswith(("git+", "http://", "https://")):
        raise ValueError(
            f"non-PyPI source {spec!r} not supported yet (git/local-path extensions are a follow-up)"
        )
    # Strip version specifiers, extras, etc.: take the part before the first
    # comparator/extras/semicolon.
    for sep in ("[", "<", ">", "=", "!", ";", " @ "):
        stripped = stripped.split(sep, 1)[0]
    name = stripped.strip().lower()
    if not _PYPI_NAME_RE.match(name):
        raise ValueError(f"invalid extension name {name!r} (must be a valid PyPI project name)")
    return name


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
    venv_path: Path
    manifest_path: Path


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ExtensionManager:
    """Install and discover cothis extensions (per-extension isolated uv venvs)."""

    def __init__(self, cothis_home: Path | None = None) -> None:
        self._home = cothis_home or _cothis_home()

    # ------------------------------------------------------------------ install

    def install(self, spec: str) -> Extension:
        """Install *spec* into a fresh/existing per-extension venv via ``uv``.

        Idempotent: re-running ``uv venv`` on an existing path is a no-op.
        Returns the :class:`Extension` (with the read-back version).
        """
        name = _sanitize_name(spec)
        ext_dir = _extensions_dir(self._home) / name
        venv_path = ext_dir / "venv"
        ext_dir.mkdir(parents=True, exist_ok=True)

        uv = _find_uv()

        # 1. Create (or refresh) the venv.
        self._run_uv(uv, ["venv", str(venv_path), "--python", "3.14"])
        # 2. Install the package into the venv.
        self._run_uv(uv, ["pip", "install", spec, "--python", str(venv_path)])
        # 3. Read back the installed version.
        version = self._read_version(uv, name, venv_path)

        # 4. Write manifest atomically.
        manifest = {
            "name": name,
            "spec": spec,
            "version": version,
            "installed_at": datetime.now(UTC).isoformat(),
            "venv_path": str(venv_path),
            "cothis_extension_api": 1,
        }
        mpath = ext_dir / "extension.json"
        _write_atomic(mpath, json.dumps(manifest, indent=2) + "\n")

        return Extension(
            name=name,
            spec=spec,
            version=version,
            venv_path=venv_path,
            manifest_path=mpath,
        )

    # --------------------------------------------------------------- discover

    def discover(self) -> list[Extension]:
        """List installed extensions (parsed from manifests; corrupt → skip)."""
        ext_root = _extensions_dir(self._home)
        if not ext_root.is_dir():
            return []
        results: list[Extension] = []
        for child in sorted(ext_root.iterdir()):
            mpath = child / "extension.json"
            if not mpath.is_file():
                continue
            try:
                data = json.loads(mpath.read_text(encoding="utf-8"))
                results.append(
                    Extension(
                        name=data["name"],
                        spec=data.get("spec", data["name"]),
                        version=data.get("version"),
                        venv_path=Path(data.get("venv_path", child / "venv")),
                        manifest_path=mpath,
                    )
                )
            except Exception:
                logger.warning("skipping corrupt extension manifest: %s", mpath)
        return results

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

    def _read_version(self, uv: str, name: str, venv_path: Path) -> str | None:
        """Read back the installed version of *name* from ``uv pip list``."""
        proc = self._run_uv(uv, ["pip", "list", "--python", str(venv_path)])
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == name:
                return parts[1]
        return None


# ---------------------------------------------------------------------------
# Future loading (I9+ stub — documented, NOT wired)
# ---------------------------------------------------------------------------


class ExtensionLoader:
    """Loads an installed extension's tools into the agent (I9+).

    Per-extension venvs forbid in-process import. The planned strategy: each
    extension's package may expose a console-script entry point (or
    ``python -m <pkg> --mcp-stdio``) that speaks MCP over stdio. cothis spawns
    ``<venv>/bin/<entrypoint>`` and consumes it via the existing MCP-stdio tool
    source (``cothis.tools.mcp``, type ``mcp.stdio``). No new IPC protocol.
    """

    def __init__(self, ext: Extension) -> None:
        self._ext = ext

    def load(self) -> None:
        raise NotImplementedError(
            "extension loading into the agent loop is a follow-up (I9+); "
            "this stub is intentionally not wired."
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
