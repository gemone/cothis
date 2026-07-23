"""Profile ``cothis --help`` startup cost via ``-X importtime``.

When ``COTHIS_PROFILE_STARTUP`` is set, ``cothis.cli`` calls
:func:`maybe_profile` *before* its third-party imports run. The helper
re-execs a subprocess with ``-X importtime``, parses the per-module
self-cost from stderr, and prints the top entries by self-time to
stderr. The original process exits.

Keeping this logic in a dedicated module (instead of inline in
``cli.py``) means the only cost on the cold path is one
``os.environ.get`` plus an import of this module (which itself imports
only stdlib). ``-X importtime`` is the source of truth for per-import
cost; we just make it digestible.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

_IMPORTTIME_LINE = re.compile(
    r"^import time:\s*(?P<self_us>\d+)\s*\|\s*(?P<cum_us>\d+)\s*\|\s*(?P<module>.+)$"
)


def maybe_profile() -> None:
    """If ``COTHIS_PROFILE_STARTUP`` is set, run the profiler and exit."""
    if not os.environ.get("COTHIS_PROFILE_STARTUP"):
        return
    rows = _measure()
    _print_top(rows, top_n=25)
    sys.exit(0)


def _measure() -> list[tuple[int, int, str]]:
    """Spawn a subprocess with ``-X importtime``; parse per-module rows."""
    # Clear COTHIS_PROFILE_STARTUP in the subprocess so it doesn't
    # re-trigger maybe_profile and recurse.
    env = {k: v for k, v in os.environ.items() if k != "COTHIS_PROFILE_STARTUP"}
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import cothis.cli"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    rows: list[tuple[int, int, str]] = []
    for line in result.stderr.splitlines():
        m = _IMPORTTIME_LINE.match(line)
        if not m:
            continue
        self_us = int(m.group("self_us"))
        cum_us = int(m.group("cum_us"))
        module = m.group("module").strip()
        rows.append((self_us, cum_us, module))
    return rows


def _print_top(rows: list[tuple[int, int, str]], top_n: int = 25) -> None:
    """Print the top-N modules by self-cost to stderr."""
    by_self = sorted(rows, key=lambda r: r[0], reverse=True)
    total_self = sum(r[0] for r in rows)
    print(
        f"cothis startup profile — {len(rows)} imports, "
        f"{total_self / 1000:.1f}ms total self-time\n",
        file=sys.stderr,
    )
    print(f"{'self_ms':>8}  {'cum_ms':>8}  module", file=sys.stderr)
    for self_us, cum_us, module in by_self[:top_n]:
        print(
            f"{self_us / 1000:>8.2f}  {cum_us / 1000:>8.2f}  {module}",
            file=sys.stderr,
        )


def _format_for_test(rows: list[tuple[int, int, str]]) -> str:
    """Test helper: format the top-N as a string instead of stderr."""
    buf: list[str] = []
    total_self = sum(r[0] for r in rows)
    buf.append(
        f"cothis startup profile — {len(rows)} imports, "
        f"{total_self / 1000:.1f}ms total self-time"
    )
    for self_us, _cum_us, module in sorted(rows, key=lambda r: r[0], reverse=True)[:5]:
        buf.append(f"{self_us / 1000:.2f}ms  {module}")
    return "\n".join(buf)


def _main(argv: Any = None) -> None:
    """CLI entry: ``python -m cothis._profile_startup``."""
    rows = _measure()
    _print_top(rows)


if __name__ == "__main__":
    _main()
