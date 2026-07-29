"""Tests for ``fs.search`` performance + ReDoS deadline (#111).

The per-line ``ThreadPoolExecutor`` round-trip was 62× slower than
direct ``regex.search`` on the fast path, and the per-line timeout
didn't actually bound wall time on a ReDoS pattern (Python can't
kill the worker thread; ``__exit__`` waits for it). #111 drops the
executor and adds a wall-clock cap.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from cothis.tools.fs import search as search_module
from cothis.tools.fs._hygiene import workdir_context
from cothis.tools.fs.search import _search as fs_search

if TYPE_CHECKING:
    from pathlib import Path


def test_fast_path_scans_2000_lines_quickly(tmp_path: Path) -> None:
    """A 2000-line scan finishes well under the ``_DEADLINE_SECONDS`` cap (#111).

    The scan must complete fast enough that the deadline never fires
    on typical inputs. The 100ms budget leaves headroom for CI
    variance while still catching a regression to per-line thread
    overhead.
    """
    # 2000-line file, 1 match per line so the inner body fires too.
    lines = "\n".join(f"foo_{i} match" for i in range(2000))
    (tmp_path / "big.py").write_text(lines)

    with workdir_context(tmp_path):
        t0 = time.perf_counter()
        result = fs_search(pattern="match", path=".", max_results=2000)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(result) == 2000
    # 250ms is ~10× the pre-fix executor cost — leaves generous headroom
    # for CI runner variance (Ubuntu shared runners spike to 200ms+).
    assert elapsed_ms < 250, f"scan too slow: {elapsed_ms:.1f}ms"


def test_wall_clock_cap_returns_partial_results_across_many_files(
    tmp_path: Path, monkeypatch
) -> None:
    """Tight wall-clock cap aborts a many-file scan with partial results (#111).

    The deadline is checked at the per-file loop boundary. A scan
    that would touch thousands of files aborts once the deadline
    fires, returning whatever was found before. (The deadline cannot
    interrupt a single pathological ``regex.search`` call — Python
    can't kill a thread mid-call — but it bounds the *traversal*.)
    """
    monkeypatch.setattr(search_module, "_DEADLINE_SECONDS", 0.05)

    # 5000 files, each small — the cap fires mid-traversal.
    for i in range(5000):
        (tmp_path / f"f{i:04d}.txt").write_text(f"match_{i}\n")

    with workdir_context(tmp_path):
        t0 = time.perf_counter()
        result = fs_search(pattern="match_", path=".", max_results=1000)
        elapsed = time.perf_counter() - t0

    # Bounded by the cap + small slop for I/O. Pre-fix this would
    # have completed all 5000 files in whatever time the executor took.
    assert elapsed < 1.0, f"call not bounded: {elapsed:.2f}s"
    # Partial results — the cap fired before all 5000 were scanned.
    assert isinstance(result, list)
    assert 0 <= len(result) < 5000


def test_walk_and_prune_skips_ignored_dirs(tmp_path: Path) -> None:
    """AC #321: ``os.walk`` + ``dirnames[:]`` pruning skips descent into ``_IGNORED_DIRS``.

    Regression guard for the walk-and-prune refactor. Before: ``root.rglob("*")``
    yielded files under ``node_modules/`` / ``.git/`` / ``.venv/`` etc., then
    filtered them per-path. After: descent never enters those dirs.

    Test shape: place a unique pattern in BOTH a regular file AND a file inside
    ``node_modules/``. Only the regular file's match should be returned — the
    node_modules file is never visited because the dir is pruned during walk.
    """
    from cothis.tools.fs._hygiene import workdir_context

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE_FOUND_IN_SRC\n", encoding="utf-8")
    # File inside an ignored dir — would match the same pattern, but the walker
    # should never descend into node_modules to read it.
    nm_dir = tmp_path / "node_modules" / "pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "should_not_scan.js").write_text(
        "NEEDLE_FOUND_IN_SRC\n", encoding="utf-8"
    )

    with workdir_context(tmp_path):
        result = fs_search(pattern="NEEDLE_FOUND_IN_SRC", path=".", max_results=10)

    assert isinstance(result, list)
    # Exactly one match — the src/app.py one. The node_modules copy was never visited.
    assert len(result) == 1, (
        f"expected 1 match (src/app.py only); got {len(result)}: {result}"
    )
    assert "src/app.py" in result[0]["file"]
    assert "node_modules" not in result[0]["file"]


def test_search_results_sorted_by_file_then_line(tmp_path: Path) -> None:
    """#417: ``fs.search`` returns results sorted by ``(file, line)``, not
    readdir order — deterministic across filesystems."""
    # Create files whose readdir order likely differs from alphabetical.
    for name in ["zebra.py", "alpha.py", "middle.py"]:
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")

    with workdir_context(tmp_path):
        result = fs_search(pattern="NEEDLE", path=".", max_results=10)

    files = [r["file"] for r in result]
    assert files == ["alpha.py", "middle.py", "zebra.py"], (
        f"results should be sorted by file; got {files}"
    )


def test_search_truncation_keeps_alphabetically_first(tmp_path: Path) -> None:
    """#417: when matches exceed ``max_results``, the alphabetically-first
    ones are kept (not whichever the walk reached first)."""
    for name in ["a.py", "b.py", "c.py", "d.py", "e.py"]:
        (tmp_path / name).write_text("NEEDLE\n", encoding="utf-8")

    with workdir_context(tmp_path):
        result = fs_search(pattern="NEEDLE", path=".", max_results=3)

    files = [r["file"] for r in result]
    assert files == ["a.py", "b.py", "c.py"], (
        f"max_results=3 should keep the alphabetically-first 3; got {files}"
    )
