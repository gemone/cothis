"""Real-PTY regression test for TUI keyboard input (#375).

The in-process Textual pilot (``run_test`` + ``pilot.press``) parses keys
inside the process — it never exercises the terminal byte-parsing path
where the old ``InputBar(Container)`` wrapper dropped every keystroke
under a real driver. That is exactly why the ``test_tui.py`` suite stayed
green while the TUI accepted no input in a real terminal.

This test drives the actual TUI over a pseudo-terminal: it sends byte
keystrokes through a PTY and asserts the typed characters land in the
``TextArea`` document (i.e. they are painted back to the terminal). A
regression that re-wraps the ``TextArea`` in a ``Container`` (or otherwise
breaks key delivery under the real driver) fails here even though
``test_tui.py`` stays green.

PTYs are a Unix concept, so the whole module is skipped on Windows.
"""

from __future__ import annotations

import os
import re
import select
import struct
import subprocess
import sys
import time

import pytest

# ``pty`` only exists on Unix; skip this whole module on Windows.
pytest.importorskip("pty")

_ANSI_CSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
_ANSI_OSC = re.compile(rb"\x1b\][^\x07]*\x07")


def _spawn_tui_pty(cols: int = 90, rows: int = 24) -> tuple[subprocess.Popen, int]:
    """Spawn ``python -m cothis.tui`` on a fresh PTY; return (proc, master_fd).

    The slave is sized via ``TIOCSWINSZ`` so Textual lays out the 4 grid rows
    (header / conversation / input / footer) instead of falling back to a
    0x0 PTY.
    ``TERM=xterm-256color`` makes Textual pick its real terminal driver —
    the code path the bug lived in.

    ``pty`` / ``fcntl`` / ``termios`` are imported lazily and their Unix-only
    members reached via ``getattr`` so the type checker does not flag them on
    Windows (the module is skipped at runtime there via ``importorskip``).
    """
    import fcntl
    import pty
    import termios

    openpty = getattr(pty, "openpty")  # noqa: B009 — deliberate: Windows ty-safe member access
    ioctl = getattr(fcntl, "ioctl")  # noqa: B009 — deliberate: Windows ty-safe member access
    tiocswinsz = getattr(termios, "TIOCSWINSZ")  # noqa: B009 — deliberate: Windows ty-safe member access

    master, slave = openpty()
    ioctl(slave, tiocswinsz, struct.pack("HHHH", rows, cols, 0, 0))
    env = {
        **os.environ,
        "TERM": "xterm-256color",
        "LINES": str(rows),
        "COLUMNS": str(cols),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "cothis.tui"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    return proc, master


def _drain(master: int, deadline: float = 2.0, quiet_for: float = 0.3) -> bytes:
    """Read the PTY until it goes quiet for ``quiet_for`` seconds or deadline.

    Returns once the TUI stops emitting (a render finished) so the caller
    can act on a stable screen instead of a partial one.
    """
    chunks: list[bytes] = []
    last_data = time.monotonic()
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        timeout = min(0.1, end - time.monotonic())
        ready, _, _ = select.select([master], [], [], timeout)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
            last_data = time.monotonic()
        elif time.monotonic() - last_data >= quiet_for:
            break
    return b"".join(chunks)


def _strip_ansi(data: bytes) -> bytes:
    """Drop CSI/OSC escape runs so a glyph substring search is reliable."""
    return _ANSI_OSC.sub(b"", _ANSI_CSI.sub(b"", data))


def test_tui_input_receives_keystrokes_over_real_pty() -> None:
    """Typing into the focused ``TextArea`` inserts characters over a real PTY.

    The composer input holds default focus at launch (``CothisApp.on_mount``
    → ``_refocus_input``), so keystrokes land in the prompt with no Tab
    navigation. Typing must insert into the TextArea — pre-#375 the
    ``InputBar(Container)`` wrapper dropped every keystroke on this exact
    path; a focus regression (default focus moved off the input) fails
    here because the marker would type into nothing.
    """
    marker = "PTYMARKER375"
    proc, master = _spawn_tui_pty()
    try:
        _drain(master, deadline=3.0)  # let the TUI finish its first render
        # Input holds default focus — type directly, no Tab navigation.
        os.write(master, marker.encode())
        time.sleep(1.0)
        output = _drain(master, deadline=2.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.close(master)

    raw_marker = marker.encode()
    # Raw substring search first — Textual emits the TextArea glyphs as a
    # contiguous literal run, so this usually hits and skips the ANSI strip.
    # The stripped fallback covers the rare case where cursor/SGR codes split
    # the run. Textual runs the PTY in raw mode (no echo), so the marker can
    # only be in the output if the TextArea actually rendered it.
    assert raw_marker in output or raw_marker in _strip_ansi(output), (
        "typed marker not painted back by the TUI — the TextArea dropped the "
        "keystrokes under the real terminal driver (regression of #375). "
        "raw output tail: " + repr(output[-500:])
    )
