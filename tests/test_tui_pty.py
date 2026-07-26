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
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

_ANSI_CSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
_ANSI_OSC = re.compile(rb"\x1b\][^\x07]*\x07")


def _spawn_tui_pty(cols: int = 90, rows: int = 24) -> tuple[subprocess.Popen, int]:
    """Spawn ``python -m cothis.tui`` on a fresh PTY; return (proc, master_fd).

    The slave is sized via ``TIOCSWINSZ`` so Textual lays out the 3 panes
    (input docked at the bottom) instead of falling back to 80x24 defaults
    that can vary by runner. ``TERM=xterm-256color`` makes Textual pick its
    real terminal driver — the code path the bug lived in.
    """
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
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

    Launch focus is ``SessionList`` (``CothisApp.on_mount``); two Tabs move
    focus SessionList -> ConversationView -> TextArea. Typing then must
    insert into the TextArea — pre-#375 the ``InputBar(Container)`` wrapper
    dropped every keystroke on this exact path.
    """
    marker = "PTYMARKER375"
    proc, master = _spawn_tui_pty()
    try:
        _drain(master, deadline=3.0)  # let the TUI finish its first render
        os.write(master, b"\t\t")  # focus the TextArea
        time.sleep(0.4)
        _drain(master, deadline=0.5)
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

    assert marker.encode() in _strip_ansi(output), (
        "typed marker not painted back by the TUI — the TextArea dropped the "
        "keystrokes under the real terminal driver (regression of #375). "
        "raw output tail: " + repr(output[-500:])
    )
