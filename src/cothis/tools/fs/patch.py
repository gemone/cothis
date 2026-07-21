"""``cothis.tools.fs.patch`` — v4 diff format parser + applier (pure).

The deep module behind ``fs.write``'s codex ``apply_patch`` content
format. Knows nothing about the filesystem: dict-in/dict-out, paths
opaque strings. Disk I/O, cwd resolution, and atomicity live at the
``fs.write`` layer (#51–#53); this module is the testable, disk-agnostic
core.

## v4 diff format

A patch is a YAML-ish text block bracketed by ``*** Begin Patch`` and
``*** End Patch``. Between those markers, lines starting with ``***``
introduce ops; every other line is content of the current op.

::

    *** Begin Patch
    *** Add File: path/to/new
    +content line 1
    +content line 2
    *** Update File: path/to/existing
    @@ context line 1
    @@ context line 2
    -line to remove
    +line to add
    *** Delete File: path/to/gone
    *** End Patch

- **Add File**: every ``+`` line is the new file's content.
- **Update File**: ``@@`` context lines anchor the hunk; ``-`` lines are
  pre-image (must appear in the file in order); ``+`` lines replace the
  block of ``-`` lines. Multiple hunks per Update are allowed.
- **Delete File**: removes the file (no body).

## Whitespace tolerance

Pre-image matching strips trailing whitespace on both sides — editors
strip trailing ws, patches shouldn't fail purely on that. Fuzz beyond
trailing-ws is out of scope (#46).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ERR_PREVIEW_LIMIT = 80


class PatchError(Exception):
    """Base class for parser + applier errors.

    Subclasses attach ``file`` and ``line`` so the LLM can self-correct
    next turn (per #46 acceptance criteria). ``line`` is 1-based when
    known, ``0`` if the error is structural (e.g. missing ``Begin Patch``).
    """

    def __init__(self, message: str, *, file: str | None = None, line: int = 0) -> None:
        super().__init__(message)
        self.file = file
        self.line = line

    def __str__(self) -> str:
        msg = super().__str__()
        if self.file is None and not self.line:
            return msg
        parts = [msg]
        if self.file is not None:
            parts.append(f"file={self.file}")
        if self.line:
            parts.append(f"line={self.line}")
        return ", ".join(parts)


class ParseError(PatchError):
    """Raised by ``parse_patch`` on malformed input."""


class ApplyError(PatchError):
    """Raised by ``apply_patch`` when an op can't be applied."""


@dataclass(frozen=True)
class Hunk:
    """One hunk inside an Update op: context + pre-image + post-image.

    ``context`` anchors the hunk's location (zero or more lines).
    ``removes`` is the pre-image that must appear in the file (in order).
    ``adds`` replaces the ``removes`` block.
    """

    context: list[str] = field(default_factory=list)
    removes: list[str] = field(default_factory=list)
    adds: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AddFile:
    path: str
    content: str


@dataclass(frozen=True)
class UpdateFile:
    path: str
    hunks: list[Hunk]


@dataclass(frozen=True)
class DeleteFile:
    path: str


Op = AddFile | UpdateFile | DeleteFile


# ---------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------


_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD_PREFIX = "*** Add File: "
_UPDATE_PREFIX = "*** Update File: "
_DELETE_PREFIX = "*** Delete File: "


def parse_patch(text: str) -> list[Op]:
    """Parse v4 diff text into a list of ops.

    Ops are ``AddFile`` / ``UpdateFile`` / ``DeleteFile`` instances.
    ``ParseError`` is raised for any structural problem with file + line
    attached so the caller can surface it to the LLM.
    """
    lines = text.splitlines()
    if not lines or lines[0] != _BEGIN:
        raise ParseError(
            "patch must start with '*** Begin Patch'",
            line=1 if lines else 0,
        )
    if _END not in lines[1:]:
        raise ParseError(
            "patch must end with '*** End Patch'",
            line=len(lines),
        )

    ops: list[Op] = []
    i = 1  # skip Begin Patch
    n = len(lines)
    while i < n and lines[i] != _END:
        line = lines[i]
        if line.startswith(_ADD_PREFIX):
            path, content, next_i = _collect_add_body(lines, i)
            ops.append(AddFile(path=path, content=content))
            i = next_i
        elif line.startswith(_UPDATE_PREFIX):
            path, hunks, next_i = _collect_update_body(lines, i)
            ops.append(UpdateFile(path=path, hunks=hunks))
            i = next_i
        elif line.startswith(_DELETE_PREFIX):
            path = line[len(_DELETE_PREFIX):].strip()
            ops.append(DeleteFile(path=path))
            i += 1
        elif line.startswith("*** ") and line.endswith(":"):
            # cothis: scope ceiling — Rename/Move op is intentionally
            # out of scope per #46 PRD ("Out of Scope: Rename/Move op").
            # Upgrade path: add a MoveFile dataclass + arm parse_patch
            # and apply_patch when cross-issue demand lands.
            raise ParseError(
                f"unknown op marker: {_preview(line)}",
                line=i + 1,
            )
        else:
            raise ParseError(
                f"unexpected line outside any op: {_preview(repr(line))}",
                line=i + 1,
            )
    return ops


def _collect_add_body(
    lines: list[str], header_idx: int,
) -> tuple[str, str, int]:
    """Return ``(path, content, next_i)`` for an Add File op.

    ``content`` is the joined ``+`` lines with a trailing newline.
    ``next_i`` points at the next op marker or ``*** End Patch``.
    """
    header = lines[header_idx]
    path = header[len(_ADD_PREFIX):].strip()
    body: list[str] = []
    i = header_idx + 1
    while i < len(lines) and lines[i] != _END and not _is_op_marker(lines[i]):
        line = lines[i]
        if not line.startswith("+"):
            raise ParseError(
                f"Add File body line must start with '+': {_preview(repr(line))}",
                file=path,
                line=i + 1,
            )
        body.append(line[1:])
        i += 1
    content = "".join(s + "\n" for s in body)
    return path, content, i


def _collect_update_body(
    lines: list[str], header_idx: int,
) -> tuple[str, list[Hunk], int]:
    """Return ``(path, hunks, next_i)`` for an Update File op.

    ``next_i`` points at the next op marker or ``*** End Patch`` so the
    caller can continue the outer loop without re-scanning this op's body.
    """
    header = lines[header_idx]
    path = header[len(_UPDATE_PREFIX):].strip()
    hunks: list[Hunk] = []
    i = header_idx + 1
    n = len(lines)
    while i < n and lines[i] != _END and not _is_op_marker(lines[i]):
        line = lines[i]
        if line.startswith("@@"):
            hunk, i = _collect_one_hunk(lines, i, path)
            hunks.append(hunk)
        elif line.startswith("-") or line.startswith("+"):
            raise ParseError(
                f"Update File body line outside any @@ hunk: {_preview(repr(line))}",
                file=path,
                line=i + 1,
            )
        else:
            raise ParseError(
                f"unexpected line in Update File body: {_preview(repr(line))}",
                file=path,
                line=i + 1,
            )
    if not hunks:
        raise ParseError(
            "Update File has no hunks",
            file=path,
            line=header_idx + 1,
        )
    return path, hunks, i


def _collect_one_hunk(
    lines: list[str], start_idx: int, path: str,
) -> tuple[Hunk, int]:
    """Collect one hunk starting at ``lines[start_idx]`` (a ``@@`` line).

    Returns ``(hunk, next_idx_after_hunk)``. ``@@`` may be bare or carry
    trailing context text (stripped); bare ``@@`` yields no context line.
    """
    context: list[str] = []
    removes: list[str] = []
    adds: list[str] = []
    i = start_idx
    n = len(lines)
    while i < n and lines[i].startswith("@@"):
        ctx = lines[i][2:].strip()
        if ctx:
            context.append(ctx)
        i += 1
    while i < n and (lines[i].startswith("-") or lines[i].startswith("+")):
        line = lines[i]
        if line.startswith("-"):
            removes.append(line[1:])
        else:
            adds.append(line[1:])
        i += 1
    if not removes and not adds:
        raise ParseError(
            "hunk has no - or + body lines",
            file=path,
            line=start_idx + 1,
        )
    return Hunk(context=context, removes=removes, adds=adds), i


def _is_op_marker(line: str) -> bool:
    return (
        line.startswith(_ADD_PREFIX)
        or line.startswith(_UPDATE_PREFIX)
        or line.startswith(_DELETE_PREFIX)
    )


def _preview(value: object) -> str:
    """Truncate stringified value to keep log/UI lines short — patch
    lines may carry pasted secrets, and full ``repr`` of a 1KB line
    is not actionable to the LLM either."""
    text = str(value)
    if len(text) <= _ERR_PREVIEW_LIMIT:
        return text
    return text[:_ERR_PREVIEW_LIMIT - 3] + "..."


# ---------------------------------------------------------------------
# applier
# ---------------------------------------------------------------------


def apply_patch(
    files: dict[str, str], ops: list[Op],
) -> dict[str, str]:
    """Apply parsed ops to a ``{path: content}`` snapshot.

    Returns a NEW dict; ``files`` is not mutated (callers use the input
    as a rollback snapshot). Raises ``ApplyError`` (with ``file`` +
    ``line``) on any failure — pre-image not found, Add on existing path,
    Delete on missing path.
    """
    result = dict(files)
    for op in ops:
        if isinstance(op, AddFile):
            _apply_add(result, op)
        elif isinstance(op, UpdateFile):
            _apply_update(result, op)
        elif isinstance(op, DeleteFile):
            _apply_delete(result, op)
        else:  # pragma: no cover — exhaustiveness check via Op union
            raise ApplyError(f"unknown op type: {type(op).__name__}")
    return result


def _apply_add(files: dict[str, str], op: AddFile) -> None:
    if op.path in files:
        raise ApplyError(
            "Add File target already exists",
            file=op.path,
            line=1,
        )
    files[op.path] = op.content


def _apply_delete(files: dict[str, str], op: DeleteFile) -> None:
    if op.path not in files:
        raise ApplyError(
            "Delete File target does not exist",
            file=op.path,
            line=1,
        )
    del files[op.path]


def _apply_update(files: dict[str, str], op: UpdateFile) -> None:
    if op.path not in files:
        raise ApplyError(
            "Update File target does not exist",
            file=op.path,
            line=1,
        )
    content = files[op.path]
    for hunk in op.hunks:
        content = _apply_hunk(content, op.path, hunk)
    files[op.path] = content


def _apply_hunk(content: str, path: str, hunk: Hunk) -> str:
    """Apply one hunk to ``content``; return the new content.

    Locate the pre-image block (``removes`` lines, in order) within
    ``content``, with trailing-ws tolerance on both sides. Replace that
    block with ``adds``. The optional ``context`` lines anchor the match
    position — find the first occurrence of context followed by removes;
    if no context, find removes directly.
    """
    lines = content.splitlines(keepends=True)
    # cothis: scope ceiling — trailing-ws tolerance is the only fuzz
    # applied. Indentation/leading-ws changes are real diffs and must
    # round-trip verbatim (per #46 PRD "Out of Scope: patch fuzz
    # beyond trailing-ws"). Upgrade path: plug in a config to opt into
    # difflib's SequenceMatcher fuzzy matching if a real need surfaces.
    norm_lines = [ln.rstrip() for ln in lines]

    # cothis: detect dominant line ending + tail-newline state so
    # replacement lines match the file's convention (#96). Pre-#96
    # hardcoded ``"\n"`` produced mixed-endings on CRLF files, added
    # spurious trailing newlines, and concatenated lines on pure
    # insertions at EOF.
    sample = content
    nl = "\r\n" if "\r\n" in sample else "\n"
    tail_unterminated = bool(lines) and not lines[-1].endswith(("\r\n", "\n"))

    if hunk.context:
        anchor = [c.rstrip() for c in hunk.context]
        start = _find_subseq(norm_lines, anchor)
        if start is None:
            raise ApplyError(
                "hunk context not found in file",
                file=path,
            )
        search_from = start + len(anchor)
    else:
        search_from = 0

    removes_norm = [r.rstrip() for r in hunk.removes]
    if removes_norm:
        idx = _find_subseq(norm_lines, removes_norm, start=search_from)
        if idx is None:
            raise ApplyError(
                "hunk pre-image not found within context "
                f"(search started at line {search_from + 1})",
                file=path,
                line=search_from + 1,
            )
        replacement = [a + nl for a in hunk.adds]
        new_lines = lines[:idx] + replacement + lines[idx + len(removes_norm):]
        # cothis: if the replace block reaches EOF on a tail-unterminated
        # file, drop the trailing nl from its last line so the file's
        # no-trailing-newline state is preserved (#96 bug 2).
        if tail_unterminated and idx + len(removes_norm) == len(lines) and replacement:
            new_lines[-1] = new_lines[-1].rstrip("\r\n")
    else:
        replacement = [a + nl for a in hunk.adds]
        # cothis: pure insertion at EOF on a tail-unterminated file
        # would concatenate the existing last line with the first
        # inserted line (no separator). Prepend ``nl`` to the first
        # replacement line so the splice stays well-formed (#96 bug 3).
        if tail_unterminated and search_from == len(lines) and replacement:
            replacement[0] = nl + replacement[0]
        new_lines = lines[:search_from] + replacement + lines[search_from:]
    return "".join(new_lines)


def _find_subseq(
    haystack: list[str], needle: list[str], *, start: int = 0,
) -> int | None:
    """Return the index of the first occurrence of ``needle`` in
    ``haystack`` at or after ``start``, or ``None``."""
    if not needle:
        return start
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return None
