"""``cothis.session`` — in-memory conversation state + durable SQLite backing.

The Session is what survives process exit. ``Agent`` owns zero or one
``Session`` (``ask``: none, ephemeral; ``chat``: one, persisted). When
attached, the Agent enqueues every user message, every per-execution
tool_result, and every assistant message (atomically, at MessageStop);
when the Agent closes, the Session drains its queue and closes the
SQLite connection + fcntl lock.

The split between this module and :mod:`cothis.session.storage`:

- **Here** (``Session``): in-memory ``messages: list[dict]`` (Anthropic
  shape, isomorphic to ``Agent._messages``), index allocation
  (``seq``/``msg_idx``/``block_idx``), the write queue + daemon consumer,
  the fcntl cross-process lock, the lazy session-row / title / .gitignore
  policy, the field mapping between Anthropic dicts and storage row tuples,
  and the load-side rebuild (with orphan-``tool_use`` drop-trailing).
- **There** (``Storage``): SQLite CRUD only. Connection, schema, DDL,
  ``write_atomic``. No knowledge of the queue, the lock, or Anthropic
  shape.

Concurrency model (Q7-A: all sync): the Agent calls ``append_*`` from the
async event loop's thread (sync ``queue.put``, never blocks); a daemon
``threading.Thread`` consumes the queue and is the only writer to SQLite.
``Session.close`` (sync; ``Agent.aclose`` wraps it via
``asyncio.to_thread``) sets a stop flag, joins the consumer (generous
timeout), and closes the connection. ``atexit`` is the process-level
fallback for the case where ``aclose`` never runs (e.g. ``os._exit``);
it's unregistered on a clean close so it can't double-fire.

Temporal partitioning of the SQLite connection (no mutex): ``load`` runs
its single SELECT on the calling thread *before* the consumer starts;
after that, only the consumer thread touches the connection for writes;
``close`` joins the consumer before the main thread closes it.
"""

from __future__ import annotations

import atexit
import importlib.metadata
import json
import logging
import os
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from cothis.session import graph as _graph
from cothis.session.graph import SessionNotFoundError
from cothis.session.storage import BlockRow, SessionRow, Storage, is_visible

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """UTC now as ISO-8601. SQLite stores as TEXT; lexicographic == chronological."""
    return datetime.now(UTC).isoformat()


def _cli_version() -> str:
    """Read the installed package version. Works under ``uv run`` (PEP 396)."""
    try:
        return importlib.metadata.version("cothis")
    except importlib.metadata.PackageNotFoundError:
        # ponytail: dev mode without installed metadata. Storage stores it
        # for debugging only; "unknown" is honest.
        return "unknown"


def _truncate_title(text: str) -> str:
    """First line, then 60 chars, with ``...`` suffix when truncated.

    ``cothis history`` lists sessions one-per-line, so a title with a
    newline breaks alignment. Long titles get ellipsised.
    """
    first_line = text.splitlines()[0] if text else ""
    if len(first_line) <= 60:
        return first_line
    return first_line[:57] + "..."


def _validate_session_id(sid: str, *, name: str = "session_id") -> None:
    """Reject anything that isn't a 32-char lowercase hex string.

    ``session_id`` becomes a filesystem path in ``_lock_path`` and a
    SQLite primary key. Safe in ``Session.new`` (uuid4 hex, internal),
    but ``load`` / ``delete`` / ``fork`` receive ids from callers — a
    bad value here would either traverse the filesystem (``../``) or
    fail an FK constraint deep in SQLite with a worse error message.
    """
    if len(sid) != 32 or not all(c in "0123456789abcdef" for c in sid):
        raise ValueError(f"invalid {name}: {sid!r}")


def _block_to_row(
    session_id: str,
    seq: int,
    msg_idx: int,
    block_idx: int,
    role: str,
    block: dict[str, Any],
    ts: str,
) -> BlockRow:
    """Map an Anthropic content-block dict to a storage :class:`BlockRow`.

    Inverse of :func:`_row_to_block`. The five block types ``Agent``
    produces (text / thinking / tool_use / tool_result / image) each have
    a fixed column mapping; ``is_error`` on ``tool_result`` is deliberately
    not persisted — it's an ephemeral flag for the model's *next* turn,
    and by reload time the assistant has already reacted to it.
    """
    btype = block.get("type")
    content: str | None = None
    signature: str | None = None
    tool_id: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_use_id: str | None = None
    tool_output: str | None = None
    image_source: str | None = None

    if btype == "text":
        content = block.get("text")
    elif btype == "thinking":
        content = block.get("thinking")
        signature = block.get("signature")
    elif btype == "tool_use":
        tool_id = block.get("id")
        tool_name = block.get("name")
        tool_input = json.dumps(block.get("input", {}))
    elif btype == "tool_result":
        tool_use_id = block.get("tool_use_id")
        tool_output = block.get("content")
    elif btype == "image":
        source = block.get("source")
        image_source = json.dumps(source) if source is not None else None
    else:
        # ponytail: unknown block type — defensive, shouldn't happen for
        # blocks Agent itself produced. Store the whole dict so reload is
        # lossless via the inverse fallback in _row_to_block.
        content = json.dumps(block)

    return BlockRow(
        session_id=session_id,
        seq=seq,
        msg_idx=msg_idx,
        block_idx=block_idx,
        role=role,
        type=btype if btype is not None else "text",
        ts=ts,
        content=content,
        signature=signature,
        tool_id=tool_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
        tool_output=tool_output,
        image_source=image_source,
    )


def _row_to_block(row: BlockRow) -> dict[str, Any]:
    """Inverse of :func:`_block_to_row` — rebuild an Anthropic content block.

    ``is_error`` is intentionally not restored (see :func:`_block_to_block`).
    An unknown ``type`` (one that round-tripped through the ``content =
    json.dumps(block)`` fallback) is restored by ``json.loads``-ing the
    content verbatim.
    """
    if row.type == "text":
        return {"type": "text", "text": row.content or ""}
    if row.type == "thinking":
        out: dict[str, Any] = {"type": "thinking", "thinking": row.content or ""}
        if row.signature is not None:
            out["signature"] = row.signature
        return out
    if row.type == "tool_use":
        return {
            "type": "tool_use",
            "id": row.tool_id or "",
            "name": row.tool_name or "",
            "input": json.loads(row.tool_input) if row.tool_input else {},
        }
    if row.type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": row.tool_use_id or "",
            "content": row.tool_output if row.tool_output is not None else "",
        }
    if row.type == "image":
        return {
            "type": "image",
            "source": json.loads(row.image_source) if row.image_source else {},
        }
    # Unknown type — restore the verbatim block we json.dumps'd at write time.
    if row.content is not None:
        try:
            loaded = json.loads(row.content)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {"type": row.type, "text": row.content or ""}


def _rebuild_messages(
    rows: list[BlockRow],
) -> tuple[list[dict[str, Any]], int | None]:
    """Rebuild Anthropic-shape ``messages`` from flat block rows.

    Returns ``(messages, cut_msg_idx)`` where ``cut_msg_idx`` is the
    ``msg_idx`` of the first orphan-``tool_use`` message, or ``None`` if no
    truncation happened. The caller (``Session.load``) uses ``cut_msg_idx``
    to DELETE the orphan rows from ``blocks`` so a later reload doesn't
    re-truncate at the same point (and silently erase any messages appended
    between reloads — finding from review pass).

    Groups ``rows`` by ``msg_idx`` (already ordered by ``msg_idx,
    block_idx`` from ``load_blocks``), emits one ``{role, content: [...]}``
    per group. Then runs the orphan-``tool_use`` drop-trailing pass: if any
    assistant message's ``tool_use`` block has no matching ``tool_result``
    in any *later* user message, the entire trailing run from that
    assistant message onward is dropped. This is the crash case Q2-A guards
    against — a partial assistant write would otherwise leave an unpaired
    ``tool_use`` that 400s the next ``amessages`` call.

    The dropped suffix is logged at ``WARNING`` so a user inspecting logs
    sees that history was truncated on reload.
    """
    if not rows:
        return [], None

    messages: list[dict[str, Any]] = []
    current_idx: int | None = None
    current_role: str | None = None
    current_blocks: list[dict[str, Any]] = []
    for row in rows:
        if row.msg_idx != current_idx:
            if current_idx is not None:
                messages.append(
                    {"role": current_role, "content": current_blocks}
                )
            current_idx = row.msg_idx
            current_role = row.role
            current_blocks = [_row_to_block(row)]
        else:
            current_blocks.append(_row_to_block(row))
    if current_idx is not None:
        messages.append({"role": current_role, "content": current_blocks})

    cut_at = _orphan_truncate_index(messages)
    if cut_at is None:
        return messages, None
    # Map the in-memory message index back to a msg_idx (the storage-side
    # identifier). messages[i] came from the i-th distinct msg_idx group;
    # walk rows once to recover that map.
    msg_idx_of_message: list[int] = []
    seen: set[int] = set()
    for r in rows:
        if r.msg_idx not in seen:
            seen.add(r.msg_idx)
            msg_idx_of_message.append(r.msg_idx)
    cut_msg_idx = msg_idx_of_message[cut_at]
    logger.warning(
        "Session reload: dropping %d trailing message(s) with unpaired "
        "tool_use (crash recovery, Q2-A); persisting the truncate.",
        len(messages) - cut_at,
    )
    return messages[:cut_at], cut_msg_idx


def _orphan_truncate_index(
    messages: list[dict[str, Any]],
) -> int | None:
    """Return the index to truncate ``messages`` at, or ``None`` if clean.

    Walks messages forward, collecting every seen ``tool_use`` id; for each
    ``tool_result`` removes its matching id from the pending set. If any
    message leaves a still-pending ``tool_use``, that message is the start
    of the orphan tail — return its index so the caller slices it off.

    The matching is *later*-only: a ``tool_result`` for an id we haven't
    seen yet is itself suspicious but not poison (Anthropic permits it on
    the wire only after the ``tool_use``, so on a clean write it can't
    happen; on a dirty reload we don't try to repair it).
    """
    pending: dict[str, int] = {}  # tool_use_id -> msg_idx that introduced it
    for i, msg in enumerate(messages):
        for block in msg.get("content", []):
            btype = block.get("type")
            if btype == "tool_use":
                pending[block.get("id")] = i
            elif btype == "tool_result":
                pending.pop(block.get("tool_use_id"), None)
    if not pending:
        return None
    return min(pending.values())


class SessionLockedError(RuntimeError):
    """Another live cothis process holds the lock for this session.

    Raised by :meth:`Session.new` / :meth:`Session.load` when the
    cross-process file lock (``filelock.FileLock`` with ``timeout=0``)
    can't be acquired. The CLI surfaces this with a clear message and a
    non-zero exit — it is *not* retried automatically, because the other
    process may be mid-write and retrying would race it.

    Cross-platform by design: ``filelock`` uses ``fcntl.flock`` on POSIX
    and ``msvcrt.locking`` + ``kernel32`` on Windows, so the guard works
    on every platform cothis supports.
    """


class SessionHasChildrenError(RuntimeError):
    """``cothis delete`` was called on a non-leaf session node.

    The fork tree's invariant is "no orphans": deleting a node with
    living children would orphan them (their ``parent_id`` would dangle).
    Delete the children first. The exception carries the offending id
    and its direct children so the CLI can name them in the error.
    """

    def __init__(self, session_id: str, children: list[str]) -> None:
        super().__init__(
            f"session {session_id!r} has {len(children)} child fork(s); "
            f"delete them first"
        )
        self.session_id = session_id
        self.children = children


class Session:
    """In-memory conversation state backed by SQLite.

    Construct via :meth:`new` (fresh id, lazy row) or :meth:`load` (resume
    by id). Both take the cross-process file lock eagerly — the second
    process to reach a live session is refused (:class:`SessionLockedError`).

    The Session's in-memory ``messages`` is the ground truth the Agent
    reads from; SQLite is its durable mirror. After :meth:`load`, reads
    never hit the DB — the single SELECT has already populated
    ``messages``. Writes flow through an in-process queue to a daemon
    consumer (or, in test mode, inline via ``flush_sync=True``).
    """

    # --- construction ---------------------------------------------------

    def __init__(
        self,
        *,
        db_path: Path,
        session_id: str,
        storage: Storage,
        cwd: Path,
        model: str,
        cli_version: str,
        created_at: str,
        messages: list[dict[str, Any]],
        session_row_written: bool,
        parent_id: str | None = None,
        parent_seq: int | None = None,
        next_seq: int = 0,
        next_msg_idx: int = 0,
        flush_sync: bool = False,
    ) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._storage = storage
        self._cwd = cwd
        self._model = model
        self._cli_version = cli_version
        self._created_at = created_at
        self.messages = messages  # public: Agent reads this after load
        self._session_row_written = session_row_written
        # Fork-tree link. ``None`` on roots and on sessions loaded by id
        # (the ancestor chain is already assembled into ``messages`` at
        # load time). ``fork`` passes both; ``new`` / ``load`` don't.
        self._parent_id = parent_id
        self._parent_seq = parent_seq

        # Index allocation state. Mutated only at enqueue (Agent thread).
        # Derived from the loaded rows on resume (see load()).
        self._next_seq = next_seq
        self._next_msg_idx = next_msg_idx

        self._closed = False
        self._flush_sync = flush_sync

        # Queue + consumer thread. Skipped entirely in flush_sync mode —
        # _drain_one is called inline from append_message instead.
        self._queue: queue.SimpleQueue | None = None
        self._consumer: threading.Thread | None = None
        self._stop = threading.Event()
        if not flush_sync:
            self._queue = queue.SimpleQueue()
            self._consumer = threading.Thread(
                target=self._consumer_loop,
                name=f"cothis-session-{session_id[:8]}",
                daemon=True,
            )
            self._consumer.start()
        atexit.register(self._drain_sync)

    @classmethod
    def new(
        cls,
        db_path: Path,
        *,
        cwd: Path,
        model: str,
        flush_sync: bool = False,
    ) -> Session:
        """Create a fresh session: allocate id, take lock, open storage.

        Does NOT write the ``sessions`` row — that's lazy, written on the
        first enqueue's drain along with the title. The cross-process lock
        is eager: a second process reaching this id is refused immediately.

        ``db_path`` is the resolved SQLite file path (CLI's
        ``_resolve_db_path`` handles the three modes: default single-file,
        project split, custom-Dir split). Lock files live separately under
        the cache dir (see ``_lock_path``), decoupled from the db location.
        """
        db_path = db_path.expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        session_id = uuid.uuid4().hex
        lock = cls._take_lock(cls._lock_path(session_id))
        try:
            storage = Storage(db_path)
        except BaseException:
            lock.release()
            raise
        session = cls(
            db_path=db_path,
            session_id=session_id,
            storage=storage,
            cwd=cwd,
            model=model,
            cli_version=_cli_version(),
            created_at=_now_iso(),
            messages=[],
            session_row_written=False,
            flush_sync=flush_sync,
        )
        session._lock = lock
        return session

    @classmethod
    def load(
        cls,
        db_path: Path,
        session_id: str,
        *,
        cwd: Path | None = None,
        flush_sync: bool = False,
    ) -> Session:
        """Resume an existing session by id.

        Takes the cross-process lock (refuses if held), runs the single
        SELECT to rebuild ``messages`` (with orphan-``tool_use``
        drop-trailing), and starts the consumer. The ``sessions`` row
        already exists — no lazy write, no title computation, no
        ``.gitignore`` write.

        For a forked session, ancestor-chain context is assembled here:
        walk ``SessionGraph.ancestors(session_id)`` (root → parent),
        load each ancestor's blocks through that link's ``parent_seq``
        cap, run the orphan-truncate per segment, and prepend the
        ancestor segments to this session's own messages. The result
        is one flat ``messages`` list the Agent reads as if it were a
        single linear conversation — git-branch semantics, no merge.

        ``cwd`` (when given) enforces the visibility filter used by
        ``cothis chat --resume``: a session whose ``cwd`` is neither
        ``cwd`` nor an ancestor of it is treated as not-found
        (``KeyError``), matching ``cothis history``'s listing scope.
        """
        db_path = db_path.expanduser()
        _validate_session_id(session_id)
        lock = cls._take_lock(cls._lock_path(session_id))
        try:
            storage = Storage(db_path)
        except BaseException:
            lock.release()
            raise
        sr = storage.load_session(session_id)
        if sr is None:
            storage.close()
            lock.release()
            raise KeyError(f"session {session_id!r} not found")
        if cwd is not None and not is_visible(Path(sr.cwd), cwd):
            storage.close()
            lock.release()
            raise KeyError(f"session {session_id!r} not found")
        graph = _graph.build(storage.list_sessions())
        rows = storage.load_blocks(session_id)
        messages, cut_msg_idx = _rebuild_messages(rows)
        if cut_msg_idx is not None:
            storage.delete_blocks_from_msg_idx(session_id, cut_msg_idx)
            rows = [r for r in rows if r.msg_idx < cut_msg_idx]
        if sr.parent_id is not None:
            ancestor_segments = cls._assemble_ancestors(graph, storage, session_id)
            messages = ancestor_segments + messages
        next_seq = (max(r.seq for r in rows) + 1) if rows else 0
        next_msg_idx = (max(r.msg_idx for r in rows) + 1) if rows else 0
        session = cls(
            db_path=db_path,
            session_id=session_id,
            storage=storage,
            cwd=Path(sr.cwd),
            model=sr.model or "",
            cli_version=sr.cli_version or _cli_version(),
            created_at=sr.created_at,
            messages=messages,
            session_row_written=True,
            next_seq=next_seq,
            next_msg_idx=next_msg_idx,
            flush_sync=flush_sync,
        )
        session._lock = lock
        return session

    @classmethod
    def list_visible(
        cls,
        db_path: Path,
        cwd: Path,
    ) -> list[SessionRow]:
        """Sessions visible from ``cwd`` — used by ``cothis history``.

        Visible means: the session's ``cwd`` equals ``cwd`` or is an
        ancestor of it (so project-root sessions are listed from a
        subdirectory). No lock is taken — this is a read-only listing.
        Sorted by ``updated_at`` descending.
        """
        db_path = db_path.expanduser()
        storage = Storage(db_path)
        try:
            return storage.list_sessions_in_cwd_tree(cwd)
        finally:
            storage.close()

    @classmethod
    def delete(
        cls,
        db_path: Path,
        session_id: str,
    ) -> None:
        """Leaf-only delete of ``session_id``.

        Takes the cross-process lock (refuses if held), builds the
        graph, refuses with :class:`SessionHasChildrenError` if the
        node is not a leaf, then DELETEs the session's blocks + row.
        Hot-DB only in this slice (#35); cold-DB delete lands in #36.
        """
        db_path = db_path.expanduser()
        _validate_session_id(session_id)
        lock = cls._take_lock(cls._lock_path(session_id))
        try:
            storage = Storage(db_path)
        except BaseException:
            lock.release()
            raise
        try:
            target = storage.load_session(session_id)
            if target is None:
                raise KeyError(f"session {session_id!r} not found")
            graph = _graph.build(storage.list_sessions())
            if not _graph.is_leaf(graph, session_id):
                children = _graph.children_of(graph, session_id)
                raise SessionHasChildrenError(session_id, children)
            storage.delete_session(session_id)
        finally:
            storage.close()
            lock.release()

    @classmethod
    def peek_messages(
        cls,
        db_path: Path,
        session_id: str,
        *,
        cwd: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Read-only message preview for ``cothis history <id>``.

        No lock is taken (display only); the caller accepts that a session
        mid-write may show a partial last message. ``KeyError`` propagates
        if the id is unknown OR (when ``cwd`` is passed) the session is
        out of scope — same predicate as :meth:`load`.
        """
        db_path = db_path.expanduser()
        _validate_session_id(session_id)
        storage = Storage(db_path)
        try:
            sr = storage.load_session(session_id)
            if sr is None:
                raise KeyError(f"session {session_id!r} not found")
            if cwd is not None and not is_visible(Path(sr.cwd), cwd):
                raise KeyError(f"session {session_id!r} not found")
            rows = storage.load_blocks(session_id)
            messages, _ = _rebuild_messages(rows)
            return messages
        finally:
            storage.close()

    @staticmethod
    def _assemble_ancestors(
        graph: dict[str, SessionRow],
        storage: Storage,
        session_id: str,
        *,
        override_link: tuple[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Load ancestor blocks (capped at each link's ``parent_seq``) and reuild.

        ``override_link`` is ``(parent_id, parent_seq)`` for ``session_id``
        when the new fork's row isn't in ``graph`` yet (the in-flight
        case during ``Session.fork``). When ``session_id`` is in ``graph``,
        the override is ignored.
        """
        start_parent_id: str | None = None
        start_parent_seq: int | None = None
        if session_id not in graph and override_link is not None:
            start_parent_id, start_parent_seq = override_link
        chain = _graph.walk_ancestors(
            graph,
            session_id,
            start_parent_id=start_parent_id,
            start_parent_seq=start_parent_seq,
        )
        segments: list[list[dict[str, Any]]] = []
        for ancestor_id, cap in chain:
            if cap is None:
                continue
            cap_rows = storage.load_blocks_through_seq(ancestor_id, cap)
            seg_messages, _ = _rebuild_messages(cap_rows)
            if seg_messages:
                segments.append(seg_messages)
        return [m for seg in segments for m in seg]

    @classmethod
    def fork(
        cls,
        db_path: Path,
        parent_session_id: str,
        parent_seq: int,
        *,
        cwd: Path,
        model: str,
        flush_sync: bool = False,
    ) -> Session:
        """Fork a new session from ``parent_session_id`` at ``parent_seq``.

        The new session gets a fresh id, a fresh ``seq``/``msg_idx``
        counter starting at 0 (independent numbering — git-branch
        semantics, no merge), and ``parent_id``/``parent_seq`` set on
        its lazy row so the fork tree records the link. Ancestor-chain
        context is loaded eagerly so ``session.messages`` reads as one
        flat conversation the Agent can resume from directly.
        """
        db_path = db_path.expanduser()
        _validate_session_id(parent_session_id, name="parent_session_id")
        if parent_seq < 0:
            raise ValueError(f"parent_seq must be >= 0, got {parent_seq}")
        new_id = uuid.uuid4().hex
        lock = cls._take_lock(cls._lock_path(new_id))
        try:
            storage = Storage(db_path)
        except BaseException:
            lock.release()
            raise
        parent_row = storage.load_session(parent_session_id)
        if parent_row is None:
            storage.close()
            lock.release()
            raise KeyError(f"parent session {parent_session_id!r} not found")
        graph = _graph.build(storage.list_sessions())
        ancestor_segments = cls._assemble_ancestors(
            graph,
            storage,
            new_id,
            override_link=(parent_session_id, parent_seq),
        )
        session = cls(
            db_path=db_path,
            session_id=new_id,
            storage=storage,
            cwd=cwd,
            model=model,
            cli_version=_cli_version(),
            created_at=_now_iso(),
            messages=list(ancestor_segments),
            session_row_written=False,
            parent_id=parent_session_id,
            parent_seq=parent_seq,
            next_seq=0,
            next_msg_idx=0,
            flush_sync=flush_sync,
        )
        session._lock = lock
        # cothis: see ADR-0009 §3 for the eager fork-row write.
        session._write_fork_row()
        return session

    def _write_fork_row(self) -> None:
        """Persist the sessions row eagerly so ``--resume <fork_id>`` works.

        See ADR-0009 §3 for why this deviates from ``Session.new``'s
        lazy-row strategy.
        """
        updated_at = _now_iso()
        self._maybe_write_gitignore()
        title = self._derive_title()
        session_row = SessionRow(
            id=self._session_id,
            parent_id=self._parent_id,
            parent_seq=self._parent_seq,
            cwd=str(self._cwd),
            cli_version=self._cli_version,
            model=self._model,
            title=title,
            created_at=self._created_at,
            updated_at=updated_at,
        )
        try:
            self._storage.write_atomic(session_row, [], updated_at)
        except Exception:  # noqa: BLE001 — log + continue; first block drain retries
            logger.exception(
                "Session %s: fork-row write failed; will retry on first append.",
                self._session_id,
            )
            return
        self._session_row_written = True

    # --- lock -----------------------------------------------------------

    @staticmethod
    def _cache_dir() -> Path:
        """XDG cache dir for lock files.

        ``$XDG_CACHE_HOME`` if set, else ``~/.cache``. Locks are not
        durable state — they're flock carriers, regenerable, safe to wipe
        on reboot / tmpfs clear. Keeping them out of ``$COTHIS_HOME``
        leaves user-edited files (tools/, AGENTS.md) clean and lets the OS
        manage lockfile lifecycle. Cross-platform (POSIX): Windows uses
        the same path; a future revision can resolve ``LOCALAPPDATA`` if
        needed.
        """
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "cothis"
        return Path.home() / ".cache" / "cothis"

    @classmethod
    def _lock_path(cls, session_id: str) -> Path:
        """Lock file path: ``<cache_dir>/<session_id>.lock``.

        ``session_id`` is a uuid4 hex (globally unique), so no db-scoping
        is needed — two different dbs can't produce colliding ids.
        """
        return cls._cache_dir() / f"{session_id}.lock"

    # The cross-process lock. Held for the Session's lifetime, released on
    # close(). Per-instance (not class-level): two Sessions may briefly
    # coexist (e.g. test helpers, or a future multi-session feature); a
    # class-level slot would have s1.close() release s2's lock. ``FileLock``
    # is a Python object that wraps the platform lock (fcntl on POSIX,
    # msvcrt + kernel32 on Windows) — holding the object alive is what
    # keeps the OS lock held; ``release()`` drops it.
    _lock: FileLock | None = None

    @staticmethod
    def _take_lock(lock_path: Path) -> FileLock:
        """Acquire the cross-process lock on ``lock_path``; refuse if held.

        ``timeout=0`` makes the acquire non-blocking — a held lock raises
        ``filelock.Timeout`` immediately, which we translate to
        :class:`SessionLockedError`. The lockfile's parent directory is
        created if missing (cache dir on first run).
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path), timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise SessionLockedError(
                f"session {lock_path.stem!r} is in use by another cothis process"
            ) from exc
        return lock

    def _release_lock(self) -> None:
        if self._lock is not None:
            try:
                self._lock.release()
            finally:
                self._lock = None

    # --- enqueue API (Agent calls these) --------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    def append_message(self, role: str, blocks: list[dict[str, Any]]) -> None:
        """Append a multi-block message atomically (one txn on drain).

        Used for ``user`` (text) and ``assistant`` (the post-MessageStop
        content list). The whole ``blocks`` list shares one transaction
        on drain — the assistant-atomic invariant (Q2-A) depends on this:
        an orphan ``tool_use`` without its sibling blocks would poison
        the next turn.

        Anthropic requires strict user/assistant alternation, so a
        same-role call (only happens for per-execution ``tool_result``
        via :meth:`append_block`) **merges** into the last message's
        ``content`` rather than starting a new message dict — the
        in-memory mirror stays alternation-clean.
        """
        if self._closed:
            raise RuntimeError("append_message on a closed Session")
        if not blocks:
            return
        rows = self._alloc_and_map(role, blocks)
        if self.messages and self.messages[-1]["role"] == role:
            self.messages[-1]["content"].extend(blocks)
        else:
            self.messages.append({"role": role, "content": list(blocks)})
        self._enqueue(rows)

    def append_block(self, role: str, block: dict[str, Any]) -> None:
        """Append a single block as its own atomic write (per-execution).

        Sugar for ``append_message(role, [block])``. Used for per-execution
        ``tool_result`` blocks (Q22): each tool execution is durable as
        soon as it finishes, not batched at end-of-turn. Consecutive
        same-role ``append_block`` / ``append_message`` calls merge into
        the same user message (Anthropic alternation invariant).
        """
        self.append_message(role, [block])

    def _alloc_and_map(
        self, role: str, blocks: list[dict[str, Any]]
    ) -> list[BlockRow]:
        """Allocate seq/msg_idx/block_idx for each block + map to BlockRow.

        Q3-A allocation rule, mirror-driven: if the last in-memory message
        has the same role (a continuation — only happens for per-exec
        ``tool_result``), reuse its ``msg_idx`` and start ``block_idx``
        at the current content length; else open a new ``msg_idx`` at 0.
        ``seq`` is monotonic across the whole session. The mirror is the
        single source of truth — no parallel ``_last_role`` counter that
        could drift from it on resume.
        """
        if self.messages and self.messages[-1]["role"] == role:
            msg_idx = self._next_msg_idx - 1
            block_idx_base = len(self.messages[-1]["content"])
        else:
            msg_idx = self._next_msg_idx
            self._next_msg_idx += 1
            block_idx_base = 0

        ts = _now_iso()
        rows: list[BlockRow] = []
        for offset, block in enumerate(blocks):
            rows.append(
                _block_to_row(
                    self._session_id,
                    self._next_seq,
                    msg_idx,
                    block_idx_base + offset,
                    role,
                    block,
                    ts,
                )
            )
            self._next_seq += 1
        return rows

    def _enqueue(self, rows: list[BlockRow]) -> None:
        """Send one atomic write through the queue, or inline if flush_sync."""
        if self._flush_sync or self._queue is None:
            self._drain_one(rows)
        else:
            self._queue.put(rows)

    # --- consumer -------------------------------------------------------

    def _consumer_loop(self) -> None:
        """Drain the queue until _stop, then drain any residual."""
        assert self._queue is not None
        while not self._stop.is_set():
            try:
                rows = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._drain_one(rows)
        # Residual drain — anything enqueued before close() set _stop.
        while True:
            try:
                rows = self._queue.get_nowait()
            except queue.Empty:
                break
            self._drain_one(rows)

    def _drain_one(self, rows: list[BlockRow]) -> None:
        """Persist one atomic write. Idempotent on the session-row init.

        First call (for a ``new`` session): compute the title from the
        first user-text block in ``self.messages``, write the ``.gitignore``
        if applicable, then ``write_atomic`` with a fresh ``SessionRow``.
        Subsequent calls / all ``load`` calls: pass ``session_row=None``.
        """
        if not rows:
            return
        updated_at = _now_iso()
        session_row: SessionRow | None = None
        if not self._session_row_written:
            self._maybe_write_gitignore()
            title = self._derive_title()
            session_row = SessionRow(
                id=self._session_id,
                parent_id=self._parent_id,
                parent_seq=self._parent_seq,
                cwd=str(self._cwd),
                cli_version=self._cli_version,
                model=self._model,
                title=title,
                created_at=self._created_at,
                updated_at=updated_at,
            )
        # ponytail: set the flag AFTER write_atomic returns. Setting it
        # before would mean a failed first drain leaves no sessions row
        # yet every subsequent INSERT into blocks hits the FK constraint —
        # silent total durability loss. The flag's job is to gate the lazy
        # row write, and that write is only done once it actually commits.
        try:
            self._storage.write_atomic(session_row, rows, updated_at)
        except Exception:  # noqa: BLE001 — log + swallow so consumer survives
            # ponytail: a failed write must not kill the consumer (would
            # block all future enqueues). Log loudly; the Agent continues
            # against the in-memory messages, so the turn isn't lost from
            # the user's view, only from durability. Upgrade path: retry
            # queue or fail-fast depending on the failure mode.
            logger.exception(
                "Session %s: write_atomic failed for %d block(s); data is "
                "in memory but not durable.",
                self._session_id,
                len(rows),
            )
            return
        if session_row is not None:
            self._session_row_written = True

    def _derive_title(self) -> str:
        """First user-text block across ``self.messages``; line+60 truncated."""
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    return _truncate_title(block["text"])
        return ""

    def _maybe_write_gitignore(self) -> None:
        """Write ``.gitignore`` (``*``) if the db dir is in the project.

        Condition: the db's parent dir resolves inside the process's cwd
        at Session construction time (captured as ``self._cwd``). Triggers
        only in split mode (``TYPE=project`` → ``<cwd>/.agents/sessions/``,
        or ``DIR=<cwd>/foo``); default mode's ``~/.cothis/`` is never in a
        project. Action: write ``*`` to ``<db_dir>/.gitignore`` only when
        it does not already exist (Q9-A: respect user rules).
        """
        try:
            db_dir = self._db_path.parent.resolve()
            in_project = db_dir.is_relative_to(self._cwd.resolve())
        except (OSError, ValueError):
            return
        if not in_project:
            return
        gitignore = db_dir / ".gitignore"
        if gitignore.exists():
            return
        try:
            gitignore.write_text("*\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write %s: %s", gitignore, exc)

    # --- shutdown -------------------------------------------------------

    def _drain_sync(self) -> None:
        """atexit fallback: drain + close if close() never ran.

        Sync context, no asyncio loop. Idempotent — if close() already
        ran, _closed short-circuits.
        """
        try:
            self.close()
        except Exception:  # noqa: BLE001 — atexit must not raise
            logger.exception("Session %s: atexit drain failed", self._session_id)

    def close(self) -> None:
        """Drain the queue, join the consumer, close storage + lock.

        Idempotent. Unregisters the atexit hook so the process-exit path
        can't double-fire. The consumer is given 5s to drain; if it
        hasn't finished by then (huge backlog, slow disk), the lock is
        released but storage is left OPEN — closing a SQLite connection
        from this thread while the daemon consumer is mid-``write_atomic``
        would make its next write raise ``ProgrammingError`` (swallowed
        by ``_drain_one``), losing the entire remaining queue. The
        daemon finishes its residual drain on its own; SQLite's WAL
        recovery handles any incomplete txn at process exit. Same loss
        ceiling as ``kill -9``, strictly better than force-close.
        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._drain_sync)
        self._stop.set()
        consumer_alive = False
        if self._consumer is not None:
            # 5s is generous: a single sqlite commit is 30-70ms (issue's
            # measurement); 5s drains dozens of backlogged writes. If the
            # consumer is somehow stuck (a bug), don't hang the CLI for
            # half a minute — release the lock and let the daemon finish.
            self._consumer.join(timeout=5)
            consumer_alive = self._consumer.is_alive()
            if consumer_alive:
                logger.warning(
                    "Session %s: consumer still alive after 5s close; "
                    "leaving storage open for daemon to finish draining "
                    "(lock released; loss ceiling = kill -9).",
                    self._session_id,
                )
        # Note: close() runs on the caller's thread (NOT via
        # asyncio.to_thread) — filelock's lock counter is thread-local,
        # so acquire (in new/load, on the main thread) and release must
        # be on the same thread. The blocking cost is bounded (drain +
        # 5s join ceiling) and only paid once at session end.
        try:
            if not consumer_alive:
                self._storage.close()
        finally:
            self._release_lock()
