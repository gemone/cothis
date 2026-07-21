# Cold/hot delete extension

`cothis delete <id>` from #35 only knew about the hot DB. Once #36
introduced cold DBs, a session could live in either place; the
delete command needed to find rows wherever they were. #87 extends
`Session.delete` to span both DBs without changing the public API.

## 1. Hot-first lookup, archive index as fallback

`Session.delete` loads from hot first. On a hit, the existing #35
path runs (load → leaf-check → delete). On a hot miss, it consults
`ArchiveIndex`. If the index has the session, `delete_cold_session`
ATTACHes the cold DB, DELETEs rows + VACUUMs, drops the index entry.
A hot miss with no index entry is a real `KeyError`.

### Considered

- **Always check cold, even on hot hit.** Rejected: redundant work.
  A session in hot is by definition not archived (the index entry is
  dropped on promote per #86).
- **CLI flag (`--cold`) that picks the DB explicitly.** Rejected: the
  user shouldn't have to know where a session lives. The index is
  the source of truth — let the lookup decide.

### Consequences

- The hot path keeps its single-SELECT cost. Cold path adds one
  `index.get` (in-memory after the JSON load) + one ATTACH transaction.
- A stale index entry (cold DB file deleted out of band) drops on
  first contact — `delete_cold_session` returns `False`, the caller
  raises `KeyError`, the index is left cleaner than it was.

## 2. Leaf-only check spans both DBs

The #35 leaf-only invariant refuses delete on a node with children.
With cold storage, children of a hot parent may have sunk to cold
(archived), and vice versa. The check now runs against both:

```python
hot_children = storage.children_of(session_id) if has_children else []
cold_children = cold_session_children(archive_dir, session_id)
if hot_children or cold_children:
    raise SessionHasChildrenError(session_id, hot_children + cold_children)
```

`cold_session_children` walks every `YYYY-MM.db` under `archive_dir`
and queries each — a child can be in any monthly bucket.

### Considered

- **Walk only the index, not the cold DB files.** Rejected: the index
  is per-session, not per-parent. To find children we'd have to load
  every entry's row. Globbing the monthly files and asking SQLite is
  cheaper (one indexed SELECT per file).
- **Refuse hot delete if the node was ever a parent (i.e., keep a
  tombstone).** Rejected: that's a schema change for a corner case.
  The cross-DB children check catches it at delete time.

### Consequences

- The error message lists children from both DBs; the user sees one
  combined list and must delete them all (hot or cold) before the
  parent can go.
- A hot parent whose children were archived still appears in the hot
  DB — `cothis history` shows it until the cold children are deleted
  and the parent itself becomes a leaf.

## 3. Storage-layer helpers, not CLI flags

`delete_cold_session` and `cold_session_children` are module-level
functions in `cothis.session.archive`. They share the parameter-bound
ATTACH pattern from `archive_session` / `promote_session`. No CLI
flag, no `Storage` method additions — `Storage` stays hot-only.

### Considered

- **Add `Storage.delete_cold` / `Storage.cold_children` methods.**
  Rejected: `Storage` is one-connection-per-hot-DB. The cold helpers
  open transient connections to the cold DBs; bolting them onto
  `Storage` would either need a second connection (which the class
  doesn't model) or an ATTACH that pollutes the writer connection.
  Module functions match the `archive_session` precedent and keep
  `Storage` focused.

### Consequences

- The CLI's `cothis delete` command needs no change — `Session.delete`
  absorbs the new logic. Existing tests of the CLI command still pass.
- Future code paths that need to query cold state (e.g., `cothis
  history --include-archived`) reuse these helpers without going
  through `Session`.
