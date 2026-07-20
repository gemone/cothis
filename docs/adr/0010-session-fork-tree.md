# Session fork tree

The session store (#29 / #34) gained a fork tree: each session
optionally records `(parent_id, parent_seq)` and the Agent, on resume,
walks the ancestor chain to assemble one flat message list. Four
interdependent decisions are recorded together — splitting them across
four ADRs would hide the design.

## 1. Tree shape: flat rows + in-memory free-function module

Each session row carries `parent_id` and `parent_seq`; the fork tree
is the transitive closure of `parent_id` links. SQLite stores flat
rows (one per session). Tree operations (`roots` / `ancestors` /
`subtree` / `children_of` / `is_leaf` / `walk_ancestors`) live in
`cothis.session.graph` as **free functions over `dict[str, SessionRow]`**,
not a class wrapping state. The map is built once per CLI invocation
from `Storage.list_sessions` and treated as immutable.

Measured against a recursive CTE on a 10k-session fixture: the
in-memory walk is 3-10x faster (`flat_load` ≈ 9ms; recursive CTE
across the same set is materially slower). The CTE also proved harder
to reason about under concurrent `BEGIN deferred` writers (validated
at fairness=1.0 in #34's research).

### Considered

- **Recursive CTE in SQLite.** Rejected on perf + legibility.
- **`SessionGraph` class wrapping `dict[str, SessionRow]` + inverse
  maps.** Rejected in round 2 of #56's review: the class wrapped
  dicts, carried no real invariant, every method was `_require` + dict
  lookup. Free functions over the input map match "boring over
  clever" and drop ~30 lines of boilerplate.

### Consequences

- The graph is single-threaded by design (one per CLI invocation); no
  concurrent mutation is supported.
- `Session.fork` passes `(parent_id, parent_seq)` directly to
  `walk_ancestors(..., start_parent_id=, start_parent_seq=)` for the
  in-flight fork row that hasn't been flushed yet — no mutation of the
  graph is needed mid-fork.

## 2. Numbering: independent per session

A forked session starts `seq` / `msg_idx` / `block_idx` from 0, not
from the parent's counter at the fork point. Context assembly walks
ancestors first (blocks with `seq <= parent_seq`), then the current
session, ordered by `(msg_idx, block_idx)` per segment.

### Considered

- **Inherit the parent's counter.** Rejected: the parent keeps
  growing its own counter post-fork, and inheriting would force a
  cross-session monotonic invariant that complicates crash recovery
  (orphan-`tool_use` truncate) and per-session `next_seq` bookkeeping.

### Consequences

- Git-branch semantics, no merge: a fork never sees the parent's
  post-fork blocks.
- Same-role adjacency across segment boundaries is possible (an
  ancestor segment ending in `user` followed by a fork-segment
  starting in `user`). Anthropic rejects consecutive `user` messages,
  so `Agent._ensure_messages` merges on first `run` after resume (the
  same path that handles trailing-user crashes).

## 3. Eager fork-row write

`Session.fork` writes the `sessions` row eagerly with `parent_id` /
`parent_seq` and a title derived from the ancestor chain's first user
text. This deviates from `Session.new`'s lazy-row strategy (no row
until the first user message).

### Considered

- **Reuse the lazy-row strategy.** Rejected: `cothis chat --resume
  <fork_id>` would `KeyError` until the user sent the first message
  because `Session.load` requires the row. The eager write is the
  smallest change that keeps the fork discoverable across processes.

### Consequences

- A fork that's created then immediately abandoned wastes one row
  (negligible cost; the catalog/history listing already tolerates
  empty sessions).
- The title may be empty if the ancestor chain had no user text (rare;
  the fork point had only assistant/tool blocks). Acceptable: the
  catalog displays `(no title)` for empty titles.

## 4. Visibility filter + leaf-only delete

`Session.load(cwd=...)` raises `KeyError` when the session's `cwd` is
neither `cwd` nor an ancestor of it — so `cothis chat --resume <id>`
won't accidentally resume a different project's session. The same
predicate drives `cothis history`'s listing.

`Session.delete` refuses non-leaf nodes with `SessionHasChildrenError`:
deleting a node with living children would orphan them (their
`parent_id` would dangle). Delete the children first.

### Considered

- **Cascade delete.** Rejected: a surprise loss of an entire subtree
  is worse than asking the user to delete children explicitly.
- **Soft delete (`state='archived'`).** Out of scope here; the
  per-block `state` column lands with skill-marked blocks (#30) and
  the deactivation path. Cold-DB delete lands in #36.

### Consequences

- `cothis delete` is the only way a session leaves the hot DB (other
  than archival, which physically moves rows to a cold DB). Manual
  `sqlite3 DELETE` would bypass the leaf-only check and could orphan
  children — documented in `delete_session`'s docstring.
- Hot-DB only in #35. Cold-DB delete (the same command, scoped to
  archived sessions) lands in #36.
