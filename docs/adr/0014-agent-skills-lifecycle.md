# Agent Skills lifecycle

Agent Skills (#30) extend the agent's capability at run time with
on-disk skill packages the model can browse, activate on demand, and
retire. Each skill is a directory with a ``SKILL.md`` (YAML
frontmatter + body) plus optional resource files. This ADR records
the lifecycle decisions behind the four sub-systems — discovery,
activation, deactivation, and resume continuity — that PR2 (#30)
shipped across #68, #156-#158, #164, #167-#170, #71, #72, #74.

## 1. Progressive disclosure: catalog in system, body on activation

The agent is told *about* every skill but does not *load* any until
it asks. ``discover_skills`` runs at agent startup from three layers
(project ``.agents/skills/`` > user-cothis ``$COTHIS_HOME/skills/``
> user-agents ``~/.agents/skills/``), resolves cross-layer name
conflicts by shadowing (higher precedence wins; a WARNING names both
sources), and renders an ``<available_skills>`` block listing
``name: description`` for each. The block is appended to the system
prompt with ``cache_control: {type: ephemeral}``. ``load_skill(name)``
returns the skill body wrapped in ``<skill_content>`` plus a
``<skill_resources>`` listing — only when the model invokes it.

### Considered

- **Eager-load every skill into the system prompt.** Rejected: 10
  skills at 4KB each = 40KB of system prompt per turn, regardless of
  relevance. Progressive disclosure keeps the catalog constant per
  agent run (cache-friendly) and pays the body cost only when
  actually needed.
- **Per-skill system-prompt slot with model-driven lazy fetch.**
  Rejected: there is no native "fetch this skill" instruction in the
  Anthropic API; the catalog-as-text + ``load_skill`` tool is the
  closest analogue. The tool name and ``<available_skills>`` framing
  are the protocol.
- **Ad-hoc YAML parsing per skill.** Rejected: malformed frontmatter
  or invalid encoding would silently corrupt the system prompt
  (#166). Two-tier UTF-8 → locale-fallback → skip-with-WARNING decode
  (matching ``agent._read_text`` for AGENTS.md) keeps prompt-bound
  text strict.

## 2. Persist-time tagging via ``skill_marker``

A skill's blocks carry ``skill=name`` on disk so later passes
(deactivation, resume rebuild, projection skip) can find them without
parsing the tool input. The marker is set at enqueue time on
``tool_use`` blocks for tools that declare ``skill_marker=True`` on
their ``ToolDef`` (``load_skill`` and ``deactivate_skill`` both do).
The matching ``tool_result`` block is tagged the same way before
flush. ``_cothis_skill`` is a private key on the in-memory block
dict; ``_block_to_row`` reads it into ``BlockRow.skill``;
``_request_messages`` strips every ``_cothis_*`` key before send so
the private marker never reaches the model.

### Considered

- **Tag at session level via a parallel table (block_id → skill).**
  Rejected: a second table doubles the write surface and adds a JOIN
  on every read. The block dict already flows through one enqueue
  path; piggybacking the marker on it is one extra dict key.
- **Retroactive UPDATE after ``load_skill`` returns.** Rejected: the
  tool's ``tool_use`` and ``tool_result`` blocks arrive in separate
  drain cycles. A retroactive UPDATE would need to track in-flight
  blocks by id and re-UPDATE on the second half. Persist-time tagging
  marks both halves at enqueue — one write per block, no second pass.
- **Tag with a separate ``tool_name``-based lookup.** Rejected: a
  skill-marker tool's ``input.name`` is the skill name, but tying
  tagging to ``tool_name == "load_skill"`` couples the persistence
  layer to the skills module. The ``skill_marker`` flag is the
  protocol-level declaration; the agent honours it without knowing
  what the tool does.

## 3. ``.session`` handler as a tool-protocol extension

Some tools need to mutate session state (``load_skill`` adds the
skill to ``active_skills``; ``deactivate_skill`` archives its blocks).
This is a tool-protocol concern, not a tool-result concern — handlers
decide via session state (catalog membership), never via text
inspection of the result. ``ToolDef`` gains two flags:
``inject_session=True`` causes ``Agent._execute_tool`` to pass the
live ``Session`` as a ``_session`` kwarg (also stripped from the
LLM-facing schema so the model never sees it); ``skill_marker=True``
opts the tool into persist-time tagging (§2). The ``.session``
handler is the post-execution hook the tool may register to act on
the result + session together.

### Considered

- **Parse tool result text for state intent.** Rejected: prompt
  injection via a malicious skill body could forge "I activated X"
  text and the parser would dutifully set state. State mutations
  must come from the tool definition, not the model's output.
- **Special-case ``load_skill`` / ``deactivate_skill`` in the agent.**
  Rejected: hardcoding skill-tool names in ``Agent`` would re-introduce
  the per-tool branching the unified dispatch removed (ADR-0004).
  The ``.session`` handler keeps ``Agent`` tool-agnostic.
- **Pass the session to every tool.** Rejected: most tools don't
  need session state and shouldn't depend on it. ``inject_session``
  is opt-in per tool.

## 4. Two-half ``mark_archived`` (Delete strategy, Summarize deferred)

Deactivation archives a skill's tagged blocks so context assembly
skips them on later turns. The Delete strategy (drop the blocks from
the model's view) is implemented; Summarize (produce a summary before
archiving) is deferred — a skill declaring ``deactivation: summarize``
in its YAML falls back to Delete with a logged WARNING.

Archival is a four-part mechanism:

- **Half A (#167):** ``_deactivate_skill(name)`` adds the name to
  ``_archived_skills``. ``Session.append_message`` stamps
  ``_cothis_state='archived'`` on blocks whose ``_cothis_skill``
  matches, so future writes for that skill land archived directly.
- **Half B (#168):** the same call posts an ``_ArchiveOp`` to the
  write queue. The consumer thread runs ``Storage.archive_skill_blocks``
  (``UPDATE blocks SET state='archived' WHERE session_id=? AND skill=?``)
  so historical and in-flight rows land archived regardless of drain
  timing. FIFO queue ordering guarantees the UPDATE runs strictly
  after any in-flight INSERT.
- **In-memory walk (#169):** ``_deactivate_skill`` also walks the
  ``messages`` mirror and stamps matching blocks. The next request's
  ``_request_messages`` projection filters blocks with
  ``_cothis_state == "archived"`` — paired-skip holds because both
  members of a ``tool_use`` / ``tool_result`` pair tagged for the
  same skill land archived together.
- **Resume rebuild (#71):** ``Session.load`` replays the
  ``load_skill`` / ``deactivate_skill`` ``tool_use`` sequence per
  skill (most-recent wins) to derive ``active_skills``. The
  ``state`` column is *not* consulted — it is the derived product of
  deactivation; the load/deactivate sequence is the source of truth.
  Skills that vanished from disk since the last session are dropped
  from the active set, their tagged blocks archived via Half B, and
  a WARNING logged per vanished skill — the session continues with
  the surviving skills.

### Considered

- **Physical ``DELETE`` of archived blocks.** Rejected: history is
  lost; the user cannot inspect what was deactivated. ``state='archived'``
  is a soft flag — the blocks stay on disk, only hidden from the
  model. ``cothis`` CLI inspection tools see them with ``state``
  populated.
- **Single-pass retroactive UPDATE only (skip Half A).** Rejected:
  the queue is async; between calling ``_deactivate_skill`` and the
  consumer draining the UPDATE, new blocks for that skill could be
  enqueued with ``state=None`` and the UPDATE would miss them. Half
  A guarantees future writes are marked at enqueue; Half B catches
  the history.
- **Tag ``archived`` blocks at projection time only (skip the
  ``state`` column).** Rejected: the projection runs every turn and
  has no memory of past deactivations across resume. Persisting
  ``state`` keeps archival durable; the projection's job is just to
  honour it.
- **Block a deactivation if it would orphan a ``tool_use``.**
  Rejected: Half A + Half B both tag all blocks for the same skill
  together — paired-skip is structural, not enforced per-call. The
  invariant holds because the trigger is the skill name, not the
  block id.

## 5. Active-skills footer on the latest user-typed message

When ``active_skills`` is non-empty, ``_request_messages`` appends an
``<active_skills>`` text block listing the active skill names and
reminding the model about ``deactivate_skill``. The footer is
projection-only (never persisted); it walks backwards past trailing
``tool_result``-only user messages (the post-tool-call state) so it
attaches to the latest user-typed text message — appending a text
block to a tool-result-only message would corrupt Anthropic's
tool-flow shape (#72 review caught this).

### Considered

- **Footer in the system prompt.** Rejected: the system prompt is
  cache-controlled and constant per agent run. A turn-by-turn footer
  there would invalidate the cache every turn. The latest-user-message
  slot is fresh per turn without that cost.
- **Footer on every user message.** Rejected: historical user
  messages are immutable; mutation would confuse the model and
  complicate resume.

## Sibling sub-issues

- #68 — skill discovery (3-layer, shadow resolution)
- #156 — schema v2 (``skill`` / ``state`` columns)
- #157 — tool-protocol ``.session`` handler + ``inject_session`` +
  ``skill_marker``
- #158 — ``load_skill`` tool + ``active_skills`` session state
- #164 — persist-time skill tagging
- #167 — ``deactivate_skill`` tool + Half A in-memory archived set
- #168 — queued UPDATE for archived state (Half B)
- #169 — context-assembly skip ``state='archived'`` blocks (read side)
- #170 — ``deactivation: summarize`` declaration + Delete fallback
- #71 — resume rebuild of ``active_skills`` + vanished-skill archival
- #72 — ``<active_skills>`` footer on latest user-typed message
- #74 — ``/reload-skills`` slash command (catalog rebuild +
  vanished-archival)
