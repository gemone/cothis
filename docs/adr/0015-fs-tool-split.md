# fs.write → fs.create / fs.modify / fs.delete tool split

The original ``fs.write`` tool used the codex ``apply_patch`` format
— a single multi-op document that could add, update, and delete
files atomically. While powerful, the format was unrecognizable to
non-Claude/GPT-4 models (DeepSeek, smaller open-source models).
Every first-write attempt took 3–4 failed round-trips before the
model either learned the format from the project's own tests or
gave up (#190 added examples; #198 identified the deeper problem).

This ADR records the decision to replace ``fs.write`` + its parser
(``patch.py``, 433 LOC) with three single-purpose tools whose
schemas are immediately usable by any model.

## 1. Three tools, not one

``fs.create(path, content)``, ``fs.modify(path, start_line, end_line,
content)``, ``fs.delete(path)``. Each tool does one operation on
one file. The model picks the tool based on the operation type;
there is no format to learn.

### Considered

- **Keep ``fs.write`` + add ``fs.create``/``fs.delete`` as simpler
  alternatives.** Rejected: the model still has to choose between
  ``fs.write`` and ``fs.create`` for the same operation, adding
  decision overhead. Two paths to the same outcome is confusing.
  The grill-with-docs session (#198) decided on a clean break: no
  ``fs.write`` at all.

- **Replace ``apply_patch`` with unified diff.** Rejected: unified
  diff is still a format the model has to produce correctly. Line
  numbers (from ``fs.read``) are a simpler anchor than diff hunks
  because the model already knows them.

- **Keep ``apply_patch`` as an escape hatch for multi-file atomic
  writes.** Rejected by the user during the grill session: "不要
  write 了呀，拆分三个原子了" — three atomic per-file tools, no
  escape hatch. Multi-file atomicity is lost but acceptable (the
  model works file-by-file; ``fs.write`` was rarely used for true
  multi-file batches).

## 2. ``fs.modify`` uses line numbers, not ``str_replace``

``fs.modify(path, start_line, end_line, content)`` replaces lines
``start_line`` through ``end_line`` (1-based, inclusive) with
``content``. The model gets line numbers from ``fs.read`` — no
content matching, no diff hunks, no context lines.

### Considered

- **``str_replace`` style (``old_text``, ``new_text``).** Rejected
  by the user: "需要包含行号" — line numbers are the anchor.
  ``str_replace`` fails on whitespace differences and ambiguous
  matches. Line ranges are unambiguous (the model knows exact
  positions from ``fs.read``).

- **Full-content overwrite (``fs.modify(path, content)``).**
  Rejected: wastes tokens for small edits (rewriting a 500-line
  file to change 1 line). Line-range editing is more economical.

## 3. Per-file, not multi-file atomicity

Each tool operates on one file. There is no multi-file batch or
atomic rollback. If the model creates ``a.py`` successfully but
``b.py`` fails, ``a.py`` stays on disk.

### Considered

- **Transaction/batch tool.** Rejected: re-introduces the complexity
  that the split was designed to eliminate. The model's natural
  workflow is sequential (one file at a time), and a failed write
  is visible on the next turn — the model reads the error and
  retries. Atomicity was a ``fs.write`` feature, not a requirement.

## 4. Clean break, no deprecation

``fs.write`` and ``patch.py`` are deleted entirely. No deprecated
alias, no gradual rollout. Existing sessions that reference
``fs.write`` receive an "unknown tool" error; the user re-runs with
the new tools.

### Considered

- **Keep ``fs.write`` as a deprecated alias.** Rejected: single-
  owner codebase (gemone/cothis), no external consumers. An alias
  would confuse the model (two paths to the same outcome) and keep
  700+ lines of dead code (``write.py`` + ``patch.py`` + their
  tests) in the repo indefinitely.

## Sibling sub-issues

- #206 — ``fs.create`` + ``fs.delete`` (sub-issue A)
- #207 — ``fs.modify`` (sub-issue B)
- #208 — remove ``fs.write`` + ``patch.py``, migrate tests
  (sub-issue C)
- #209 — this ADR + CONTEXT.md (sub-issue D)
