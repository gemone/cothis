# TUI core — Textual 3-pane layout (design-review slice)

Issue #216 (Textual TUI + Durable Notify Bus) ships the TUI shell in
slices. The MVP slice (#228) is the **visual layout**: three panes
that render + accept the design-review pass. Real interactivity
(WS attach, notify_events polling, session reload) lands in
follow-ups.

This ADR records the decisions for the MVP slice + the boundaries of
what is intentionally deferred.

## 1. Textual, not prompt_toolkit

The current ``cothis chat`` uses ``prompt_toolkit.shortcuts.PromptSession``
+ ``rich.Live`` for a streaming REPL. It works for a single
prompt → single response but doesn't extend to a multi-pane layout
with a separate session list + tool-call cards.

### Considered

- **prompt_toolkit full-screen application.** Rejected: prompt_toolkit's
  full-screen API is verbose + the project's existing ``rich``-based
  rendering wouldn't carry over cleanly. Textual is built on ``rich``
  primitives + adds the layout / widget / messaging layer.
- **curses.** Rejected: cross-platform support (Windows) is fragile
  + the cost of reimplementing focus management + theming from
  scratch outweighs the dependency savings.
- **Rich + manual layout (no framework).** Rejected: rich renders
  content but doesn't manage widgets, focus, or events; building a
  real widget framework is out of scope for the project.

### Decision

``textual`` (v0.47+). Already transitively available via ``rich``;
declared as a direct dependency to make the import non-fragile.
Textual's App + Pilot (test harness) cover both production + testing.

## 2. Three-pane layout (SessionList / ConversationView / InputBar)

- **SessionList** (left, dock): sessions from the session table.
- **ConversationView** (center): scrollable Markdown for the
  streamed answer + tool-call cards.
- **InputBar** (bottom, dock): multiline input with send hotkeys.

### Considered

- **Single-pane REPL (current ``cothis chat``).** Rejected by the
  PRD: the user wants session switching, tool-call visibility, and
  a separate input box — none of which a single pane provides.
- **Two-pane (chat + sidebar).** Rejected: the input box docked
  inside the chat view fights with Markdown rendering for space; a
  dedicated bottom pane keeps them cleanly separated.
- **Four-pane (add a tool-call pane).** Deferred (not rejected): the
  MVP folds tool-call cards into the conversation as inline widgets
  (matches Claude Code's pattern). A separate pane can land if user
  feedback shows the cards are noisy.

### Decision

3 panes via Textual containers. Each pane is a custom widget class
so the layout is explicit in source.

## 3. Design-review slice — visual first, interactivity later

The PR ships the layout + a placeholder data source. The acceptance
criteria that require real data (send → WS → streamed answer, tool
calls as cards, historical reload) are **deferred to follow-ups**.

### Considered

- **Ship everything in one PR.** Rejected: the visual design + the
  interactive plumbing are independent axes. A reviewer who wants
  to tweak the layout shouldn't have to read 500 LOC of WS client +
  polling to give feedback.
- **Ship interactive first, visual later.** Rejected: a non-visual
  PR for #228 wouldn't give the reviewer anything to look at — the
  whole point of this slice is sign-off on the look.

### Decision

MVP scope:

- 3-pane shell with placeholder data (session-1, session-2).
- ``ConversationView.append_markdown`` API (forward-compatible with
  the future poller that calls it per ``assistant_delta``).
- ``InputBar.get_text`` / ``set_text`` API (forward-compatible with
  the future send-and-clear flow).
- Textual Pilot tests verify the panes exist + the APIs work.

Deferred (filed as follow-up when the time comes):

- Real WS attach to a worker (depends on the Supervisor's spawn
  path, deferred from #227 to #250).
- notify_events polling → ``ConversationView.append_markdown``.
- Tool-call card rendering (specific widget shape pending design
  sign-off).
- Historical session reload (needs Session.load integration).
- Send hotkey binding + WS ``run_turn`` forward.

## 4. Pilot tests, not unit tests of widgets

The Textual Pilot runs the app under a fake terminal driver so the
``query_one`` API + ``pause`` for settle behave like real
interactions. This catches layout regressions (e.g. a CSS rule that
hides a pane) without spinning up a real terminal.

### Considered

- **Pure widget unit tests (no Pilot).** Rejected: the widget's
  ``compose`` method only runs inside an active app; testing it
  without Pilot requires duplicating Textual's lifecycle.
- **No tests, manual review only.** Rejected: regression mode
  (pane silently disappears after a refactor) is invisible without
  a Pilot query.

### Decision

Pilot tests via ``app.run_test()``. Three tests cover: panes exist,
Markdown renders, input bar accepts text.

## 5. Single-session MVP

The app renders + accepts input for ONE session. Multi-session
concurrency (#230) is a later slice that adds session switching +
attach/detach.

### Considered

- **Multi-session from day one.** Rejected: adds WS multiplexing +
  per-session ConversationView state + attach/detach to the
  design-review slice — too much surface for a visual sign-off.

### Decision

One SessionList placeholder (shows the shape) + one
ConversationView + one InputBar. The selected-session model + the
attach/detach interaction lands with #230.

## 6. Out of scope for #228 (follow-up #252)

- **WS attach + ``run_turn`` forward** — depends on Supervisor
  spawn (deferred from #227 to #250).
- **notify_events polling → render** — needs a poller that calls
  ``append_markdown`` per ``assistant_delta``.
- **Tool-call cards** — specific widget shape pending design
  sign-off on the basic layout.
- **Historical session reload** — needs ``Session.load`` integration
  + ConversationView state management.
- **Send hotkey binding** — Textual ``Binding`` on the InputBar
  that calls ``get_text`` + clears.
- **Multi-session (#230)** — SessionList selection → reattach to
  a different worker.

All six are tracked as follow-up **#252** (blocked on #250).
