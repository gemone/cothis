# SessionWorker — child process + WebSocket run_turn protocol

Issue #216 (Textual TUI + Durable Notify Bus) separates the Agent
loop from the UI: each session runs in its own process (the
SessionWorker), and the TUI drives workers over a WebSocket. This
ADR records the design of the worker-side interface for the MVP slice
(#225):

- The worker owns one ``Agent`` + ``Session``.
- A loopback WebSocket carries control messages between the TUI and
  the worker.
- A bearer token on the ``Authorization`` header authenticates each
  connection.

## 1. WebSocket, not stdin/stdout or a binary IPC

The TUI needs to attach + detach without losing state, multi-plex
across sessions, and stream assistant deltas back in real time. A
full-duplex channel is the natural fit.

### Considered

- **stdin / stdout pipes.** Rejected: attaching the TUI to an
  already-running worker (detached-then-re-attached) is impossible
  without an out-of-band channel. Pipes are also process-coupled —
  if the TUI crashes, the pipe dies with it.
- **Named pipes / Unix domain sockets with ad-hoc framing.**
  Rejected: each platform has its own quirks (Windows lacks POSIX
  signals; Unix has no native named-pipe equivalent to NT's). A
  single ``websockets`` library standardises both.
- **HTTP long-polling.** Rejected: streaming assistant deltas over
  long-polling adds latency + complexity (request cadence, ordering)
  that WS frames solve for free.

### Decision

``websockets`` library. Already transitively available via ``mcp``;
declared as a direct dependency to make the import non-fragile. Runs
on asyncio, integrates with the existing event loop without thread
boundaries.

## 2. Loopback-only bind

The worker binds ``127.0.0.1`` only. Remote access is not a goal: the
Supervisor is a parent process on the same host, and exposing a
WebSocket that drives an Agent with file-write tools to the network
is a privilege-escalation surface.

### Considered

- **Bind ``0.0.0.0`` with strong TLS + auth.** Rejected: the cost of
  cert management + the blast radius if the cert leaks (full Agent
  control across the network) outweighs the use case. Local-first is
  the project stance.
- **Unix domain sockets.** Rejected (for now): would be cleaner than
  TCP, but ``websockets`` doesn't natively support UDS, and the
  port-randomisation model is sufficient at this scale. Revisit if a
  use case appears (e.g. multi-user host).

### Decision

``127.0.0.1:0`` (random port). The Supervisor receives the chosen port
when the worker starts; the TUI receives it from the Supervisor.

## 3. Bearer token over ``Authorization`` header

Each worker spawns with a cryptographically random token
(``secrets.token_urlsafe(32)``). The TUI presents it as
``Authorization: Bearer <token>`` on the WS handshake. Missing or
wrong token → HTTP 401 + handshake rejected.

### Considered

- **Token in the URL query string (``?token=...``).** Rejected: URLs
  leak into proxy logs, server logs, ``ps`` output. Headers are
  internal to the WS frame.
- **No auth (loopback-only is enough).** Rejected: any local process
  that can guess the port can then drive the Agent. The token makes
  port-scanning + unauthorized connections fail-closed.
- **mTLS.** Rejected at this scale: cert generation per worker is
  overkill for a single-host loopback. The bearer token gives
  near-equivalent guarantees with one ``secrets.token_urlsafe`` call.

### Decision

Bearer token. The supervisor hands the token to the TUI when it
spawns the worker; the TUI stores it in memory (never on disk).

## 4. Control message envelope

JSON envelopes with ``type`` discriminator. Bi-directional.

From TUI → worker:

- ``run_turn`` (``prompt: str``) — drive ``Agent.run_stream``.
- ``attach_input`` / ``detach_input`` — terminal attach (real
  semantics land with #230; accepted + ignored for now).
- ``shutdown`` — close the worker cleanly.
- ``ping`` — health check.

From worker → TUI:

- ``assistant_delta`` (``text: str``) — one ``TextDelta`` from the
  agent stream.
- ``tool_call_started`` (``tool: str``, ``arguments: dict``) —
  emitted before each tool dispatch.
- ``tool_call_result_pointer`` — reserved for the durable-pointer
  pattern (#224's notify bus already records this; the WS variant
  is for real-time UI feedback, not the source of truth).
- ``pong`` — ping reply.
- ``error`` (``message: str``) — recoverable errors (bad JSON,
  unknown message type, agent exception during ``run_turn``).

### Considered

- **Binary MsgPack framing.** Rejected: the volume is low (tens of
  messages per turn); JSON is debuggable from a ``wireshark`` capture
  without a decoder.
- **Separate WS per message class (one for control, one for stream).**
  Rejected: ``websockets`` multiplexes fine on one connection, and
  ordering matters (``run_turn`` must reach the worker before its
  streamed deltas can be interpreted).

### Decision

One WS connection per session; JSON envelopes with ``type``.

## 5. Worker wraps ``Agent.run_stream`` — not the whole Agent

The worker calls ``Agent.run_stream`` directly. It does not own a
secondary "agent runner" abstraction — the Agent's stream API is
already the right shape (``AsyncIterator[str | ToolCallEvent]``).

### Considered

- **New ``Agent.run_for_worker(prompt)`` method.** Rejected: would
  duplicate ``run_stream``'s contract. The worker translates between
  ``Agent``'s Python iterator and the WS frame format; the Agent
  shouldn't know it's running under a worker.
- **Worker that calls ``Agent.run`` (non-streaming).** Rejected:
  defeats the point of the WS — the TUI would see no deltas until the
  turn completes.

### Decision

Worker calls ``Agent.run_stream`` + forwards each yielded event as
one WS message. The translation is mechanical: ``str`` →
``assistant_delta``, ``ToolCallEvent`` → ``tool_call_started``.

## 6. anyio — implemented (#248)

The worker's WS surface sits behind a ``WSTransport`` seam
(``cothis.ws``), and the worker itself uses anyio's backend-neutral
primitives (``anyio.fail_after`` for the turn timeout). The criterion
deferred from #225 — *"anyio WS wrappers are isolated and unit-testable
(mock the transport)"* — is satisfied by the seam: production wires the
``websockets`` adapter; ``tests/test_ws_transport.py`` injects a
``FakeTransport`` / ``FakeConnection`` so dispatch, auth, and the timeout
cancel scope run with **no socket bound**.

anyio is the runtime abstraction, not the WS implementation — anyio ships
no websocket primitive. The honest reading of "anyio WS wrappers" is
therefore two things, both delivered:

- a transport **seam** (``WSTransport`` protocol + ``Connection`` protocol)
  that isolates the I/O surface so it can be mocked, and
- **anyio primitives** in the worker (``fail_after`` cancel scope instead of
  ``asyncio.timeout``) so the worker names no ``asyncio`` symbol — if a
  second backend appears, only the adapter in ``cothis.ws`` changes.

The ``websockets`` library is asyncio-only, but it runs unchanged under
anyio's asyncio backend (the loop is the same). anyio is declared as a
direct dependency (it was already present transitively via httpx/mcp).

### Considered

- **anyio-based WS wrappers from day one.** Rejected (originally): the
  project runs on asyncio today, so it read as speculative. Re-evaluated
  in #248: the *seam* earns its keep independently of anyio — it makes the
  worker's dispatch logic hermetically testable — and the maintainer
  (decision 2026-07-24) authorised the introduction as a deliberate,
  subjective call.
- **anyio on trio backend.** Still not done: trio would require rewriting
  ``Agent.run_stream``'s internals. The current change is structured so
  that work, if it ever appears, is local to ``cothis.ws``.

### Decision

``WSTransport`` seam + anyio primitives in the worker (#248). The Agent's
stream API remains the other abstraction boundary; the WS layer is
mechanical translation, now mockable through the seam.

## 7. Out of scope for #225

- **Supervisor process** — #227. The Supervisor spawns workers,
  tracks status, restarts on crash with ``always_backoff``. This
  ADR covers only the worker-side interface.
- **Terminal attach** — #230. ``attach_input`` / ``detach_input``
  messages are accepted + ignored; the real TTY-forwarding logic is
  multi-session concurrency.
- **ask_user / interactive tool flow** — #229. The worker forwards
  ``tool_call_started`` for every tool, including ``ask_user``; the
  TUI decides how to render + collect user input.
- **Multi-session concurrency** — #230. The MVP runs one worker at a
  time per TUI; the Supervisor handles the multi-worker case.
