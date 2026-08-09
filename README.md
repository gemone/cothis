# cothis

A complete coding agent that talks to LLM providers through their direct
SDKs — Anthropic, OpenAI, Google (Gemini), and OpenRouter (via the OpenAI
SDK) — with a small ReAct-style loop that calls tools. Built-in tools are `fs.read`, `fs.list`, `fs.search`, `fs.create`, `fs.modify`, and `fs.delete`; you
can add more as YAML shell tools, Python `@tool` functions, or MCP servers
under `.agents/tools/` (see [Custom tools](#custom-tools)).

> **Architecturally inspired by [pi](https://github.com/earendil-works/pi).**
> cothis reuses pi's separation of an AI provider layer from the agent
> loop, translated to idiomatic Python and direct provider SDKs.

- `cothis ask "..."` — one-shot prompt, plain-text output (pipe-friendly).
- `cothis chat` — interactive multi-turn session in the Textual TUI (3-pane layout, worktree-aware sessions, `--legacy` for the old REPL).

Requires Python ≥ 3.14.

---

## Quick start (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run cothis ask "List the files in this repo and explain what each one does."
```

`openrouter` is the default provider and `openai/gpt-oss-120b` is the
default model — both can be overridden.

> **Heads-up:** OpenRouter's free / default models are **not stable under
> long context** — once a session's accumulated history (prompt + tool
> output + history) grows past a few thousand tokens, the default model
> starts dropping tool calls, truncating output, or returning errors. For
> multi-turn `chat` sessions or large file reads, switch to a more
> capable model via `-m` (e.g. `anthropic/claude-3.5-haiku`,
> `openai/gpt-4.1-mini`).

No key yet? Grab a free one
from OpenRouter's
[free-models collection](https://openrouter.ai/collections/free-models)
and pass any of those IDs via `-m`:

```bash
uv run cothis ask -m "meta-llama/llama-3.3-70b-instruct:free" "Hello"
```

## Installation

```bash
git clone https://github.com/gemone/cothis.git
cd cothis
uv sync
```

Run the CLI via `uv run cothis`, or install the entry point:

```bash
uv pip install -e .
cothis --help
```

## Usage

### `ask` — one-shot prompt

```bash
# Defaults: openrouter + openai/gpt-oss-120b
uv run cothis ask "What is 47 * 83?"

# Switch model on OpenRouter
uv run cothis ask -m anthropic/claude-3.5-haiku "Hello"

# Switch provider entirely
export OPENAI_API_KEY=sk-...
uv run cothis ask -p openai -m gpt-4.1-mini "Hello"

# ask prints plain text to stdout, so it composes with pipes:
uv run cothis ask "list three primes" | wc -l

# Show full traceback on error
uv run cothis --debug ask "hi"
# or
DEBUG=1 uv run cothis ask "hi"

# Show tool-call I/O (what the model sent, what the tool returned)
# without the openai/httpx HTTP noise
uv run cothis -v ask "list the files in src"
# or
VERBOSE=1 uv run cothis ask "list the files in src"
```

### `chat` — interactive multi-turn session (launches the TUI)

```bash
uv run cothis chat
```

`chat` launches the **Textual TUI** by default (#237): a focused
transcript shell — a full-height scrollable `ConversationView` with a fixed
composer dock below it (input + shortcut hint) and a one-line status bar.
Session navigation is transient (`/sessions` picker), never a sidebar; the
input holds focus so you can always type. Press `Ctrl+N` to create a new
session (worktree picker or current directory); type a prompt + `Ctrl+Enter`
to send. The worker streams deltas, tool-call cards, and interactive
questions (`AskUserModal`).

```bash
uv run cothis chat -m anthropic/claude-3.5-haiku
```

The same `--provider` / `-p`, `--model` / `-m` flags apply as to `ask`.
`--resume` also works in the TUI (auto-spawns the session on startup):

```bash
uv run cothis chat --resume <session_id>
```

`--legacy` keeps the old REPL (for `--skill` which the TUI doesn't yet
support):

```bash
uv run cothis chat --legacy --skill <name>
```

Run `cothis --help`, `cothis ask --help`, or `cothis chat --help` for
the full list of flags.


### `tui` — alias for the default TUI path

```bash
uv run cothis tui
```

Same as `chat` — launches the Textual TUI with Supervisor-backed session
spawn. Provided as a separate entrypoint for users who want the TUI
without the `chat` name. Same `--model` / `--provider` flags.


### Session management — `history` / `delete` / `archive`

```bash
uv run cothis history                 # list sessions visible from cwd
uv run cothis history <session_id>    # inspect one + pick resume/fork point
uv run cothis delete <session_id>     # remove a leaf session
uv run cothis archive <session_id>    # move a session to the cold store
```

`history` lists every session whose `cwd` is the current directory or an
ancestor of it (project-root sessions are visible from subdirectories).
With a session id: prints the message list numbered, then prompts for
an index — `r` (or Enter on the last) resumes at the end of `main`; a
number forks a new session from that message (git-branch semantics; the
original is untouched).

`delete` is leaf-only (refuses sessions with children). `archive` moves
the session to a monthly cold-store DB under `<db_path parent>/archive/`
and updates `archive/index.json`; `cothis archive restore <id>` brings
it back. Run `cothis history --help` / `cothis delete --help` /
`cothis archive --help` for the full flag list.


## Custom tools

cothis discovers shell tools as YAML files under `.agents/tools/` (relative
to the current working directory) or under `$COTHIS_HOME/tools/` (default
`~/.cothis/tools/`, overridable via the `COTHIS_HOME` environment variable)
for tools shared across all projects. Each file declares a `name`, a
`description` (shown to the LLM), and a `command:`. Two execution modes,
driven by the type of `command:`:

- **argv mode** — `command:` is a YAML list. The list is passed straight
  to `execve` (no shell), so each element is one argv item. Safe by
  default; `argv[0]` must be on PATH or the tool is not registered.
  ```yaml
  command: ["git", "status", "--short"]
  ```
- **shell mode** — `command:` is a string. A `shell:` field naming the
  interpreter (`bash`, `pwsh`, …) is supported; if omitted, cothis
  auto-selects the OS default (`sh` on POSIX). The string is passed to
  that shell, supporting pipes / `&&` / redirection. The shell must be
  on PATH or the tool is not registered.
  ```yaml
  shell: bash
  command: grep foo file | wc -l
  ```

  **`shell: cmd` is rejected.** cmd.exe cannot safely quote argument
  values — `&`, `|`, and `%VAR%` are live metacharacters that
  `subprocess.list2cmdline` does not neutralise, so a value like
  `foo&echo PWNED` would inject. Declare `shell: pwsh` (PowerShell —
  single-quote quoting is sound) or switch to argv mode (`command:
  [list]`, inherently safe). On Windows, authors who omit `shell:`
  must declare `shell: pwsh` or use argv mode; the compile error names
  both migration paths.

Arguments are declared under `args:` and substituted into the command at
`{arg_name}` placeholders. Only args actually referenced by the selected
command appear in the LLM schema (declared-but-unused args are dropped
with a warning).

Per-platform variants live under `platforms:` (keys: `linux`, `macos`,
`unix` = linux+macOS, `windows`). The top-level `command:` / `shell:` /
`args:` are the default; a matching platform entry overrides them.

```yaml
# .agents/tools/date/current.yaml
name: date.current
description: Get the current date and time as YYYY-MM-DD HH:MM:SS.
command: ["date", "+%Y-%m-%d %H:%M:%S"]
platforms:
  windows:
    shell: pwsh
    command: "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"
```

Tools whose executable (argv[0] or the declared `shell:`) is not on PATH
are silently not registered — the LLM never sees a tool it can't run here.

#### Multiple tools per file (`tools:`)

Related commands can share one YAML file under a top-level `tools:` list.
Each entry is a full tool spec (the same `name` / `command` / `args:` /
`platforms:` shape as above) and compiles to its own tool. Use this to
keep a family of subcommands together:

```yaml
# .agents/tools/uv.yaml — every uv subcommand as its own tool
tools:
  - name: uv.add
    description: Add a package to the project.
    command: ["uv", "add", "{package}"]
    args:
      - name: package
        type: str
        description: Package name (with optional version, e.g. "requests>=2").
  - name: uv.run
    description: Run a command in the project environment.
    command: ["uv", "run", "{cmd}"]
    args:
      - name: cmd
        type: str
```

Rules:

- A `tools:` list is the **only** key allowed at the top level of such a
  file — don't mix it with single-tool fields (`name:`, `command:`, …).
- Each entry is validated exactly like a standalone tool file, so a nested
  `tools:` inside an entry is rejected.
- Tool `name:`s must be unique within the file (duplicates raise, naming
  both entries). Gating still applies per tool: an entry whose executable
  is off PATH is skipped without affecting the others.

### Python tools (built-in `@tool`)

Built-in tools (`fs.read`, `fs.list`, `fs.create`, `fs.modify`, `fs.delete`) are defined with the
`@tool` decorator. It reads a Google-style docstring (summary → tool
description, `Args:` → per-arg descriptions) and `inspect.signature`
(types + defaults), then pre-builds an OpenAI schema so descriptions
reach the LLM (bypassing the cothis.ai schema path).

```python
from cothis import tool

@tool("greet.name")
def greet(name: str, formal: bool = False) -> str:
    """Greet someone by name.

    Args:
        name: The person to greet.
        formal: If true, use a formal greeting.
    """
    return f"Hello, {name}" if not formal else f"Good day, {name}."
```

Three forms: `@tool` (name from `__name__`), `@tool("ns.name")`
(positional name), `@tool(name=…, description=…)`. Per-arg descriptions
come from the docstring's `Args:` section.

*(Python-tool discovery — auto-scanning `.py` files under `.agents/tools/`
for `@tool`-decorated functions — shipped with PR #24. See
`CONTEXT.md` "Tool source" and ADR-0005 for the design, including the
deviation from PRD story 38.)*

### MCP servers (`type: mcp.stdio` / `type: mcp.http`)

An [MCP](https://modelcontextprotocol.io) server is another YAML tool
type. One declaration exposes *all* of the server's tools to the agent —
discovered at startup via the MCP protocol, dispatched over a persistent
session:

```yaml
# .agents/tools/browser.yaml — stdio: cothis spawns a subprocess
type: mcp.stdio
name: browser              # optional label (defaults to the file stem)
command: uvx               # the server executable
args: [browser-use, --mcp] # its arguments
env:                       # subprocess environment (secrets — never logged)
  BROWSER_USE_API_KEY: sk-...
```

```yaml
# .agents/tools/context7.yaml — http: cothis connects to a remote server
type: mcp.http
name: context7
url: https://mcp.context7.com/mcp
headers:                   # HTTP headers (secrets — never logged)
  Authorization: Bearer ...
```

cothis connects, lists the server's tools, and registers each one with a
**prefixed name**: a server `name: context7` exposing a `query-docs` tool
registers it as `context7.query-docs` (not the bare `query-docs`), so it
can't collide with a builtin or user tool of the same remote name. The
model sees and calls `context7.query-docs`; cothis strips the prefix back
to the bare name when dispatching to the server. The session is **managed
by the resource-handle subsystem** (ADR-0005): connected once at startup
(that connection is adopted, not wasted), then reclaimed when idle past
`keepalive` (default 600s) and re-acquired on the next call. Set
`pin: true` to keep a session alive for the whole run instead:

```yaml
type: mcp.stdio
name: browser
command: uvx
args: [browser-use, --mcp]
keepalive: 300   # reclaim the session after 300s of idleness (default 600)
pin: true        # keep the session alive until the agent exits (default false)
```

Only the **transport** differs between `mcp.stdio` (subprocess) and
`mcp.http` (remote); discovery, dispatch, and result handling are shared.
A server that fails to connect logs a warning (naming the command/url —
never the `env`/`headers` secrets) and is skipped — the rest of your tools
still load.

### Tool output format

When a tool returns a `dict` or `list`, cothis serialises it for the tool
message according to `COTHIS_TOOL_OUTPUT_FORMAT` (default `json`). `str`
results bypass formatting — text is text.

```bash
COTHIS_TOOL_OUTPUT_FORMAT=csv cothis ask "list files in src"
COTHIS_TOOL_OUTPUT_FORMAT=yaml cothis chat
```

CSV/TSV flatten nested dicts with dotted paths; bare lists of scalars
fall back to JSON. YAML handles every shape natively.

### Tool lifecycle hooks

Every tool passes through five lifecycle stages, from discovery to
dispatch. Register callbacks via decorator methods on the `ToolDef`
returned by `@tool`. Multiple callbacks per stage form a chain (see
`CONTEXT.md` "Tool lifecycle" for the full semantics):

| Stage | When | Input | Return | Chain | Exception → |
|---|---|---|---|---|---|
| `pre_load` | discovery, before registration | none | `False` = skip | short-circuit AND | skip, `on_error` |
| `after_load` | discovery, after `pre_load` passes | none | unused | all run | skip, `on_error` |
| `pre_execute` | `_execute`, before tool body | `args: dict` | `dict` (modified) | pipeline | error to LLM, `on_error` |
| `after_execute` | `_execute`, after tool body | `result, args` | `result` (modified) | pipeline | use original result, `on_error` |
| `on_error` | any prior stage raised | `exc, phase, args, result` | `None` (side-effect) | short-circuit on own exc | swallowed to `logger.debug` |

```python
from cothis import tool

@tool("git.commit")
def commit(message: str, amend: bool = False) -> str:
    """Create a git commit.

    Args:
        message: The commit message.
        amend: Whether to amend the previous commit.
    """
    ...

# pre_load × 2: environment gates (both must pass — short-circuit AND)
@commit.pre_load()
def check_git_on_path():
    """Gate 1: skip if git isn't on PATH."""
    import shutil
    return shutil.which("git") is not None

@commit.pre_load()
def check_repo_initialized():
    """Gate 2: skip if not inside a git repo."""
    from pathlib import Path
    return Path(".git").exists()

# after_load × 1: initialisation (side-effect only)
@commit.after_load()
def warm_branch_name():
    """Init: cache the current branch so commit doesn't re-discover it."""
    import subprocess
    _branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()

# pre_execute × 2: input pipeline (normalize → validate; each sees the previous output)
@commit.pre_execute()
def normalize_message(args):
    """Pipeline 1: strip trailing whitespace from the message."""
    args["message"] = args["message"].rstrip()
    return args

@commit.pre_execute()
def reject_empty_message(args):
    """Pipeline 2: reject empty messages (sees normalized output from pipeline 1)."""
    if not args["message"]:
        raise ValueError("commit message must not be empty")
    return args

# after_execute × 1: output pipeline
@commit.after_execute()
def truncate_verbose_output(result, args):
    """Pipeline: cap git output at 500 chars so it doesn't flood the context."""
    return result[:500] if isinstance(result, str) else result

# on_error × 1: failure observer (side-effect only; cannot recover)
@commit.on_error()
def log_to_telemetry(exc, phase, args, result):
    """Observer: record failures for debugging. Cannot recover."""
    print(f"git.commit failed at {phase}: {exc}")
```
```

All hooks are optional. A tool with no hooks dispatches exactly as before.


## Configuration

All configuration is via environment variables. The provider/model
pair controls *which* LLM you hit; the matching `*_API_KEY` env var is
read by cothis's provider layer based on the chosen provider.

### cothis

| Variable                   | Purpose                                   | Default                |
| -------------------------- | ----------------------------------------- | ---------------------- |
| `COTHIS_PROVIDER`          | provider key (see table below).   | `openrouter`           |
| `COTHIS_MODEL`             | Model identifier for the chosen provider. | `openai/gpt-oss-120b`  |
| `COTHIS_MAX_TOKENS`        | Override the output-token cap (otherwise resolved per-model from bundled litellm metadata). | *(unset)* |
| `COTHIS_TOOL_OUTPUT_FORMAT`| How `dict`/`list` tool results are serialised: `json`, `csv`, `tsv`, `yaml`. `str` results bypass this. | `json` |
| `COTHIS_AGENTS_PATTERN`    | Comma-separated filenames scanned for the AGENTS.md context block (first match per layer wins). | `AGENTS.md` |
| `COTHIS_AGENTS_ORDER`      | Ordered layer names for AGENTS.md assembly. Unknown names are skipped. | `user-agents,user-cothis,project` |
| `COTHIS_AGENTS_USER_GLOBAL`| If falsy (`0`/`false`/`no`/`off`), skip the user-global layers (`~/.agents`, `~/.cothis`). | `1` |
| `COTHIS_HOME`              | Root for user-global state: tools, skills, agents.db, supervisor.db. | `~/.cothis` |
| `COTHIS_SESSIONS_TYPE`     | Set to `project` to scope sessions to `<cwd>/.agents/sessions/session.db` (per-project layout). | *(unset — single global db)* |
| `COTHIS_SESSIONS_DIR`      | Override the sessions directory (split layout: `<dir>/session.db`). | *(unset)* |
| `COTHIS_PROFILE_STARTUP`   | If set, re-exec under `-X importtime` and exit before any third-party import. Startup-latency profiling. | *(unset)* |
| `DEBUG`                    | If truthy, show all debug logs + tracebacks. | *(unset)*           |
| `VERBOSE`                  | If truthy, show cothis tool-call I/O (no openai/httpx noise). | *(unset)* |

Command-line flags (`-p` / `-m` / `--max-tokens` / `--debug`) take precedence
over env vars, which take precedence over defaults.

### API keys

cothis reads the API key for the active provider from a well-known
env var. Set the one that matches your `COTHIS_PROVIDER`:

```bash
# OpenRouter (default)
export OPENROUTER_API_KEY=sk-or-...

# Mistral
export MISTRAL_API_KEY=...
export COTHIS_PROVIDER=mistral
export COTHIS_MODEL=mistral-small-latest

# OpenAI
export OPENAI_API_KEY=sk-...
export COTHIS_PROVIDER=openai
export COTHIS_MODEL=gpt-4.1-mini
```

## Supported providers

cothis talks to four providers through their direct SDKs:

| Provider key  | SDK              | API key env var        | Notes |
| ------------- | ---------------- | ---------------------- | ----- |
| `anthropic`   | `anthropic`      | `ANTHROPIC_API_KEY`    | Native Messages API — pass-through, no translation. |
| `openai`      | `openai`         | `OPENAI_API_KEY`       | Chat Completions; messages + tool schemas translated to/from the Anthropic shape. |
| `google`      | `google-genai`   | `GOOGLE_API_KEY`       | Gemini via the unified Google GenAI SDK. |
| `openrouter`  | `openai` (base URL override) | `OPENROUTER_API_KEY` | Default provider; any OpenRouter model id (e.g. `openai/gpt-oss-120b`). |
| `mistral`     | `openai` (base URL override) | `MISTRAL_API_KEY`    | OpenAI-compatible; routed through the OpenAI provider with Mistral's base URL. |

Each provider normalises its stream into the Anthropic Messages event
shape internally (see `src/cothis/ai/`), so the agent loop is
provider-agnostic.


## Model metadata

`max_tokens` (the output-token cap passed to the model) is resolved per
model from a bundled copy of litellm's
[`model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
at `src/cothis/data/model_prices.json`. The resolver matches the model
id, then `{provider}/{model}`, falling back to `8192` when neither is
present. No network call at runtime.

Override the resolved value with `--max-tokens` (or `COTHIS_MAX_TOKENS`)
on either `ask` or `chat`:

```bash
uv run cothis ask --max-tokens 4096 "..."
COTHIS_MAX_TOKENS=4096 uv run cothis chat
```

The bundled JSON is refreshed by the
[`update-model-prices`](https://github.com/gemone/cothis/actions/workflows/update-model-prices.yml)
workflow — a weekly Sunday 09:00 UTC cron (also runnable manually from
the Actions tab). When litellm's source changes, the workflow opens a PR
against `src/cothis/data/model_prices.json`; no PR when there's no diff.

**Known ceiling**: litellm's `litellm_provider` field names diverge from
cothis's provider keys (e.g. `together_ai` vs `together`). cothis does
not fuzzy-match on provider, so a model whose only key in litellm is
provider-prefixed under a *different* name resolves to the 8192
fallback. Override with `--max-tokens` in that case.


## Debug

By default, errors print as `Error: <message>` without a traceback.
Two logging levels:

- **`-v` / `--verbose`** — shows cothis tool-call I/O (`→ fs.read(path='...')`
  / `← fs.read: ...`) without openai/httpx noise. The day-to-day way to
  check what reached the model.
- **`--debug`** (`DEBUG=1`) — everything at DEBUG level (cothis + openai +
  httpx + httpcore) + full tracebacks on error.

```bash
uv run cothis -v ask "list files in src"
DEBUG=1 uv run cothis chat
```

## Development

The dev dependency group includes [`ruff`](https://docs.astral.sh/ruff/)
(formatting / lint), [`ty`](https://docs.astral.sh/ty/) (type checking),
and [`pytest`](https://docs.pytest.org/) (tests):

```bash
uv sync                              # install dev deps
uv run ruff check src/ tests/       # lint + import sorting
uv run ty check                     # type check
uv run pytest                       # unit tests (pure helpers, no network)
```

> **Renamed the project directory?** Entry-point scripts under `.venv/bin/`
> keep their old absolute shebang, so `.venv/bin/pytest` can silently target
> a stale Python from the previous path. Always invoke the tools through
> `uv run …` (`uv run pytest`, or `uv run python -m pytest` for the module
> form); after a rename, run `uv sync --reinstall` to regenerate the
> scripts (#286).

Tests cover the silent-breakage surfaces of the project: the
streaming chat path (by-index merge of streamed tool-call fragments,
best-effort JSON parse for on-screen display), the YAML tool loader
(command rendering, type-driven execution mode, per-arg description
carry-through to the LLM schema, malformed-YAML error paths), the
the `@tool` decorator (docstring parsing, schema construction, type mapping),
the ReAct loop (empty-message retry, tool-crash recovery), the tool
output formatter (json/csv/tsv/yaml), and the MCP adapter, stdio and
http transports (tool discovery, result normalisation, persistent-session
lifecycle, secret redaction). Tests run offline — no LLM calls. (YAML-tool tests do spawn
short-lived subprocesses like `echo`, and MCP tests run an in-memory
server; they never touch the network.)

## License

[Apache-2.0](LICENSE)
