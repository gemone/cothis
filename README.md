# cothis

A basic coding agent built on top of
[`any-llm`](https://github.com/mozilla-ai/any-llm) — talk to any LLM
provider through a single interface, with a small ReAct-style loop that
call tools. Built-in tools are `fs.read`, `fs.write`, and `fs.dir`; you
can add more as YAML shell tools or Python `@tool` functions under
`.agents/tools/` (see [Custom tools](#custom-tools)).

- `cothis ask "..."` — one-shot prompt, plain-text output (pipe-friendly).
- `cothis chat` — interactive multi-turn session, streamed Markdown output.

Requires Python ≥ 3.14.

---

## Quick start (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run cothis ask "List the files in this repo and explain what each one does."
```

`openrouter` is the default provider and `openai/gpt-oss-120b` is the
default model — both can be overridden. No key yet? Grab a free one
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

### `chat` — interactive multi-turn session

```bash
uv run cothis chat
```

`chat` reuses one agent across turns, so conversation history
accumulates. Each turn's final answer is streamed token-by-token and
rendered as Markdown; tool calls (`fs.read`, `fs.dir`, `fs.write`, and
any custom tools you've added) are printed inline as `calling <name>(<args>)`. Exit
with `Ctrl-D` or `Ctrl-C`.

The same `--provider` / `-p`, `--model` / `-m`, and `--max-iterations`
flags apply as to `ask`:

```bash
uv run cothis chat -m anthropic/claude-3.5-haiku
```

Run `cothis --help`, `cothis ask --help`, or `cothis chat --help` for
the full list of flags.


## Custom tools

cothis discovers shell tools as YAML files under `.agents/tools/` (relative
to the current working directory). Each file declares a `name`, a
`description` (shown to the LLM), and a `command:`. Two execution modes,
driven by the type of `command:`:

- **argv mode** — `command:` is a YAML list. The list is passed straight
  to `execve` (no shell), so each element is one argv item. Safe by
  default; `argv[0]` must be on PATH or the tool is not registered.
  ```yaml
  command: ["git", "status", "--short"]
  ```
- **shell mode** — `command:` is a string. A `shell:` field naming the
  interpreter (`bash`, `pwsh`, …) is **required**; the string is passed
  to that shell, supporting pipes / `&&` / redirection. The shell must
  be on PATH or the tool is not registered.
  ```yaml
  shell: bash
  command: grep foo file | wc -l
  ```

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

### Python tools (built-in `@tool`)

Built-in tools (`fs.read`, `fs.dir`, `fs.write`) are defined with the
`@tool` decorator. It reads a Google-style docstring (summary → tool
description, `Args:` → per-arg descriptions) and `inspect.signature`
(types + defaults), then pre-builds an OpenAI schema so descriptions
reach the LLM (bypassing any-llm's lossy `callable_to_tool`).

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

*(Discovery of user-authored `.py` files under `.agents/tools/` is the
next slice — issue #1, stories 33–35. The decorator is ready; the loader
isn't.)*

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


## Configuration

All configuration is via environment variables. The provider/model
pair controls *which* LLM you hit; the matching `*_API_KEY` env var is
read automatically by `any-llm` based on the chosen provider.

### cothis

| Variable                   | Purpose                                   | Default                |
| -------------------------- | ----------------------------------------- | ---------------------- |
| `COTHIS_PROVIDER`          | any-llm provider key (see table below).   | `openrouter`           |
| `COTHIS_MODEL`             | Model identifier for the chosen provider. | `openai/gpt-oss-120b`  |
| `COTHIS_TOOL_OUTPUT_FORMAT`| How `dict`/`list` tool results are serialised: `json`, `csv`, `tsv`, `yaml`. `str` results bypass this. | `json` |
| `DEBUG`                    | If truthy, show all debug logs + tracebacks. | *(unset)*           |
| `VERBOSE`                  | If truthy, show cothis tool-call I/O (no openai/httpx noise). | *(unset)* |

Command-line flags (`-p` / `-m` / `--debug`) take precedence over env
vars, which take precedence over defaults.

### API keys

`any-llm` reads the API key for the active provider from a well-known
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

cothis can talk to any provider supported by
[`any-llm`](https://docs.mozilla.ai/providers) — OpenAI, Anthropic,
Mistral, OpenRouter, Ollama, Gemini, Groq, and many more.

For the full list and each provider's API key env var / capabilities,
see the [any-llm providers page](https://docs.mozilla.ai/providers).


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

Tests cover the silent-breakage surfaces of the project: the
streaming chat path (by-index merge of streamed tool-call fragments,
best-effort JSON parse for on-screen display), the YAML tool loader
(command rendering, type-driven execution mode, per-arg description
carry-through to the LLM schema, malformed-YAML error paths), the
`@tool` decorator (docstring parsing, schema construction, type mapping),
the ReAct loop (empty-message retry, tool-crash recovery), and the tool
output formatter (json/csv/tsv/yaml). Tests run offline — no LLM calls.
(YAML-tool tests do spawn short-lived subprocesses like `echo`; they
never touch the network.)

## License

[Apache-2.0](LICENSE)
