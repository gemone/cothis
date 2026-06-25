# cothis

A basic coding agent built on top of
[`any-llm`](https://github.com/mozilla-ai/any-llm) — talk to any LLM
provider through a single interface, with a small ReAct-style loop that
call tools. Built-in tools are `fs.read` and `fs.write`; you can add more
as YAML shell tools under `.agents/tools/` (see [Custom tools](#custom-tools)).

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
git clone <this-repo>
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
```

### `chat` — interactive multi-turn session

```bash
uv run cothis chat
```

`chat` reuses one agent across turns, so conversation history
accumulates. Each turn's final answer is streamed token-by-token and
rendered as Markdown; tool calls (`fs.read`, `fs.write`, and any YAML
tools you've added) are printed inline as `calling <name>(<args>)`. Exit
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
`description` (shown to the LLM), and a `command:`. A flat `command:` runs
on every platform; a `command:` list lets you pick a branch per platform
or condition via GitHub-Actions-style `if:` expressions, with an optional
`default:` fallback.

```yaml
# .agents/tools/date/current.yaml
name: date.current
description: Get the current date and time as YYYY-MM-DD HH:MM:SS.
command:
  - if: runner.os == 'Linux' || runner.os == 'macOS'
    run: date "+%Y-%m-%d %H:%M:%S"
  - if: runner.os == 'Windows'
    shell: pwsh
    run: Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
```

Tools for which no branch matches the current platform (and no
`default:`) are silently not registered — the LLM never sees a tool it
can't run here.
Arguments are declared under `args:` and substituted into the command at
`{arg_name}` placeholders; `has_shell('pwsh')` / `has_exe('git')` in `if:`
gate a branch on a binary actually being on PATH.


## Configuration

All configuration is via environment variables. The provider/model
pair controls *which* LLM you hit; the matching `*_API_KEY` env var is
read automatically by `any-llm` based on the chosen provider.

### cothis

| Variable           | Purpose                                   | Default                |
| ------------------ | ----------------------------------------- | ---------------------- |
| `COTHIS_PROVIDER`  | any-llm provider key (see table below).   | `openrouter`           |
| `COTHIS_MODEL`     | Model identifier for the chosen provider. | `openai/gpt-oss-120b`  |
| `DEBUG`            | If truthy (`1`/`true`/`yes`/`on`), show full tracebacks on error. | *(unset)* |

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
Set `DEBUG=1` or pass `--debug` to see the full stack:

```bash
DEBUG=1 uv run cothis ask "hi"
# or
uv run cothis --debug ask "hi"
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
best-effort JSON parse for on-screen display) and the YAML tool loader
(command rendering, the GitHub-Actions-style `if:` evaluator, per-arg
description carry-through to the LLM schema, and malformed-YAML error
paths). Tests run offline — no LLM calls. (YAML-tool tests do spawn
short-lived subprocesses like `echo`; they never touch the network.)

## License

[Apache-2.0](LICENSE)
