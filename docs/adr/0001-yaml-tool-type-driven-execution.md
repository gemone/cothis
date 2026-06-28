# YAML tool execution: type-driven argv vs shell

A YAML tool's `command:` field determines its execution mode by its YAML
type: a **list** runs in argv mode (`subprocess.run(list, shell=False)`,
each element one argv item, safe by default), a **string** runs in shell
mode (`subprocess.run(str, shell=True, executable=shell)`, supports
pipes/`&&`/redirection) and requires a `shell:` field naming the
interpreter. The executable (argv[0] or `shell:`) is gated via
`shutil.which`; if missing, the tool is silently not registered.

## Considered options

- **Type-driven (chosen)** — list=argv, string=shell. The field's YAML
  type *is* the mode selector; no separate `mode:` field.
- **`executable:` + `{exe}`** — one field naming the binary, command
  references it via `{exe}`. Rejected: conflates argv[0] and the shell
  interpreter under one field with two meanings.
- **Default system shell** — string commands default to `sh`/`cmd`.
  Rejected: violates AGENTS.md's "relying on a default you assumed rather
  than confirmed is silent breakage" — which shell, does it exist?
- **GA-style `if:` expressions** (the previous design) — per-branch
  predicate language with `has_shell()`/`has_exe()`. Rejected: 250-line
  recursive-descent parser for a feature no shipped tool needed beyond
  "split by OS", which `platforms:` map keys express directly.

## Consequences

- Simple commands (`git status`) written as YAML lists get execve-safety
  for free — no `shlex.quote`, values with spaces stay one argv item.
- Shell features (pipes, `&&`) require the author to declare a shell
  explicitly — an intentional speed bump, not a silent default.
- The `if:` expression language and its `has_shell`/`has_exe` predicates
  are gone; platform branching is now `platforms:` map keys only.
  Reintroducing conditional logic beyond platform identity requires
  revisiting this ADR.
