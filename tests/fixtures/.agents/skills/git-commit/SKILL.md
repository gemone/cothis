---
name: git-commit
description: Write a conventional-commit message for staged changes.
deactivation: delete
---

Write a conventional commit message for the currently staged diff.

Format the subject line as `type(scope): imperative subject`:
- `type` is one of feat, fix, refactor, docs, test, chore, perf.
- `scope` names the module or area touched; omit it when no single scope fits.
- The subject uses imperative mood, lowercase, no trailing period, capped at 72 characters.

Use the body to explain *why* the change matters rather than restate the diff.
Reference the iteration or roadmap item that motivated the work (for example a
task slug or issue number) so each commit traces back to its origin.

Never append sign-off lines or `Co-Authored-By:` trailers. Prepare the message
only; run `git commit` solely when explicitly asked.
