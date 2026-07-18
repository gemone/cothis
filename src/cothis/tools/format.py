"""Tool output formatting — serialise structured results for the tool message.

Extracted from the original ``tools.py``. ``format_tool_output`` is called by
``Agent._execute`` on every structured (dict/list) tool result; the format is
chosen via ``COTHIS_TOOL_OUTPUT_FORMAT`` (``json`` | ``csv`` | ``tsv`` |
``yaml``), defaulting to ``json``. Has no dependency on the rest of the tools
package — only stdlib (``csv``/``io``/``json``/``os``) plus ``yaml``.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from typing import Any


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with dotted key paths (``{"a": {"b": 1}}`` → ``{"a.b": 1}``).

    Non-dict values (including lists) are left as-is on the leaf — they'll be
    JSON-encoded per cell by the CSV writer.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def to_tabular(data: Any, delimiter: str) -> str | None:
    """Render ``data`` as CSV/TSV (``delimiter`` = ``,`` or ``\t``).

    Returns ``None`` when ``data`` isn't tabular (bare list of scalars, or a
    shape CSV can't express) — caller falls back to JSON. Nested dicts are
    flattened with dotted paths; nested lists/scalars are JSON-encoded per cell.
    """
    # Normalise to a list of single-row records.
    if isinstance(data, dict):
        rows = [flatten_dict(data)]
    elif isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        rows = [flatten_dict(r) for r in data]
    else:
        return None

    # Union of keys across rows preserves column discovery order.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        # Cells must be strings; JSON-encode non-scalars so values stay
        # model-parseable instead of Python repr.
        writer.writerow(
            {
                k: v if isinstance(v, str) else ("" if v is None else json.dumps(v))
                for k, v in row.items()
            }
        )
    return buf.getvalue().rstrip("\r\n")


def format_tool_output(result: Any) -> str:
    """Serialise a structured tool result for the tool message.

    Format is chosen via ``COTHIS_TOOL_OUTPUT_FORMAT`` (``json`` | ``csv`` |
    ``tsv`` | ``yaml``), defaulting to ``json``. Only ``dict``/``list`` results
    go through this path; ``str`` results bypass it (text is text).

    CSV/TSV fall back to JSON when the shape isn't tabular (bare list of
    scalars, deeply nested structures). YAML handles every shape natively.
    """
    fmt = os.environ.get("COTHIS_TOOL_OUTPUT_FORMAT", "json").lower()
    if fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        rendered = to_tabular(result, delim)
        if rendered is not None:
            return rendered
    if fmt == "yaml":
        # ``allow_unicode=True`` keeps CJK / emoji readable; ``sort_keys=False``
        # preserves insertion order so the model sees fields in the author's
        # intended order.
        return yaml.dump(result, allow_unicode=True, sort_keys=False).rstrip("\n")
    return json.dumps(result)
