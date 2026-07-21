"""Tests for ``cothis.tools.core`` griffe deferral (#81).

The 84 ms ``import griffe`` tax was previously paid at ``core.py``
module-load time. #81 moves it into ``_parse_docstring`` so paths
that don't decorate ``@tool`` functions don't pay it.

The local fix is verified via AST: no module-level ``import griffe``,
and ``_parse_docstring`` is the sole function with a function-level
``import griffe``. (The end-to-end CLI startup cost is bounded by a
separate concern — ``cothis.tools.__init__`` imports builtins which
trigger ``@tool`` decoration at module load. That cascade is out of
scope for #81; this PR closes the direct import.)
"""

from __future__ import annotations

import ast
from pathlib import Path


def _core_source() -> str:
    path = (
        Path(__file__).resolve().parent.parent
        / "src" / "cothis" / "tools" / "core.py"
    )
    return path.read_text(encoding="utf-8")


def test_griffe_not_imported_at_module_top() -> None:
    """Top-level imports must not include griffe."""
    tree = ast.parse(_core_source())
    module_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                module_imports.append(node.module)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Imports after the first class/def are inside that scope,
            # not module-level.
            break
    assert "griffe" not in module_imports


def test_griffe_imported_only_inside_parse_docstring() -> None:
    """``_parse_docstring`` is the sole function with ``import griffe``."""
    tree = ast.parse(_core_source())
    griffe_import_sites: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name == "griffe":
                            griffe_import_sites.append(node.name)
    assert griffe_import_sites == ["_parse_docstring"]
