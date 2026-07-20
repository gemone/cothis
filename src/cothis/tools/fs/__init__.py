"""``cothis.tools.fs`` — filesystem tool subpackage.

Holds the codex ``apply_patch``-based write/read/list/search tools. The
subpackage is built up incrementally per #46's vertical slices:

- slice 1 (#48): ``patch.py`` deep module (this package's first resident).
- slice 2 (#49): tool runtime context (``WORKDIR`` ``ContextVar``).
- slice 3 (#50): ``fs.read`` multi-path migration.
- slice 4–6 (#51–#53): ``fs.write`` signature + cwd boundary + atomicity.
- slice 7–8 (#54–#55): ``fs.list`` (fd backend) + ``fs.search`` (rg backend).

Real exports are added slice by slice; this module stays minimal for now.
"""
