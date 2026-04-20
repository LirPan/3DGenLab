from __future__ import annotations

from pathlib import Path

from genlab.utils import ensure_parent_dir


def write_dummy_cube_obj(output_path: str | Path) -> Path:
    """Write a valid OBJ cube mesh (8 vertices, 12 triangular faces)."""
    mesh_path = ensure_parent_dir(output_path)

    cube_obj = """# Dummy cube mesh
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 5 8
f 1 8 4
f 2 6 7
f 2 7 3
f 4 3 7
f 4 7 8
f 1 2 6
f 1 6 5
"""
    mesh_path.write_text(cube_obj, encoding="utf-8")
    return mesh_path
