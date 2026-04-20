from __future__ import annotations

from pathlib import Path


def calculate_mesh_metrics(mesh_path: str) -> dict:
    import trimesh

    mesh_file = Path(mesh_path)
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load(mesh_file, force="mesh")
    bbox = mesh.bounds.tolist() if mesh.bounds is not None else None

    metrics = {
        "mesh_path": str(mesh_file),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "bounding_box": bbox,
        "is_watertight": bool(mesh.is_watertight),
        "file_size_mb": round(mesh_file.stat().st_size / (1024 * 1024), 6),
    }
    return metrics
