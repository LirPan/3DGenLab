#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.render.blender_render import _decode_png_rgb_stats, render_mesh_with_blender


def _collect_trellis_meshes() -> list[Path]:
    mesh_dir = ROOT / "outputs" / "trellis"
    if not mesh_dir.exists():
        return []
    meshes = []
    for obj in sorted(mesh_dir.glob("*_trellis.obj")):
        # Dummy mesh is around 238 bytes; keep only likely real outputs.
        if obj.stat().st_size > 1_000_000:
            meshes.append(obj)
    return meshes[:2]


def _validate_render(path: Path) -> None:
    stats = _decode_png_rgb_stats(path)
    width = int(stats["width"])
    height = int(stats["height"])
    if width != 768 or height != 768:
        raise RuntimeError(f"unexpected render size for {path}: {(height, width)}")

    max_rgb = int(stats["max_rgb"])
    unique_rgb = int(stats["unique_rgb"])
    if max_rgb <= 4:
        raise RuntimeError(f"render is near-black: {path} max_rgb={max_rgb}")
    if unique_rgb < 2:
        raise RuntimeError(f"render has too little color variation: {path} unique_rgb={unique_rgb}")

    print(f"[render_smoke] ok: {path} max_rgb={max_rgb} unique_rgb={unique_rgb}")


def main() -> int:
    meshes = _collect_trellis_meshes()
    if not meshes:
        print("[render_smoke] no real TRELLIS meshes found (>1MB) under outputs/trellis")
        return 2

    smoke_dir = ROOT / "outputs" / "renders" / "_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    for mesh in meshes:
        output = smoke_dir / f"{mesh.stem}.png"
        render_mesh_with_blender(
            mesh,
            output,
            mode="system",
            blender_bin="blender",
            width=768,
            height=768,
            samples=32,
            engine="CYCLES",
            transparent_background=True,
        )
        _validate_render(output)

    print("")
    print("[render_smoke] hint: old noisy png files may still exist in:")
    print("  outputs/renders/triposr")
    print("  outputs/renders/instantmesh")
    print("  outputs/renders/hunyuan3d")
    print("[render_smoke] clean manually if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
