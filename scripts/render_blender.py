#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.render.blender_render import render_mesh_with_blender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mesh using Blender")
    parser.add_argument("--mesh", required=True, help="Input mesh path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument(
        "--mode",
        default="system",
        choices=("system", "python"),
        help="Render backend mode (python auto-falls back to system)",
    )
    parser.add_argument("--blender-bin", default="blender", help="Blender executable path for system mode")
    parser.add_argument("--width", type=int, default=768, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=768, help="Render height in pixels")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=("CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"),
        help="Blender render engine",
    )
    parser.add_argument(
        "--opaque-bg",
        action="store_true",
        help="Disable transparent background (default is transparent)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_mesh_with_blender(
        args.mesh,
        args.output,
        mode=args.mode,
        blender_bin=args.blender_bin,
        width=args.width,
        height=args.height,
        samples=args.samples,
        engine=args.engine,
        transparent_background=not args.opaque_bg,
    )
    print(f"[render_blender] wrote {output}")


if __name__ == "__main__":
    main()
