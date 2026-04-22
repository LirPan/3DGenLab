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
    parser = argparse.ArgumentParser(description="Render mesh using Blender (placeholder)")
    parser.add_argument("--mesh", required=True, help="Input mesh path")
    parser.add_argument("--output", required=True, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mesh_with_blender(args.mesh, args.output)


if __name__ == "__main__":
    main()
