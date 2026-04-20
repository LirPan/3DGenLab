#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.eval.mesh_metrics import evaluate_mesh
from genlab.utils import ensure_parent_dir, log_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generated mesh with lightweight metrics")
    parser.add_argument("--mesh", required=True, help="Path to mesh file")
    parser.add_argument("--output-report", required=True, help="Path to benchmark JSON report")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        out_path = ensure_parent_dir(args.output_report)
        metrics = evaluate_mesh(args.mesh)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log_step(f"[Benchmark] Report path: {out_path}")
        return 0
    except Exception as exc:
        log_step(f"[Benchmark] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
