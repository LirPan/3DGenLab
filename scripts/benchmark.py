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

from genlab.eval.mesh_metrics import calculate_mesh_metrics
from genlab.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generated mesh with lightweight metrics")
    parser.add_argument("--mesh", required=True, help="Path to mesh file")
    parser.add_argument(
        "--report-dir",
        default="outputs/reports",
        help="Directory to store benchmark JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ensure_dir(args.report_dir)

    metrics = calculate_mesh_metrics(args.mesh)
    out_path = Path(report_dir) / f"{Path(args.mesh).stem}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[Benchmark] Metrics saved to: {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
