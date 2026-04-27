#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

SUMMARY_FIELDS = [
    "case_id",
    "group",
    "model_name",
    "input_type",
    "input_path",
    "output_path",
    "output_format",
    "success",
    "runtime_seconds",
    "num_vertices",
    "num_faces",
    "is_watertight",
    "file_size_mb",
    "error_message",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for p in sorted(reports_dir.glob("*.json")):
        if p.name in {"comparison_summary.json"}:
            continue
        if p.name.endswith("_metrics.json"):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if "case_id" not in data or "model_name" not in data:
            continue
        rows.append({k: data.get(k) for k in SUMMARY_FIELDS})

    rows.sort(key=lambda r: (str(r.get("group", "")), str(r.get("case_id", "")), str(r.get("model_name", ""))))

    success_count = sum(1 for r in rows if bool(r.get("success")))
    summary_json = reports_dir / "comparison_summary.json"
    summary_csv = reports_dir / "comparison_summary.csv"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "rebuild_reports_summary.py",
        "total_runs": len(rows),
        "success_count": success_count,
        "fail_count": len(rows) - success_count,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    print(f"[rebuild] total={len(rows)} success={success_count} fail={len(rows)-success_count}")
    print(f"[rebuild] json={summary_json}")
    print(f"[rebuild] csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
