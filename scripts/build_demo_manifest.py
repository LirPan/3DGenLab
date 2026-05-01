#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "outputs" / "reports"
    outputs_dir = root / "outputs"
    demo_ready_dir = outputs_dir / "demo_ready"
    demo_ready_dir.mkdir(parents=True, exist_ok=True)

    summary_json = reports_dir / "comparison_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"summary not found: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])

    groups: dict[str, dict] = {}
    for row in rows:
        group = str(row.get("group", "ungrouped"))
        case_id = str(row.get("case_id", ""))
        model_name = str(row.get("model_name", ""))
        case_map = groups.setdefault(group, {})
        model_list = case_map.setdefault(case_id, [])
        model_list.append(
            {
                "model_name": model_name,
                "success": bool(row.get("success")),
                "output_path": row.get("output_path", ""),
                "output_format": row.get("output_format", ""),
                "runtime_seconds": row.get("runtime_seconds"),
                "num_vertices": row.get("num_vertices"),
                "num_faces": row.get("num_faces"),
                "is_watertight": row.get("is_watertight"),
                "file_size_mb": row.get("file_size_mb"),
                "render_path": row.get("render_path", ""),
                "render_success": row.get("render_success"),
                "render_error": row.get("render_error", ""),
                "error_message": row.get("error_message", ""),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": payload.get("total_runs", len(rows)),
        "success_count": payload.get("success_count", 0),
        "fail_count": payload.get("fail_count", 0),
        "groups": groups,
        "summary_paths": {
            "json": str(summary_json),
            "csv": str(reports_dir / "comparison_summary.csv"),
        },
    }

    demo_manifest_path = outputs_dir / "demo_manifest.json"
    demo_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    index_payload = {
        "demo_manifest": str(demo_manifest_path),
        "summary_json": str(summary_json),
        "summary_csv": str(reports_dir / "comparison_summary.csv"),
        "models_dir": {
            "triposr": str(outputs_dir / "triposr"),
            "instantmesh": str(outputs_dir / "instantmesh"),
            "hunyuan3d": str(outputs_dir / "hunyuan3d"),
            "trellis": str(outputs_dir / "trellis"),
        },
        "renders_dir": str(outputs_dir / "renders"),
    }
    index_path = demo_ready_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[manifest] {demo_manifest_path}")
    print(f"[manifest] {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
