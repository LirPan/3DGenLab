#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.eval.mesh_metrics import evaluate_mesh
from genlab.models.registry import MODEL_REGISTRY, get_model
from genlab.render.blender_render import render_mesh_with_blender
from genlab.utils import ensure_dir, load_yaml_config, log_step

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
    "render_path",
    "render_success",
    "render_error",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-driven multi-model 3D evaluation runner")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--case-id", help="Run one case by case_id from dataset metadata")
    scope.add_argument("--group", help="Run all cases in one group")
    scope.add_argument("--all", action="store_true", help="Run all cases in dataset metadata")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        default="dataset/metadata/cases.json",
        help="Path to canonical dataset metadata JSON",
    )
    parser.add_argument("--models", help="Comma-separated model names to include (optional)")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_const",
        const=True,
        default=None,
        help="Force dry-run mode on",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_const",
        const=False,
        help="Force dry-run mode off",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_const",
        const=True,
        default=None,
        help="Force benchmark on",
    )
    parser.add_argument(
        "--no-benchmark",
        dest="benchmark",
        action="store_const",
        const=False,
        help="Force benchmark off",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_const",
        const=True,
        default=None,
        help="Enable Blender render output for successful meshes",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_const",
        const=False,
        help="Disable Blender render output",
    )
    return parser.parse_args()


def _load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset metadata file not found: {dataset_path}")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset metadata must contain a non-empty 'cases' list")
    return cases


def _resolve_selected_cases(args: argparse.Namespace, all_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.case_id:
        selected = [c for c in all_cases if c.get("case_id") == args.case_id]
        if not selected:
            raise ValueError(f"case_id not found in dataset: {args.case_id}")
        return selected
    if args.group:
        selected = [c for c in all_cases if c.get("group") == args.group]
        if not selected:
            raise ValueError(f"group not found or empty in dataset: {args.group}")
        return selected
    return all_cases


def _parse_model_filter(models_arg: str | None) -> set[str] | None:
    if not models_arg:
        return None
    names = {name.strip().lower() for name in models_arg.split(",") if name.strip()}
    if not names:
        return None
    unsupported = sorted(name for name in names if name not in MODEL_REGISTRY)
    if unsupported:
        raise ValueError(
            f"Unsupported model(s) in --models: {', '.join(unsupported)}. "
            f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )
    return names


def _normalize_output_path(
    config: dict[str, Any],
    model_name: str,
    case_id: str,
    generated_path: Path,
) -> Path:
    model_output_dir = config["models"][model_name]["output_dir"]
    out_dir = ensure_dir(ROOT / model_output_dir)
    ext = generated_path.suffix if generated_path.suffix else ".obj"
    canonical = out_dir / f"{case_id}_{model_name}{ext}"
    if generated_path.resolve() != canonical.resolve():
        canonical.write_bytes(generated_path.read_bytes())
    return canonical


def _blank_summary_row(case: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "group": case.get("group", ""),
        "model_name": model_name,
        "input_type": case.get("type", ""),
        "input_path": case.get("input_path", ""),
        "output_path": "",
        "output_format": "",
        "success": False,
        "runtime_seconds": None,
        "num_vertices": None,
        "num_faces": None,
        "is_watertight": None,
        "file_size_mb": None,
        "render_path": "",
        "render_success": None,
        "render_error": "",
        "error_message": "",
    }


def _write_per_run_report(reports_dir: Path, row: dict[str, Any]) -> Path:
    report_path = reports_dir / f"{row['case_id']}_{row['model_name']}.json"
    report_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _write_aggregate_reports(
    reports_dir: Path,
    rows: list[dict[str, Any]],
    *,
    config_path: str,
    dataset_path: str,
    dry_run: bool,
    benchmark: bool,
) -> tuple[Path, Path]:
    success_count = sum(1 for row in rows if row["success"])
    summary_json_path = reports_dir / "comparison_summary.json"
    summary_csv_path = reports_dir / "comparison_summary.csv"

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config_path,
        "dataset": dataset_path,
        "dry_run": dry_run,
        "benchmark": benchmark,
        "total_runs": len(rows),
        "success_count": success_count,
        "fail_count": len(rows) - success_count,
        "rows": rows,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return summary_json_path, summary_csv_path


def _print_compact_summary(rows: list[dict[str, Any]]) -> None:
    print("")
    print("CASE_ID | MODEL | STATUS | RUNTIME_SEC | OUTPUT_OR_ERROR")
    print("-|-|-|-|-")
    for row in rows:
        status = "OK" if row["success"] else "FAILED"
        runtime = (
            f"{row['runtime_seconds']:.3f}"
            if isinstance(row.get("runtime_seconds"), (float, int))
            else ""
        )
        output_or_error = row["output_path"] if row["success"] else row.get("error_message", "")
        print(f"{row['case_id']} | {row['model_name']} | {status} | {runtime} | {output_or_error}")


def _resolve_render_settings(config: dict[str, Any], force_render: bool | None) -> dict[str, Any]:
    top_render = config.get("render", False)
    render_settings = config.get("render_settings", {})
    if not isinstance(render_settings, dict):
        render_settings = {}

    enabled_default = False
    if isinstance(top_render, bool):
        enabled_default = top_render
    elif isinstance(top_render, dict):
        enabled_default = bool(top_render.get("enabled", False))
        merged = dict(top_render)
        merged.update(render_settings)
        render_settings = merged

    enabled = enabled_default if force_render is None else bool(force_render)
    configured_models = render_settings.get("models", ["trellis"])
    if not isinstance(configured_models, list):
        configured_models = ["trellis"]
    models = [str(name).strip().lower() for name in configured_models if str(name).strip()]
    if not models:
        models = ["trellis"]
    return {
        "enabled": enabled,
        "mode": str(render_settings.get("mode", "system")),
        "blender_bin": str(render_settings.get("blender_bin", "blender")),
        "width": int(render_settings.get("width", 768)),
        "height": int(render_settings.get("height", 768)),
        "samples": int(render_settings.get("samples", 32)),
        "engine": str(render_settings.get("engine", "CYCLES")),
        "transparent_background": bool(render_settings.get("transparent_background", True)),
        "image_format": str(render_settings.get("image_format", "png")).lower().lstrip("."),
        "models": models,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml_config(ROOT / args.config)
    dry_run = bool(config.get("dry_run", True)) if args.dry_run is None else bool(args.dry_run)
    benchmark = bool(config.get("benchmark", True)) if args.benchmark is None else bool(args.benchmark)
    render_cfg = _resolve_render_settings(config, args.render)

    all_cases = _load_cases(ROOT / args.dataset)
    selected_cases = _resolve_selected_cases(args, all_cases)
    selected_model_filter = _parse_model_filter(args.models)

    output_root = ensure_dir(ROOT / config.get("output_root", "outputs"))
    reports_dir = ensure_dir(output_root / "reports")
    renders_root = ensure_dir(output_root / "renders") if render_cfg["enabled"] else None

    log_step(f"[DatasetEval] Config: {args.config}")
    log_step(f"[DatasetEval] Dataset: {args.dataset}")
    log_step(f"[DatasetEval] Dry run: {dry_run}")
    log_step(f"[DatasetEval] Benchmark: {benchmark}")
    log_step(f"[DatasetEval] Render: {render_cfg['enabled']}")
    log_step(f"[DatasetEval] Cases selected: {len(selected_cases)}")

    rows: list[dict[str, Any]] = []
    failed = False

    for case in selected_cases:
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError(f"Found case without case_id: {case}")
        input_type = str(case.get("type", "")).strip().lower()
        input_rel_path = case.get("input_path", "")
        input_path = ROOT / input_rel_path
        if not input_path.exists():
            raise FileNotFoundError(f"Case input_path does not exist: {input_rel_path} (case_id={case_id})")

        intended_models = [str(m).strip().lower() for m in case.get("intended_models", [])]
        if selected_model_filter:
            intended_models = [m for m in intended_models if m in selected_model_filter]
        for model_name in intended_models:
            if model_name not in config.get("models", {}):
                log_step(f"[DatasetEval] Skip model {model_name} for {case_id}: missing in config.models")
                continue
            if model_name not in MODEL_REGISTRY:
                log_step(f"[DatasetEval] Skip unsupported model {model_name} in metadata for {case_id}")
                continue

            row = _blank_summary_row(case, model_name)
            rows.append(row)

            log_step(f"[DatasetEval] Running {case_id} -> {model_name}")
            try:
                model = get_model(model_name=model_name, config=config, dry_run=dry_run)
                model.setup()

                prompt_text = None
                input_image = None
                if input_type == "image":
                    input_image = str(input_path)
                elif input_type == "text":
                    prompt_text = input_path.read_text(encoding="utf-8").strip()
                else:
                    raise ValueError(f"Unsupported case type '{input_type}' for case_id={case_id}")

                model_output_dir = str(ROOT / config["models"][model_name]["output_dir"])
                start = time.perf_counter()
                generated_mesh_path = Path(
                    model.generate(
                        input_image=input_image,
                        input_prompt=prompt_text,
                        output_dir=model_output_dir,
                    )
                )
                runtime = time.perf_counter() - start

                canonical_output = _normalize_output_path(
                    config=config,
                    model_name=model_name,
                    case_id=case_id,
                    generated_path=generated_mesh_path,
                )

                row["success"] = True
                row["runtime_seconds"] = round(runtime, 6)
                row["output_path"] = str(canonical_output)
                row["output_format"] = canonical_output.suffix.lstrip(".")

                if benchmark:
                    metrics = evaluate_mesh(canonical_output)
                    row["num_vertices"] = metrics.get("num_vertices")
                    row["num_faces"] = metrics.get("num_faces")
                    row["is_watertight"] = metrics.get("is_watertight")
                    row["file_size_mb"] = metrics.get("file_size_mb")

                if (
                    render_cfg["enabled"]
                    and renders_root is not None
                    and model_name in render_cfg["models"]
                ):
                    render_dir = ensure_dir(renders_root / model_name)
                    render_path = render_dir / f"{case_id}_{model_name}.{render_cfg['image_format']}"
                    try:
                        render_mesh_with_blender(
                            canonical_output,
                            render_path,
                            mode=render_cfg["mode"],
                            blender_bin=render_cfg["blender_bin"],
                            width=render_cfg["width"],
                            height=render_cfg["height"],
                            samples=render_cfg["samples"],
                            engine=render_cfg["engine"],
                            transparent_background=render_cfg["transparent_background"],
                        )
                        row["render_path"] = str(render_path)
                        row["render_success"] = True
                        row["render_error"] = ""
                    except Exception as render_exc:
                        row["render_path"] = str(render_path)
                        row["render_success"] = False
                        row["render_error"] = str(render_exc)
                        log_step(f"[DatasetEval] Render failed {case_id} -> {model_name}: {render_exc}")

                report_path = _write_per_run_report(reports_dir, row)
                log_step(f"[DatasetEval] Report written: {report_path}")
            except Exception as exc:
                failed = True
                row["success"] = False
                row["error_message"] = str(exc)
                _write_per_run_report(reports_dir, row)
                log_step(f"[DatasetEval] FAILED {case_id} -> {model_name}: {exc}")
                continue

    summary_json, summary_csv = _write_aggregate_reports(
        reports_dir,
        rows,
        config_path=args.config,
        dataset_path=args.dataset,
        dry_run=dry_run,
        benchmark=benchmark,
    )

    _print_compact_summary(rows)
    log_step(f"[DatasetEval] Summary JSON: {summary_json}")
    log_step(f"[DatasetEval] Summary CSV: {summary_csv}")
    log_step(f"[DatasetEval] Total runs: {len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
