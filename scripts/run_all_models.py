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
from genlab.models.registry import get_model
from genlab.utils import ensure_dir, get_stem, load_yaml_config, log_step

ALL_MODELS = ["triposr", "instantmesh", "hunyuan3d", "trellis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all supported 3D generation model adapters")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", help="Override input image path")
    parser.add_argument("--prompt", help="Override input prompt path or raw prompt text")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run placeholder generation")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after generation")
    return parser.parse_args()


def _resolve_prompt(prompt_arg: str | None) -> str | None:
    if not prompt_arg:
        return None
    prompt_path = Path(prompt_arg)
    if prompt_path.exists() and prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip()
    return prompt_arg


def _print_summary(rows: list[dict]) -> None:
    headers = ["MODEL", "STATUS", "MESH", "REPORT", "ERROR"]
    lines = [" | ".join(headers), "-|-|-|-|-"]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    row["model"],
                    row["status"],
                    row.get("mesh", ""),
                    row.get("report", ""),
                    row.get("error", ""),
                ]
            )
        )
    print("\n".join(lines))


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    configured_models = config.get("models", {})

    input_image = args.input or config.get("input_image")
    input_prompt = _resolve_prompt(args.prompt) or _resolve_prompt(config.get("input_prompt"))
    dry_run = args.dry_run or bool(config.get("dry_run", True))
    do_benchmark = args.benchmark or bool(config.get("benchmark", False))

    output_root = ensure_dir(config.get("output_root", "outputs"))
    reports_dir = ensure_dir(output_root / "reports")

    log_step(f"[RunAll] Config: {args.config}")
    log_step(f"[RunAll] Dry run: {dry_run}")
    log_step(f"[RunAll] Benchmark: {do_benchmark}")
    log_step(f"[RunAll] Input image: {input_image}")
    log_step(f"[RunAll] Input prompt: {'<provided>' if input_prompt else '<none>'}")

    summary_rows: list[dict] = []
    failed = False

    models_to_run = [m for m in ALL_MODELS if m in configured_models]
    if not models_to_run:
        raise ValueError(
            f"No supported models found in config.models. Supported: {', '.join(ALL_MODELS)}"
        )
    skipped_models = [m for m in ALL_MODELS if m not in configured_models]
    if skipped_models:
        log_step(f"[RunAll] Skipping undefined models in config: {', '.join(skipped_models)}")

    for model_name in models_to_run:
        log_step(f"[RunAll] Running model: {model_name}")
        try:
            model_output = ensure_dir(configured_models[model_name]["output_dir"])
            model = get_model(model_name=model_name, config=config, dry_run=dry_run)
            model.setup()
            mesh_path = model.generate(
                input_image=input_image,
                input_prompt=input_prompt,
                output_dir=str(model_output),
            )

            report_path = ""
            if do_benchmark:
                metrics = evaluate_mesh(mesh_path)
                report_file = reports_dir / f"{model_name}_{get_stem(mesh_path)}_metrics.json"
                report_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                report_path = str(report_file)
                log_step(f"[RunAll] Benchmark report: {report_path}")

            summary_rows.append(
                {"model": model_name, "status": "OK", "mesh": str(mesh_path), "report": report_path, "error": ""}
            )
        except Exception as exc:
            failed = True
            log_step(f"[RunAll] ERROR for model {model_name}: {exc}")
            summary_rows.append(
                {"model": model_name, "status": "FAILED", "mesh": "", "report": "", "error": str(exc)}
            )
            continue

    print("")
    _print_summary(summary_rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
