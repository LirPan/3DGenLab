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
from genlab.models.registry import get_model
from genlab.utils import ensure_dir, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3DGenLab pipeline for a single model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", help="Override input image path")
    parser.add_argument("--prompt", help="Override input prompt path or raw prompt text")
    parser.add_argument("--model", help="Model name override")
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


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_name = args.model or config.get("model_name", "triposr")
    input_image = args.input or config.get("input_image")
    input_prompt = _resolve_prompt(args.prompt) or _resolve_prompt(config.get("input_prompt"))
    dry_run = args.dry_run or bool(config.get("dry_run", True))
    do_benchmark = args.benchmark or bool(config.get("benchmark", False))

    ensure_dir(config.get("output_root", "outputs"))
    ensure_dir("outputs/reports")

    print(f"[Pipeline] Config: {args.config}")
    print(f"[Pipeline] Model: {model_name}")
    print(f"[Pipeline] Dry run: {dry_run}")
    print(f"[Pipeline] Input image: {input_image}")
    print(f"[Pipeline] Input prompt: {'<provided>' if input_prompt else '<none>'}")

    model = get_model(model_name=model_name, config=config, dry_run=dry_run)
    model.setup()
    mesh_path = model.generate(
        input_image=input_image,
        input_prompt=input_prompt,
        output_dir=config["models"][model_name]["output_dir"],
    )

    print(f"[Pipeline] Mesh generated at: {mesh_path}")

    if do_benchmark:
        metrics = calculate_mesh_metrics(mesh_path)
        report_path = Path("outputs/reports") / f"{Path(mesh_path).stem}_metrics.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[Pipeline] Benchmark report: {report_path}")


if __name__ == "__main__":
    main()
