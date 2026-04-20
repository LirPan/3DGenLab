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


def main() -> int:
    args = parse_args()
    try:
        config = load_yaml_config(args.config)
        model_name = (args.model or config.get("model_name", "triposr")).strip().lower()
        input_image = args.input or config.get("input_image")
        input_prompt = _resolve_prompt(args.prompt) or _resolve_prompt(config.get("input_prompt"))
        dry_run = args.dry_run or bool(config.get("dry_run", True))
        do_benchmark = args.benchmark or bool(config.get("benchmark", False))

        output_root = ensure_dir(config.get("output_root", "outputs"))
        reports_dir = ensure_dir(output_root / "reports")

        if model_name not in config.get("models", {}):
            raise ValueError(
                f"Model '{model_name}' is missing from config.models. "
                f"Defined: {', '.join(config.get('models', {}).keys())}"
            )

        model_output_dir = ensure_dir(config["models"][model_name]["output_dir"])

        log_step(f"[Pipeline] Config: {args.config}")
        log_step(f"[Pipeline] Model: {model_name}")
        log_step(f"[Pipeline] Dry run: {dry_run}")
        log_step(f"[Pipeline] Input image: {input_image}")
        log_step(f"[Pipeline] Input prompt: {'<provided>' if input_prompt else '<none>'}")
        log_step(f"[Pipeline] Output directory: {model_output_dir}")

        model = get_model(model_name=model_name, config=config, dry_run=dry_run)
        log_step("[Pipeline] Running model setup")
        model.setup()
        log_step("[Pipeline] Generating mesh")
        mesh_path = model.generate(
            input_image=input_image,
            input_prompt=input_prompt,
            output_dir=str(model_output_dir),
        )
        log_step(f"[Pipeline] Mesh generated at: {mesh_path}")

        if do_benchmark:
            log_step("[Pipeline] Running benchmark")
            metrics = evaluate_mesh(mesh_path)
            report_path = reports_dir / f"{model_name}_{get_stem(mesh_path)}_metrics.json"
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log_step(f"[Pipeline] Benchmark report: {report_path}")

        return 0
    except Exception as exc:
        log_step(f"[Pipeline] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
