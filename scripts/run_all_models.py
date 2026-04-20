#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.models.registry import get_model
from genlab.utils import ensure_dir, load_yaml_config

ALL_MODELS = ["triposr", "instantmesh", "hunyuan3d", "trellis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all supported 3D generation model adapters")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", help="Override input image path")
    parser.add_argument("--prompt", help="Override input prompt path or raw prompt text")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run placeholder generation")
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

    input_image = args.input or config.get("input_image")
    input_prompt = _resolve_prompt(args.prompt) or _resolve_prompt(config.get("input_prompt"))
    dry_run = args.dry_run or bool(config.get("dry_run", True))

    print(f"[RunAll] Config: {args.config}")
    print(f"[RunAll] Dry run: {dry_run}")
    print(f"[RunAll] Input image: {input_image}")
    print(f"[RunAll] Input prompt: {'<provided>' if input_prompt else '<none>'}")

    for model_name in ALL_MODELS:
        print(f"\n[RunAll] === Running model: {model_name} ===")
        model_output = config["models"][model_name]["output_dir"]
        ensure_dir(model_output)

        model = get_model(model_name=model_name, config=config, dry_run=dry_run)
        model.setup()
        mesh_path = model.generate(
            input_image=input_image,
            input_prompt=input_prompt,
            output_dir=model_output,
        )
        print(f"[RunAll] {model_name} output mesh: {mesh_path}")


if __name__ == "__main__":
    main()
