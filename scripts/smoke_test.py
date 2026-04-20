#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genlab.eval.mesh_metrics import evaluate_mesh
from genlab.models.registry import get_model
from genlab.utils import ensure_dir, ensure_parent_dir, load_yaml_config, log_step

ALL_MODELS = ["triposr", "instantmesh", "hunyuan3d", "trellis"]


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = len(data).to_bytes(4, "big")
    crc = zlib.crc32(chunk_type + data).to_bytes(4, "big")
    return length + chunk_type + data + crc


def _write_minimal_png(path: Path) -> None:
    width = 1
    height = 1
    bit_depth = 8
    color_type = 2  # RGB
    compression = 0
    filter_method = 0
    interlace = 0

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([bit_depth, color_type, compression, filter_method, interlace])
    )
    raw_scanline = b"\x00" + b"\xB4\xB4\xDC"  # no-filter + RGB pixel
    idat = zlib.compress(raw_scanline)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", idat)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _ensure_example_inputs() -> tuple[Path, Path]:
    image_path = ensure_parent_dir(ROOT / "inputs" / "images" / "example.png")
    prompt_path = ensure_parent_dir(ROOT / "inputs" / "prompts" / "example.txt")

    if not image_path.exists():
        _write_minimal_png(image_path)
        log_step(f"[Smoke] Created example image: {image_path}")

    if not prompt_path.exists():
        prompt_path.write_text("A simple chair", encoding="utf-8")
        log_step(f"[Smoke] Created example prompt: {prompt_path}")

    return image_path, prompt_path


def main() -> int:
    try:
        config = load_yaml_config(ROOT / "configs" / "default.yaml")
        image_path, prompt_path = _ensure_example_inputs()
        reports_dir = ensure_dir(ROOT / "outputs" / "reports")

        created_meshes: list[Path] = []
        created_reports: list[Path] = []

        for model_name in ALL_MODELS:
            log_step(f"[Smoke] Running dry-run for {model_name}")
            output_dir = ensure_dir(ROOT / config["models"][model_name]["output_dir"])
            model = get_model(model_name=model_name, config=config, dry_run=True)
            model.setup()
            mesh_path = Path(
                model.generate(
                    input_image=str(image_path),
                    input_prompt=prompt_path.read_text(encoding="utf-8").strip(),
                    output_dir=str(output_dir),
                )
            )
            metrics = evaluate_mesh(mesh_path)
            report_path = reports_dir / f"smoke_{model_name}_metrics.json"
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            created_meshes.append(mesh_path)
            created_reports.append(report_path)
            log_step(f"[Smoke] Created mesh/report for {model_name}")

        missing_meshes = [str(path) for path in created_meshes if not path.exists()]
        missing_reports = [str(path) for path in created_reports if not path.exists()]
        if len(created_meshes) != 4 or len(created_reports) != 4:
            raise RuntimeError("Expected 4 meshes and 4 reports from smoke test.")
        if missing_meshes or missing_reports:
            raise RuntimeError(
                f"Missing outputs. Meshes: {missing_meshes}, Reports: {missing_reports}"
            )

        print("SMOKE TEST PASSED")
        return 0
    except Exception as exc:
        log_step(f"[Smoke] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
