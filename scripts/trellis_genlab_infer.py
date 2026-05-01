#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str | None:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    if prompt:
        return prompt.strip()
    return None


def _write_obj(mesh, output_mesh: Path) -> None:
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    with output_mesh.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless TRELLIS inference helper for GenLab")
    parser.add_argument("--repo", required=True, help="TRELLIS repository root")
    parser.add_argument("--mode", choices=("image", "text"), default="image")
    parser.add_argument("--image", help="Input image path for image mode")
    parser.add_argument("--prompt", help="Text prompt for text mode")
    parser.add_argument("--prompt-file", help="Prompt file for text mode")
    parser.add_argument("--output-mesh", required=True, help="Output OBJ mesh path")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path or HF model id (optional). If omitted, TRELLIS default is used.",
    )
    parser.add_argument("--device", default="cuda", help="Device for TRELLIS pipeline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--sparse-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"[trellis_genlab_infer] ERROR: repo not found: {repo}", file=sys.stderr)
        return 2
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    output_mesh = Path(args.output_mesh).resolve()

    try:
        from PIL import Image
        from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
    except Exception as exc:
        print(f"[trellis_genlab_infer] ERROR: failed to import TRELLIS modules: {exc}", file=sys.stderr)
        return 3

    if args.mode == "image":
        if not args.image:
            print("[trellis_genlab_infer] ERROR: --image required for image mode", file=sys.stderr)
            return 2
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            print(f"[trellis_genlab_infer] ERROR: image not found: {image_path}", file=sys.stderr)
            return 2
        image = Image.open(image_path).convert("RGB")
        pipeline_cls = TrellisImageTo3DPipeline
        pipeline_input = image
    else:
        prompt = _load_prompt(args.prompt, args.prompt_file)
        if not prompt:
            print("[trellis_genlab_infer] ERROR: prompt is required for text mode", file=sys.stderr)
            return 2
        pipeline_cls = TrellisTextTo3DPipeline
        pipeline_input = prompt

    load_exc = None
    switched_hf_endpoint = False
    for attempt in range(1, 4):
        try:
            if args.model_path:
                pipeline = pipeline_cls.from_pretrained(args.model_path)
            else:
                pipeline = pipeline_cls.from_pretrained()
            break
        except TypeError:
            if args.model_path:
                pipeline = pipeline_cls.from_pretrained(args.model_path)
                break
            print(
                "[trellis_genlab_infer] ERROR: model_path is required by this TRELLIS build. "
                "Set models.trellis.inference.model_path in config.",
                file=sys.stderr,
            )
            return 2
        except Exception as exc:
            load_exc = exc
            if (
                not switched_hf_endpoint
                and os.environ.get("HF_ENDPOINT")
                and "hf-mirror.com" in str(exc).lower()
            ):
                switched_hf_endpoint = True
                os.environ.pop("HF_ENDPOINT", None)
                print(
                    "[trellis_genlab_infer] WARN: HF_ENDPOINT mirror failed, retrying with default Hugging Face endpoint",
                    file=sys.stderr,
                )
            if attempt >= 3:
                print(f"[trellis_genlab_infer] ERROR: failed to load TRELLIS pipeline: {exc}", file=sys.stderr)
                return 4
            wait_s = attempt * 2
            print(
                f"[trellis_genlab_infer] WARN: load attempt {attempt}/3 failed: {exc}; retrying in {wait_s}s",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    else:
        print(f"[trellis_genlab_infer] ERROR: failed to load TRELLIS pipeline: {load_exc}", file=sys.stderr)
        return 4

    if args.device == "cuda":
        try:
            pipeline.cuda()
        except Exception as exc:
            print(f"[trellis_genlab_infer] ERROR: failed to move pipeline to CUDA: {exc}", file=sys.stderr)
            return 5

    outputs = pipeline.run(
        pipeline_input,
        seed=args.seed,
        sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": args.sparse_cfg},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
    )
    if "mesh" not in outputs or not outputs["mesh"]:
        print("[trellis_genlab_infer] ERROR: TRELLIS output did not contain mesh", file=sys.stderr)
        return 6

    _write_obj(outputs["mesh"][0], output_mesh)
    print(f"[trellis_genlab_infer] Wrote mesh: {output_mesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
