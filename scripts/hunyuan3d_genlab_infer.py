#!/usr/bin/env python3
"""
Headless inference helper for Hunyuan3D-2.1 (shape stage only → OBJ).

This script is invoked from 3DGenLab's Hunyuan3DAdapter via subprocess with
cwd set to the upstream checkout (external/Hunyuan3D-2.1).

Text-to-3D uses the same HunyuanDiT text-to-image path as upstream Gradio
(Hunyuan3D-2 hy3dgen/text2image.py), since Hunyuan3D-2.1 does not ship hy3dgen.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def _prepend_repo_paths(repo: Path) -> None:
    r = str(repo.resolve())
    for sub in ("", "hy3dshape", "hy3dpaint"):
        p = repo / sub if sub else repo
        sp = str(p.resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _apply_torchvision_fix(repo: Path) -> None:
    os.chdir(repo)
    try:
        from torchvision_fix import apply_fix  # type: ignore

        apply_fix()
    except ImportError:
        print("[hunyuan3d_genlab_infer] Warning: torchvision_fix not found, continuing.")
    except Exception as e:
        print(f"[hunyuan3d_genlab_infer] Warning: torchvision fix failed: {e}")


# ---- Text2Image (from tencent/Hunyuan3D-2 hy3dgen/text2image.py, simplified) ----
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


class HunyuanDiTPipeline:
    """Minimal T2I wrapper for text-to-3D (image conditioning for shape)."""

    def __init__(
        self,
        model_path: str = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        device: str = "cuda",
    ) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            enable_pag=True,
            pag_applied_layers=["blocks.(16|17|18|19)"],
        ).to(device)
        self.pos_txt = ",白色背景,3D风格,最佳质量"
        self.neg_txt = (
            "文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态,"
            "残缺,多余的手指,变异的手,画得不好的手,画得不好的脸,变异,畸形,模糊,脱水,糟糕的解剖学,"
            "糟糕的比例,多余的肢体,克隆的脸,毁容,恶心的比例,畸形的肢体,缺失的手臂,缺失的腿,"
            "额外的手臂,额外的腿,融合的手指,手指太多,长脖子"
        )

    def __call__(self, prompt: str, seed: int = 0):
        import torch

        _seed_everything(seed)
        generator = torch.Generator(device=self.pipe.device)
        generator = generator.manual_seed(int(seed))
        out_img = self.pipe(
            prompt=prompt[:60] + self.pos_txt,
            negative_prompt=self.neg_txt,
            num_inference_steps=25,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False,
        )[0][0]
        return out_img


def _load_prompt_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 GenLab shape inference")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Absolute path to Hunyuan3D-2.1 repository root",
    )
    parser.add_argument(
        "--mode",
        choices=("image", "text"),
        default="image",
        help="image: use --image; text: run T2I then shape",
    )
    parser.add_argument("--image", type=str, default=None, help="Input image path (image mode)")
    parser.add_argument("--prompt", type=str, default=None, help="Raw text prompt (text mode)")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="UTF-8 file containing prompt (text mode; preferred for long / special chars)",
    )
    parser.add_argument(
        "--output-mesh",
        type=str,
        required=True,
        help="Output OBJ path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="tencent/Hunyuan3D-2.1",
        help="HF repo id or local path for shape model",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="hunyuan3d-dit-v2-1",
        help="Shape weights subfolder on the Hub",
    )
    parser.add_argument(
        "--t2i-model-path",
        type=str,
        default="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        help="HF model id for text-to-image (text mode only)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument(
        "--no-rembg",
        action="store_true",
        help="Skip background removal for RGB inputs (use if image is already RGBA/cutout)",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"[hunyuan3d_genlab_infer] ERROR: repo not found: {repo}", file=sys.stderr)
        return 2

    _prepend_repo_paths(repo)
    _apply_torchvision_fix(repo)

    from PIL import Image

    import torch
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    out_mesh = Path(args.output_mesh).resolve()
    out_mesh.parent.mkdir(parents=True, exist_ok=True)

    image_pil: Image.Image | None = None
    if args.mode == "image":
        if not args.image:
            print("[hunyuan3d_genlab_infer] ERROR: --image is required for mode=image", file=sys.stderr)
            return 2
        img_path = Path(args.image).resolve()
        if not img_path.is_file():
            print(f"[hunyuan3d_genlab_infer] ERROR: image not found: {img_path}", file=sys.stderr)
            return 2
        image_pil = Image.open(img_path)
    else:
        prompt_text = args.prompt
        if args.prompt_file:
            prompt_text = _load_prompt_from_file(Path(args.prompt_file).resolve())
        if not prompt_text:
            print(
                "[hunyuan3d_genlab_infer] ERROR: text mode needs --prompt or --prompt-file",
                file=sys.stderr,
            )
            return 2
        print("[hunyuan3d_genlab_infer] Loading text-to-image pipeline ...")
        t2i = HunyuanDiTPipeline(args.t2i_model_path, device=args.device)
        image_pil = t2i(prompt_text, seed=args.seed)
        del t2i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert image_pil is not None
    if not args.no_rembg:
        rembg = BackgroundRemover()
        image_pil = rembg(image_pil.convert("RGB"))

    print("[hunyuan3d_genlab_infer] Loading shape pipeline ...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed))

    print("[hunyuan3d_genlab_infer] Running shape inference ...")
    mesh = pipeline(
        image=image_pil,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        output_type="trimesh",
    )[0]

    mesh.export(str(out_mesh))
    print(f"[hunyuan3d_genlab_infer] Wrote mesh: {out_mesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
