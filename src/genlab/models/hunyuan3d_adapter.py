from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.models.dummy_mesh import write_dummy_cube_obj
from genlab.utils import ensure_dir, log_step


def _prompt_stem(prompt: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", prompt.strip())
    s = s.strip("_")[:max_len].strip("_")
    return s or "prompt"


class Hunyuan3DAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="hunyuan3d")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        log_step("[Hunyuan3D-2.1] setup complete")

    def _build_output_mesh_path(
        self,
        out_dir: Path,
        *,
        input_image: str | None,
        input_prompt: str | None,
        use_image: bool,
    ) -> Path:
        if use_image and input_image:
            stem = Path(input_image).stem
        elif input_prompt:
            stem = _prompt_stem(input_prompt)
        else:
            stem = "example"
        return out_dir / f"{stem}_hunyuan3d.obj"

    def _resolve_mode(
        self,
        inference_cfg: dict,
        input_image: str | None,
        input_prompt: str | None,
        *,
        allow_missing: bool = False,
    ) -> bool:
        """
        Return True if using image conditioning, False for text-to-3D (T2I then shape).
        """
        has_image = bool(input_image) and Path(input_image).is_file()
        has_prompt = bool(input_prompt and str(input_prompt).strip())

        if not has_image and not has_prompt:
            if allow_missing:
                return True
            raise ValueError(
                "[Hunyuan3D-2.1][real-mode] Need an input image and/or a text prompt. "
                "Provide --input <image_path> and/or --prompt / input_prompt in config."
            )

        prefer = str(inference_cfg.get("prefer", "image")).strip().lower()
        if prefer not in ("image", "text"):
            raise ValueError(
                "[Hunyuan3D-2.1][real-mode] models.hunyuan3d.inference.prefer must be 'image' or 'text'."
            )

        if has_image and has_prompt:
            return prefer != "text"
        if has_image:
            return True
        return False

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["hunyuan3d"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        ensure_dir(out_dir)

        inference_cfg = model_cfg.get("inference", {})
        use_image = self._resolve_mode(
            inference_cfg,
            input_image,
            input_prompt,
            allow_missing=self.dry_run,
        )

        out_mesh = self._build_output_mesh_path(
            out_dir,
            input_image=input_image,
            input_prompt=input_prompt,
            use_image=use_image,
        )

        if self.dry_run:
            write_dummy_cube_obj(out_mesh)
            log_step(f"[Hunyuan3D-2.1][dry-run] Wrote dummy cube mesh: {out_mesh}")
            return str(out_mesh.resolve())

        repo_path = Path(model_cfg.get("repo_path", "external/Hunyuan3D-2.1")).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                "[Hunyuan3D-2.1][real-mode] External repository missing: "
                f"{repo_path}. Run: bash scripts/setup_external_repos.sh"
            )

        command_image = inference_cfg.get("command_image") or inference_cfg.get("command")
        command_text = inference_cfg.get("command_text")
        if use_image:
            command_template = command_image
        else:
            command_template = command_text
        if not command_template:
            raise ValueError(
                "[Hunyuan3D-2.1][real-mode] Missing inference command template: "
                "set models.hunyuan3d.inference.command_image (or command) "
                "and command_text."
            )

        expected_mesh_cfg = inference_cfg.get("expected_mesh")
        if expected_mesh_cfg:
            stem = Path(input_image).stem if (use_image and input_image) else _prompt_stem(
                input_prompt or ""
            )
            expected_mesh_formatted = expected_mesh_cfg.format(
                input_stem=stem,
                output_dir=str(out_dir.resolve()),
            )
            out_mesh = Path(expected_mesh_formatted)
            if not out_mesh.is_absolute():
                out_mesh = Path.cwd() / out_mesh
        ensure_dir(out_mesh.parent)

        generated_mesh_cfg = inference_cfg.get(
            "generated_mesh",
            "{output_dir}/{input_stem}_hunyuan3d.obj",
        )
        stem_for_paths = (
            Path(input_image).stem
            if (use_image and input_image)
            else _prompt_stem(input_prompt or "")
        )
        generated_mesh_formatted = generated_mesh_cfg.format(
            input_stem=stem_for_paths,
            output_dir=str(out_dir.resolve()),
        )
        generated_mesh = Path(generated_mesh_formatted)
        if not generated_mesh.is_absolute():
            generated_mesh = Path.cwd() / generated_mesh

        out_dir_abs = out_dir.resolve()
        out_mesh_abs = out_mesh.resolve()

        prompt_file: Path | None = None
        format_kwargs: dict[str, str] = {
            "input_image": str(Path(input_image).resolve()) if (use_image and input_image) else "",
            "input_stem": stem_for_paths,
            "output_dir": str(out_dir_abs),
            "output_mesh": str(out_mesh_abs),
            "prompt_file": "",
        }
        if not use_image:
            prompt_file = out_dir_abs / f"_genlab_hunyuan_prompt_{stem_for_paths}.txt"
            prompt_file.write_text(str(input_prompt).strip(), encoding="utf-8")
            format_kwargs["prompt_file"] = str(prompt_file)

        cmd = command_template.format(**format_kwargs)
        log_step(f"[Hunyuan3D-2.1][real-mode] Running command: {cmd}")

        env = os.environ.copy()
        env.update(inference_cfg.get("env", {}))

        try:
            completed = subprocess.run(
                shlex.split(cmd),
                cwd=str(repo_path),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                log_step(f"[Hunyuan3D-2.1][real-mode] stdout:\n{completed.stdout.strip()}")
            if completed.stderr.strip():
                log_step(f"[Hunyuan3D-2.1][real-mode] stderr:\n{completed.stderr.strip()}")
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "").strip()[-1500:]
            raise RuntimeError(
                "[Hunyuan3D-2.1][real-mode] Command failed.\n"
                f"Command: {cmd}\n"
                f"Exit code: {exc.returncode}\n"
                f"Stderr (tail): {stderr_tail}\n"
                "Verify Hunyuan3D-2.1 install (see README), HF mirror env, and GPU memory."
            ) from exc

        if not generated_mesh.exists():
            raise FileNotFoundError(
                "[Hunyuan3D-2.1][real-mode] Expected mesh was not found: "
                f"{generated_mesh}. Check models.hunyuan3d.inference.generated_mesh and the command."
            )

        if generated_mesh.resolve() != out_mesh_abs:
            shutil.copy2(generated_mesh, out_mesh_abs)
            log_step(
                "[Hunyuan3D-2.1][real-mode] Copied generated mesh to pipeline output path: "
                f"{out_mesh_abs}"
            )

        return str(out_mesh_abs)
