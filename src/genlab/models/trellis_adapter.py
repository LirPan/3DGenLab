from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.models.dummy_mesh import write_dummy_cube_obj
from genlab.utils import ensure_dir, log_step


class TrellisAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="trellis")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        log_step("[TRELLIS] setup complete (placeholder)")

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["trellis"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        ensure_dir(out_dir)
        input_stem = Path(input_image).stem if input_image else "example"
        out_mesh = out_dir / f"{input_stem}_trellis.obj"

        if self.dry_run:
            write_dummy_cube_obj(out_mesh)
            log_step(f"[TRELLIS][dry-run] Wrote dummy cube mesh: {out_mesh}")
            return str(out_mesh)

        repo_path = Path(model_cfg.get("repo_path", "external/TRELLIS")).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                "[TRELLIS][real-mode] External repository missing: "
                f"{repo_path}. Run: bash scripts/setup_external_repos.sh"
            )

        if not input_image and not input_prompt:
            raise ValueError(
                "[TRELLIS][real-mode] Need input_image or input_prompt. "
                "Provide --input / --prompt or configure input_image/input_prompt."
            )

        inference_cfg = model_cfg.get("inference", {})
        command_image = inference_cfg.get("command_image") or inference_cfg.get("command")
        command_text = inference_cfg.get("command_text")
        prefer = str(inference_cfg.get("prefer", "image")).strip().lower()
        use_image = bool(input_image) and (prefer != "text" or not input_prompt)
        command_template = command_image if use_image else command_text
        if not command_template:
            raise ValueError(
                "[TRELLIS][real-mode] Missing inference command template. "
                "Set models.trellis.inference.command_image/command_text in config."
            )

        expected_mesh_cfg = inference_cfg.get("expected_mesh")
        if expected_mesh_cfg:
            expected_mesh_formatted = expected_mesh_cfg.format(
                input_stem=input_stem,
                output_dir=str(out_dir.resolve()),
            )
            out_mesh = Path(expected_mesh_formatted)
            if not out_mesh.is_absolute():
                out_mesh = Path.cwd() / out_mesh
        ensure_dir(out_mesh.parent)

        generated_mesh_cfg = inference_cfg.get(
            "generated_mesh",
            "{output_dir}/{input_stem}_trellis.obj",
        )
        generated_mesh_formatted = generated_mesh_cfg.format(
            input_stem=input_stem,
            output_dir=str(out_dir.resolve()),
        )
        generated_mesh = Path(generated_mesh_formatted)
        if not generated_mesh.is_absolute():
            generated_mesh = Path.cwd() / generated_mesh
        ensure_dir(generated_mesh.parent)

        prompt_file = out_dir.resolve() / f"_genlab_trellis_prompt_{input_stem}.txt"
        if input_prompt:
            prompt_file.write_text(str(input_prompt).strip(), encoding="utf-8")

        cmd = command_template.format(
            input_image=str(Path(input_image).resolve()) if input_image else "",
            input_prompt=(input_prompt or "").strip(),
            input_stem=input_stem,
            output_dir=str(out_dir.resolve()),
            output_mesh=str(out_mesh.resolve()),
            prompt_file=str(prompt_file),
            model_path=str(inference_cfg.get("model_path", "")).strip(),
            project_root=str(Path(__file__).resolve().parents[3]),
        )
        log_step(f"[TRELLIS][real-mode] Running command: {cmd}")

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
                log_step(f"[TRELLIS][real-mode] stdout:\n{completed.stdout.strip()}")
            if completed.stderr.strip():
                log_step(f"[TRELLIS][real-mode] stderr:\n{completed.stderr.strip()}")
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "").strip()[-1500:]
            submodule_hint = ""
            if "flexicubes" in stderr_tail:
                submodule_hint = (
                    "\nHint: TRELLIS FlexiCubes submodule seems missing. "
                    "Run: git -C external/TRELLIS submodule update --init --recursive"
                )
            raise RuntimeError(
                "[TRELLIS][real-mode] Command failed.\n"
                f"Command: {cmd}\n"
                f"Exit code: {exc.returncode}\n"
                f"Stderr (tail): {stderr_tail}\n"
                "Verify TRELLIS environment/dependencies and command template in config."
                f"{submodule_hint}"
            ) from exc

        if not generated_mesh.exists():
            raise FileNotFoundError(
                "[TRELLIS][real-mode] Command finished but generated mesh was not found: "
                f"{generated_mesh}. Check models.trellis.inference.generated_mesh and command."
            )

        out_mesh_abs = out_mesh.resolve()
        if generated_mesh.resolve() != out_mesh_abs:
            shutil.copy2(generated_mesh, out_mesh_abs)
            log_step(
                "[TRELLIS][real-mode] Copied generated mesh to pipeline output path: "
                f"{out_mesh_abs}"
            )
        return str(out_mesh_abs)
