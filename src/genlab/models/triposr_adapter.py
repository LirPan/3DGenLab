from __future__ import annotations

import shlex
import subprocess
import shutil
from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.models.dummy_mesh import write_dummy_cube_obj
from genlab.utils import ensure_dir, log_step


class TripoSRAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="triposr")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        log_step("[TripoSR] setup complete (placeholder)")

    def _build_output_mesh_path(
        self,
        out_dir: Path,
        input_image: str | None,
    ) -> Path:
        input_stem = Path(input_image).stem if input_image else "example"
        return out_dir / f"{input_stem}_triposr.obj"

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["triposr"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        ensure_dir(out_dir)
        out_mesh = self._build_output_mesh_path(out_dir=out_dir, input_image=input_image)

        if self.dry_run:
            write_dummy_cube_obj(out_mesh)
            log_step(f"[TripoSR][dry-run] Wrote dummy cube mesh: {out_mesh}")
            return str(out_mesh)

        repo_path = Path(model_cfg.get("repo_path", "external/TripoSR")).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                "[TripoSR][real-mode] External repository missing: "
                f"{repo_path}. Run: bash scripts/setup_external_repos.sh"
            )

        if not input_image:
            raise ValueError(
                "[TripoSR][real-mode] input_image is required. "
                "Provide --input <image_path> or set input_image in config."
            )
        input_image_path = Path(input_image).resolve()
        if not input_image_path.exists():
            raise FileNotFoundError(
                "[TripoSR][real-mode] input_image was not found: "
                f"{input_image_path}"
            )

        inference_cfg = model_cfg.get("inference", {})
        command_template = inference_cfg.get("command")
        if not command_template:
            raise ValueError(
                "[TripoSR][real-mode] Missing models.triposr.inference.command in config."
            )

        expected_mesh_cfg = inference_cfg.get("expected_mesh")
        if expected_mesh_cfg:
            expected_mesh_formatted = expected_mesh_cfg.format(
                input_stem=input_image_path.stem,
                output_dir=str(out_dir.resolve()),
            )
            out_mesh = Path(expected_mesh_formatted)
            if not out_mesh.is_absolute():
                out_mesh = Path.cwd() / out_mesh
        ensure_dir(out_mesh.parent)

        generated_mesh_cfg = inference_cfg.get("generated_mesh", "{output_dir}/0/mesh.obj")
        generated_mesh_formatted = generated_mesh_cfg.format(
            input_stem=input_image_path.stem,
            output_dir=str(out_dir.resolve()),
        )
        generated_mesh = Path(generated_mesh_formatted)
        if not generated_mesh.is_absolute():
            generated_mesh = Path.cwd() / generated_mesh
        ensure_dir(generated_mesh.parent)

        out_dir_abs = out_dir.resolve()
        out_mesh_abs = out_mesh.resolve()
        cmd = command_template.format(
            input_image=str(input_image_path),
            input_stem=input_image_path.stem,
            output_dir=str(out_dir_abs),
            output_mesh=str(out_mesh_abs),
        )
        log_step(f"[TripoSR][real-mode] Running command: {cmd}")
        try:
            completed = subprocess.run(
                shlex.split(cmd),
                cwd=str(repo_path),
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                log_step(f"[TripoSR][real-mode] stdout:\n{completed.stdout.strip()}")
            if completed.stderr.strip():
                log_step(f"[TripoSR][real-mode] stderr:\n{completed.stderr.strip()}")
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "").strip()[-1500:]
            raise RuntimeError(
                "[TripoSR][real-mode] Command failed.\n"
                f"Command: {cmd}\n"
                f"Exit code: {exc.returncode}\n"
                f"Stderr (tail): {stderr_tail}\n"
                "Please verify TripoSR dependencies and command flags in config."
            ) from exc

        if not generated_mesh.exists():
            raise FileNotFoundError(
                "[TripoSR][real-mode] Command finished but generated mesh was not found: "
                f"{generated_mesh}. Check models.triposr.inference.generated_mesh and command."
            )

        if generated_mesh.resolve() != out_mesh_abs:
            shutil.copy2(generated_mesh, out_mesh_abs)
            log_step(
                "[TripoSR][real-mode] Copied generated mesh to pipeline output path: "
                f"{out_mesh_abs}"
            )

        return str(out_mesh_abs)
