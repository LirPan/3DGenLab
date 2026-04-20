from __future__ import annotations

import shlex
import subprocess
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

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["triposr"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        ensure_dir(out_dir)
        out_mesh = out_dir / "example_triposr.obj"

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

        inference_cfg = model_cfg.get("inference", {})
        command_template = inference_cfg.get("command")
        if not command_template:
            raise ValueError(
                "[TripoSR][real-mode] Missing models.triposr.inference.command in config."
            )

        expected_mesh_cfg = inference_cfg.get("expected_mesh")
        if expected_mesh_cfg:
            out_mesh = Path(expected_mesh_cfg)
            if not out_mesh.is_absolute():
                out_mesh = Path.cwd() / out_mesh
            ensure_dir(out_mesh.parent)

        out_dir_abs = out_dir.resolve()
        out_mesh_abs = out_mesh.resolve()
        cmd = command_template.format(
            input_image=str(Path(input_image).resolve()),
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

        if not out_mesh_abs.exists():
            raise FileNotFoundError(
                "[TripoSR][real-mode] Command finished but expected mesh was not found: "
                f"{out_mesh_abs}. Check models.triposr.inference.expected_mesh and command."
            )

        return str(out_mesh_abs)
