from __future__ import annotations

from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.models.dummy_mesh import write_dummy_cube_obj
from genlab.utils import ensure_dir, log_step


class InstantMeshAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="instantmesh")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        log_step("[InstantMesh] setup complete (placeholder)")

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["instantmesh"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        ensure_dir(out_dir)
        out_mesh = out_dir / "instantmesh_result.obj"

        if self.dry_run:
            write_dummy_cube_obj(out_mesh)
            log_step(f"[InstantMesh][dry-run] Wrote dummy cube mesh: {out_mesh}")
            return str(out_mesh)

        cmd = (
            f"python {model_cfg['repo_path']}/run.py "
            f"--image {input_image} "
            f"--output_dir {out_dir}"
        )
        # TODO: Adapt to InstantMesh's official inference command and arguments.
        # TODO: Execute external command with subprocess and parse produced mesh path.
        log_step(f"[InstantMesh][real-mode] Placeholder command: {cmd}")
        return str(out_mesh)
