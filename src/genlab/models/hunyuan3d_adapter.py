from __future__ import annotations

from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.utils import write_minimal_cube_obj


class Hunyuan3DAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="hunyuan3d")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        print("[Hunyuan3D-2.1] setup complete (placeholder)")

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["hunyuan3d"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        out_mesh = out_dir / "hunyuan3d_result.obj"

        if self.dry_run:
            write_minimal_cube_obj(out_mesh)
            print(f"[Hunyuan3D-2.1][dry-run] Wrote placeholder mesh: {out_mesh}")
            return str(out_mesh)

        cmd = (
            f"python {model_cfg['repo_path']}/inference.py "
            f"--image {input_image or ''} "
            f"--prompt \"{input_prompt or ''}\" "
            f"--output {out_dir}"
        )
        # TODO: Adapt to Hunyuan3D-2.1 official command(s) and required arguments.
        # TODO: Execute external command with subprocess and parse produced mesh path.
        print(f"[Hunyuan3D-2.1][real-mode] Placeholder command: {cmd}")
        return str(out_mesh)
