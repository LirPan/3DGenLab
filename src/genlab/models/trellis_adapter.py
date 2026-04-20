from __future__ import annotations

from pathlib import Path

from genlab.models.base import Base3DGenModel
from genlab.utils import write_minimal_cube_obj


class TrellisAdapter(Base3DGenModel):
    def __init__(self, config: dict, dry_run: bool = True):
        super().__init__(name="trellis")
        self.config = config
        self.dry_run = dry_run

    def setup(self) -> None:
        print("[TRELLIS] setup complete (placeholder)")

    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        model_cfg = self.config["models"]["trellis"]
        out_dir = Path(output_dir or model_cfg["output_dir"])
        out_mesh = out_dir / "trellis_result.obj"

        if self.dry_run:
            write_minimal_cube_obj(out_mesh)
            print(f"[TRELLIS][dry-run] Wrote placeholder mesh: {out_mesh}")
            return str(out_mesh)

        if input_image:
            cmd = (
                f"python {model_cfg['repo_path']}/example.py "
                f"--image {input_image} --output {out_dir}"
            )
        else:
            cmd = f"python {model_cfg['repo_path']}/app.py"

        # TODO: Verify TRELLIS official CLI/API entrypoint and expected flags.
        # TODO: Replace placeholder commands with repository-supported invocation.
        # Example placeholders:
        #   python external/TRELLIS/app.py
        #   python external/TRELLIS/example.py --image input.png --output output_dir
        print(f"[TRELLIS][real-mode] Placeholder command: {cmd}")
        if input_prompt:
            print(f"[TRELLIS][real-mode] Prompt input received: {input_prompt}")
        return str(out_mesh)
