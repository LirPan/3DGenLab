from __future__ import annotations

from genlab.models.hunyuan3d_adapter import Hunyuan3DAdapter
from genlab.models.instantmesh_adapter import InstantMeshAdapter
from genlab.models.trellis_adapter import TrellisAdapter
from genlab.models.triposr_adapter import TripoSRAdapter


def get_model(model_name: str, config: dict, dry_run: bool = True):
    name = model_name.lower().strip()

    registry = {
        "triposr": TripoSRAdapter,
        "instantmesh": InstantMeshAdapter,
        "hunyuan3d": Hunyuan3DAdapter,
        "trellis": TrellisAdapter,
    }

    if name not in registry:
        supported = ", ".join(registry.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    return registry[name](config=config, dry_run=dry_run)
