from __future__ import annotations

from genlab.models.hunyuan3d_adapter import Hunyuan3DAdapter
from genlab.models.instantmesh_adapter import InstantMeshAdapter
from genlab.models.trellis_adapter import TrellisAdapter
from genlab.models.triposr_adapter import TripoSRAdapter

MODEL_REGISTRY = {
    "triposr": TripoSRAdapter,
    "instantmesh": InstantMeshAdapter,
    "hunyuan3d": Hunyuan3DAdapter,
    "trellis": TrellisAdapter,
}


def get_model(model_name: str, config: dict, dry_run: bool = True):
    name = model_name.strip().lower()

    if name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model name '{model_name}'. Available models: {supported}"
        )

    return MODEL_REGISTRY[name](config=config, dry_run=dry_run)
