#!/usr/bin/env bash
set -euo pipefail

ROOT="/DATA/disk1/yjh/workspace0/model/3DGenLab"
EXT="$ROOT/external/TRELLIS"
PY="$ROOT/.venvs/trellis310/bin/python"
IMG="$ROOT/inputs/images/example.png"
OUT_DIR="$ROOT/outputs/trellis"
OUT_OBJ="$OUT_DIR/example_trellis.obj"
WHEELHOUSE="/DATA/disk1/yjh/workspace0/data/genlab-models/offline-packages/trellis-cu121-py310"
MODEL_IMAGE="/DATA/disk1/yjh/workspace0/data/genlab-models/microsoft/TRELLIS-image-large"
MODEL_TEXT="/DATA/disk1/yjh/workspace0/data/genlab-models/microsoft/TRELLIS-text-base"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR"
echo "[trellis-overnight] start $(date -Iseconds)"

if [ ! -x "$PY" ]; then
  echo "[trellis-overnight] ERROR: python not found: $PY"
  exit 1
fi

if [ ! -d "$EXT/.git" ]; then
  echo "[trellis-overnight] cloning TRELLIS repo..."
  mkdir -p "$ROOT/external"
  git clone https://github.com/microsoft/TRELLIS.git "$EXT"
fi

mkdir -p "$EXT/local_models"
ln -sfn "$MODEL_IMAGE" "$EXT/local_models/TRELLIS-image-large"
ln -sfn "$MODEL_TEXT" "$EXT/local_models/TRELLIS-text-base"

if [ -d "$WHEELHOUSE" ]; then
  echo "[trellis-overnight] installing offline wheelhouse deps..."
  "$PY" -m pip install --no-index --find-links "$WHEELHOUSE" \
    torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 spconv-cu121 \
    imageio imageio-ffmpeg tqdm easydict scipy ninja rembg onnxruntime trimesh xatlas \
    pyvista pymeshfix igraph transformers || true
fi

echo "[trellis-overnight] installing missing online deps..."
"$PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "numpy<2" "opencv-python-headless<4.10" open3d plyfile tomli exceptiongroup sniffio \
  "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

echo "[trellis-overnight] running inference..."
ATTN_BACKEND=xformers SPCONV_ALGO=native HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH="$EXT${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" - <<'PYCODE'
from pathlib import Path
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

root = Path("/DATA/disk1/yjh/workspace0/model/3DGenLab")
img_path = root / "inputs/images/example.png"
out_obj = root / "outputs/trellis/example_trellis.obj"
model_dir = Path("/DATA/disk1/yjh/workspace0/data/genlab-models/microsoft/TRELLIS-image-large")

pipeline = TrellisImageTo3DPipeline.from_pretrained(str(model_dir))
pipeline.cuda()
outputs = pipeline.run(
    Image.open(img_path),
    seed=1,
    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
    slat_sampler_params={"steps": 12, "cfg_strength": 3.0},
)
mesh = outputs["mesh"][0]
verts = mesh.vertices.detach().cpu().numpy()
faces = mesh.faces.detach().cpu().numpy()
out_obj.parent.mkdir(parents=True, exist_ok=True)
with open(out_obj, "w", encoding="utf-8") as f:
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for tri in faces:
        f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
print(f"[trellis-overnight] output={out_obj}")
print(f"[trellis-overnight] vertices={len(verts)} faces={len(faces)}")
PYCODE

ls -lh "$OUT_OBJ"
echo "[trellis-overnight] done $(date -Iseconds)"
