#!/usr/bin/env bash
set -euo pipefail

ROOT="/DATA/disk1/yjh/workspace0/model/3DGenLab"
PY="$ROOT/.venvs/trellis310/bin/python"
LOG_PREFIX="[trellis-bg]"

# Mirror-first strategy (China mainland friendly)
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_FALLBACK="https://pypi.org/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
HF_ENDPOINT="https://hf-mirror.com"

# Make pip more resilient to slow/unstable links.
export PIP_DEFAULT_TIMEOUT=120
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_ENDPOINT

echo "$LOG_PREFIX start: $(date -Iseconds)"

if [ ! -x "$PY" ]; then
  echo "$LOG_PREFIX ERROR: python not found at $PY"
  exit 1
fi

"$PY" -V

# Core stack for TRELLIS (Python 3.10 + CUDA 12.1)
"$PY" -m pip install --retries 20 --index-url "$TORCH_INDEX" torch==2.4.0 torchvision==0.19.0
"$PY" -m pip install --retries 20 --index-url "$TORCH_INDEX" xformers==0.0.27.post2
"$PY" -m pip install --retries 20 -i "$PIP_MIRROR" --extra-index-url "$PIP_FALLBACK" spconv-cu121

# Basic runtime deps used by TRELLIS examples/pipeline
"$PY" -m pip install --retries 20 -i "$PIP_MIRROR" --extra-index-url "$PIP_FALLBACK" pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
"$PY" -m pip install --retries 20 -i "$PIP_MIRROR" --extra-index-url "$PIP_FALLBACK" "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

echo "$LOG_PREFIX done: $(date -Iseconds)"
