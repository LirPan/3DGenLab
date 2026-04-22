#!/usr/bin/env bash
set -euo pipefail

mkdir -p external

if [ -d "external/TripoSR/.git" ]; then
  echo "[Setup] external/TripoSR already exists. Skipping clone."
else
  echo "[Setup] Cloning TripoSR into external/TripoSR ..."
  GIT_SSH_COMMAND='ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no' \
    git clone git@github.com:VAST-AI-Research/TripoSR.git external/TripoSR
fi

if [ -d "external/InstantMesh/.git" ]; then
  echo "[Setup] external/InstantMesh already exists. Skipping clone."
else
  echo "[Setup] Cloning InstantMesh into external/InstantMesh ..."
  GIT_SSH_COMMAND='ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no' \
    git clone git@github.com:TencentARC/InstantMesh.git external/InstantMesh
fi

if [ -d "external/Hunyuan3D-2.1/.git" ]; then
  echo "[Setup] external/Hunyuan3D-2.1 already exists. Skipping clone."
else
  echo "[Setup] Cloning Hunyuan3D-2.1 into external/Hunyuan3D-2.1 ..."
  GIT_SSH_COMMAND='ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no' \
    git clone git@github.com:Tencent-Hunyuan/Hunyuan3D-2.1.git external/Hunyuan3D-2.1
fi

echo "[Setup] Done. External repositories are ready:"
echo "  - external/TripoSR"
echo "  - external/InstantMesh"
echo "  - external/Hunyuan3D-2.1"
echo "[Setup] Next: install each model's GPU dependencies in its dedicated environment."
