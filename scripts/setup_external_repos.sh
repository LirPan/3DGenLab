#!/usr/bin/env bash
set -euo pipefail

mkdir -p external

git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
git clone https://github.com/TencentARC/InstantMesh.git external/InstantMesh
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git external/Hunyuan3D-2.1
git clone https://github.com/microsoft/TRELLIS.git external/TRELLIS

echo "External repositories cloned into ./external"
