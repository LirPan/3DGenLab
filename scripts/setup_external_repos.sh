#!/usr/bin/env bash
set -euo pipefail

mkdir -p external

if [ -d "external/TripoSR/.git" ]; then
  echo "[Setup] external/TripoSR already exists. Skipping clone."
else
  echo "[Setup] Cloning TripoSR into external/TripoSR ..."
  git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
fi

echo "[Setup] Done. TripoSR repository is ready at external/TripoSR."
echo "[Setup] Next: install TripoSR GPU dependencies inside external/TripoSR per upstream docs."
