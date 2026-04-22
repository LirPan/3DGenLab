# TRELLIS Handoff (2026-04-22)

## Goal
Run TRELLIS real inference successfully in `3DGenLab` on GPU server, then wire it back into pipeline adapter.

## Project Context
- Repo: `3DGenLab`
- Purpose: adapter-based engineering pipeline (not training foundation models)
- Current status before handoff:
  - Dry-run pipeline passes.
  - `smoke_test.py` passes for 4 models.
  - TripoSR / InstantMesh / Hunyuan3D real-mode integration exists.
  - TRELLIS adapter real-mode is still placeholder.

## What Was Done In This Session
1. Repository hygiene
   - Updated `.gitignore` to include:
     - `.venvs/`
     - `logs/`
     - `.DS_Store`

2. TRELLIS source layout fixes (local-only, under ignored `external/`)
   - Ensured `external/TRELLIS` exists with expected files.
   - Filled missing FlexiCubes submodule content into:
     - `external/TRELLIS/trellis/representations/mesh/flexicubes`

3. Import compatibility workaround (local-only)
   - Added minimal shim to avoid hard dependency break during import:
     - `external/TRELLIS/kaolin/__init__.py`
     - `external/TRELLIS/kaolin/utils/__init__.py`
     - `external/TRELLIS/kaolin/utils/testing.py`
   - This provides `check_tensor` used by FlexiCubes import path.

4. Environment attempts
   - Python 3.12 path was problematic for TRELLIS + spconv.
   - Switched strategy to Python 3.10 venv: `.venvs/trellis310`.
   - Installed `python3.10-venv` on machine where session ran.
   - Installation of heavy deps was blocked by unstable external network.

5. Offline package strategy
   - Prepared wheelhouse workflow from Mac.
   - User successfully uploaded a full wheelhouse (~1.3G, 94 files), including:
     - `torch-2.4.0-cp310-...linux_x86_64.whl`
     - `torchvision-0.19.0-...`
     - `xformers-0.0.27.post2-...`
     - `spconv_cu121-2.3.8-...`
     - many runtime deps

## Major Blocker Encountered
- Session host mismatch:
  - Agent session machine: `ksy01`
  - User upload/login machine often: `ksy03`
- Same public endpoint could land on different backend nodes.
- Result: uploaded offline wheels were not visible from agent-side node.

## Key Files To Review Quickly
- `src/genlab/models/trellis_adapter.py` (currently placeholder real-mode)
- `src/genlab/models/base.py`
- `src/genlab/models/registry.py`
- `scripts/run_pipeline.py`
- `scripts/run_all_models.py`
- `scripts/smoke_test.py`
- `configs/default.yaml`
- `configs/triposr_gpu.yaml`

## Recommended Fast Path On New Server
1. Keep work on one fixed host only.
2. Ensure these exist:
   - `external/TRELLIS`
   - `offline-packages/trellis310` (optional but preferred)
3. Create Python 3.10 env:
   - `.venvs/trellis310`
4. Install core deps:
   - `torch==2.4.0`
   - `torchvision==0.19.0`
   - `xformers==0.0.27.post2`
   - `spconv-cu121`
5. Install runtime deps:
   - `pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers`
6. Run TRELLIS minimal validation:
   - `cd external/TRELLIS`
   - `ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers SPCONV_ALGO=native ../../.venvs/trellis310/bin/python example.py`
7. Validate outputs:
   - `sample.glb`, `sample.ply`, `sample_*.mp4`
8. Then implement real TRELLIS call in adapter:
   - `src/genlab/models/trellis_adapter.py`
   - Preserve dry-run behavior.
   - Keep clear errors for missing repo/deps/model path.

## Suggested Adapter Real-Mode Direction
- Do not launch Gradio app in pipeline.
- Use headless script/command style.
- Prefer image-to-3D first (with local model path support):
  - `/.../TRELLIS-image-large`
- Return deterministic output path:
  - `outputs/trellis/<input_stem>_trellis.obj` (or `.glb`, then convert if needed)

## Notes
- `external/`, `outputs/`, `.venvs/` are intended local-only and ignored.
- No model weights or external repos should be committed.

