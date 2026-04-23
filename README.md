# 3DGenLab

3DGenLab is a local scaffold for comparing four open-source 3D generation models through one consistent interface:

- TripoSR
- InstantMesh
- Hunyuan3D-2.1
- TRELLIS

Current status: dry-run pipeline is complete, and TripoSR + InstantMesh + **Hunyuan3D-2.1 (shape / text→T2I→shape)** real inference hooks are integrated for GPU server execution.

## Local Development on Mac

This repository is designed to run lightweight dry-run development on macOS (including MacBook Air). It does not run real CUDA inference locally.

Dry-run mode only generates dummy cube meshes for pipeline validation.

The package uses a `src/` layout and is defined in `pyproject.toml` as package `genlab`.

## Quick Start

Python 3.9+ is required.

```bash
bash scripts/bootstrap.sh
python scripts/smoke_test.py
```

## Dry-run Mode

Dry-run mode does not call external model repositories. It writes a valid dummy cube OBJ (8 vertices, 12 triangular faces) for each model adapter.

Run one model in dry-run mode:

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --model triposr \
  --input inputs/images/example.png \
  --prompt inputs/prompts/example.txt \
  --dry-run \
  --benchmark
```

Run all models in dry-run mode:

```bash
python scripts/run_all_models.py \
  --config configs/default.yaml \
  --input inputs/images/example.png \
  --prompt inputs/prompts/example.txt \
  --dry-run \
  --benchmark
```

Expected dry-run meshes:

- `outputs/triposr/example_triposr.obj`
- `outputs/instantmesh/example_instantmesh.obj`
- `outputs/hunyuan3d/example_hunyuan3d.obj`
- `outputs/trellis/example_trellis.obj`

## Smoke Test

Run an end-to-end local smoke check:

```bash
python scripts/smoke_test.py
```

What smoke test validates:

- creates `inputs/images/example.png` if missing
- creates `inputs/prompts/example.txt` if missing
- runs all 4 adapters in dry-run mode
- writes 4 OBJ files to model-specific output folders
- writes benchmark reports to `outputs/reports/smoke_*_metrics.json`
- prints `SMOKE TEST PASSED` on success

## Optional Conda Setup

If you prefer Miniforge/Conda on Mac:

```bash
conda create -n genlab python=3.10 -y
conda activate genlab
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/smoke_test.py
```

Conda on Mac is only for lightweight local pipeline development. Real CUDA inference belongs on a Linux GPU server.

## GPU Server Deployment

TripoSR, InstantMesh, and Hunyuan3D-2.1 real inference are integrated via subprocess calls into `external/*` checkouts. TRELLIS remains a dry-run placeholder in this stage.

This repository does not train foundation models from scratch and does not vendor full upstream model source code.

`external/` and `outputs/` are git-ignored by design.

## TripoSR Real Inference (GPU Only)

TripoSR real inference is integrated through the existing adapter-based pipeline.

- Local Mac workflow remains dry-run only (`configs/default.yaml`, `dry_run: true`).
- Real TripoSR inference is enabled with `configs/triposr_gpu.yaml` (`dry_run: false`).
- 3DGenLab does not install CUDA or TripoSR heavy dependencies automatically.
- Use a dedicated environment for `external/TripoSR` dependencies, separate from the lightweight 3DGenLab environment.
- If `external/TripoSR` is missing or the configured command fails, the pipeline exits with a clear error message.
- Output mesh path convention is:
  - `outputs/triposr/<input_stem>_triposr.obj`

## InstantMesh Real Inference (GPU Only)

- Use a dedicated environment for `external/InstantMesh` dependencies (for example `.venvs/instantmesh`).
- If your server cannot reach `huggingface.co` directly, set `HF_ENDPOINT` (for example `https://hf-mirror.com`) via `models.instantmesh.inference.env` in `configs/triposr_gpu.yaml`.
- Output mesh path convention is:
  - `outputs/instantmesh/<input_stem>_instantmesh.obj`
- Some diffusion stacks are sensitive to extremely small synthetic RGB images. Prefer a normal-resolution RGB input (or use `external/InstantMesh/examples/*.png` for smoke checks).

## Hunyuan3D-2.1 Real Inference (GPU Only)

Hunyuan3D-2.1 is wired like the other models: the adapter runs a **headless shape pipeline** (untreated mesh as OBJ) via [`scripts/hunyuan3d_genlab_infer.py`](scripts/hunyuan3d_genlab_infer.py) inside `external/Hunyuan3D-2.1`. This avoids the full Gradio stack and skips upstream **texture / PBR** compilation by default (only geometry for the benchmark path).

- **External checkout**: `bash scripts/setup_external_repos.sh` clones `external/Hunyuan3D-2.1` (SSH, same pattern as TripoSR / InstantMesh).
- **Dedicated venv**: use `.venvs/hunyuan3d` (or your own) with PyTorch + deps from upstream [`external/Hunyuan3D-2.1/requirements.txt`](external/Hunyuan3D-2.1/requirements.txt). Upstream documents Python 3.10 and CUDA PyTorch (see their README). Shape-only usage still needs `hy3dshape`, `diffusers`, `rembg` / `onnxruntime`, `trimesh`, etc.
- **HF mirror**: `models.hunyuan3d.inference.env.HF_ENDPOINT` is set to `https://hf-mirror.com` in YAML (same idea as InstantMesh).
- **Outputs**: `outputs/hunyuan3d/<stem>_hunyuan3d.obj`
- **Image vs text**:
  - **Image-conditioned** (default): set `models.hunyuan3d.inference.prefer: image` (default). Uses `command_image`.
  - **Text-conditioned**: uses HunyuanDiT **text-to-image** (same Hub id as upstream Gradio: `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled`), then shape generation. If your config still has both `input_image` and a prompt, set `prefer: text` so the prompt wins, **or** set `input_image: null` in YAML for a text-only run.

### GPU Server Commands

```bash
# 1) Clone this repository
git clone <your-3dgenlab-repo-url>
cd 3DGenLab

# 2) Clone external model repositories (idempotent)
bash scripts/setup_external_repos.sh

# 3) Prepare 3DGenLab environment (lightweight, for YAML + pipeline + benchmark)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# 4) Install TripoSR GPU dependencies manually
#    Follow upstream docs inside external/TripoSR
#    Recommended: use a dedicated venv/conda env for TripoSR itself

# 5) Run TripoSR real inference + benchmark
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml \
  --model triposr \
  --input inputs/images/example.png \
  --benchmark

# 6) Install InstantMesh GPU dependencies manually
#    Follow upstream docs inside external/InstantMesh
#    Recommended: use a dedicated venv for InstantMesh (this repo uses .venvs/instantmesh)

# 7) Run InstantMesh real inference + benchmark
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml \
  --model instantmesh \
  --input external/InstantMesh/examples/hatsune_miku.png \
  --benchmark

# 8) Install Hunyuan3D-2.1 dependencies in .venvs/hunyuan3d (follow upstream README / requirements.txt)

# 9) Image → shape (real) + benchmark
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml \
  --model hunyuan3d \
  --input inputs/images/example.png \
  --benchmark

# 10) Text → shape (real): dedicated config (no input image; uses prompt file or --prompt)
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/run_pipeline.py \
  --config configs/hunyuan3d_text_gpu.yaml \
  --model hunyuan3d \
  --prompt "a red ceramic mug on white background" \
  --benchmark
```
