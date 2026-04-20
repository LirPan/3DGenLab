# 3DGenLab

3DGenLab is a local scaffold for comparing four open-source 3D generation models through one consistent interface:

- TripoSR
- InstantMesh
- Hunyuan3D-2.1
- TRELLIS

Current status: dry-run pipeline is complete, and TripoSR real inference is integrated for GPU server execution.

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

TripoSR real inference is integrated through `external/TripoSR` via subprocess calls. InstantMesh, Hunyuan3D-2.1, and TRELLIS remain dry-run placeholders in this stage.

This repository does not train foundation models from scratch and does not vendor full upstream model source code.

`external/` and `outputs/` are git-ignored by design.

## TripoSR Real Inference (GPU Only)

Task 3 integrates only TripoSR real inference through the existing adapter-based pipeline.

- Local Mac workflow remains dry-run only (`configs/default.yaml`, `dry_run: true`).
- Real TripoSR inference is enabled with `configs/triposr_gpu.yaml` (`dry_run: false`).
- 3DGenLab does not install CUDA or TripoSR heavy dependencies automatically.
- Use a dedicated environment for `external/TripoSR` dependencies, separate from the lightweight 3DGenLab environment.
- If `external/TripoSR` is missing or the configured command fails, the pipeline exits with a clear error message.
- Output mesh path convention is:
  - `outputs/triposr/<input_stem>_triposr.obj`

### GPU Server Commands

```bash
# 1) Clone this repository
git clone <your-3dgenlab-repo-url>
cd 3DGenLab

# 2) Clone TripoSR only (idempotent)
bash scripts/setup_external_repos.sh

# 3) Prepare 3DGenLab environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# 4) Install TripoSR GPU dependencies manually
#    Follow upstream docs inside external/TripoSR
#    Recommended: use a dedicated venv/conda env for TripoSR itself

# 5) Run TripoSR real inference + benchmark
python scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml \
  --model triposr \
  --input inputs/images/example.png \
  --benchmark
```
