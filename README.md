# 3DGenLab

3DGenLab is a local scaffold for comparing four open-source 3D generation models through one consistent interface:

- TripoSR
- InstantMesh
- Hunyuan3D-2.1
- TRELLIS

Current status: dry-run pipeline is complete, and TripoSR + InstantMesh + **Hunyuan3D-2.1 (shape / text→T2I→shape)** + **TRELLIS (headless image/text entrypoint)** are integrated via adapter subprocess hooks for GPU server execution.

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

## Evaluation Dataset Layout

For practical multi-model comparison, this repository now uses a small dataset plan under `dataset/`:

- `dataset/images/`: image-conditioned cases (shared baseline across models)
- `dataset/prompts/`: text extension cases
- `dataset/metadata/cases.json`: case definitions (`case_id`, `type`, `input_path`, `group`, `category`, `intended_models`)

Groups:

- `main_image_set` (4): `robot.png`, `chair.png`, `sneaker.png`, `mug.png`
- `text_extension_set` (2): `robot_lowpoly.txt`, `treasure_chest.txt`
- `challenge_set` (2): `glass.png`, `lamp.png`

Notes:

- `inputs/` remains the smoke-test compatibility entry.
- `dataset/` is the recommended entry for formal evaluation and README/GitHub demos.
- Current image files in `dataset/images/` are lightweight placeholders and can be replaced in-place.

Quick examples:

```bash
# Single model dry-run on dataset image case
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --model triposr \
  --input dataset/images/robot.png \
  --dry-run \
  --benchmark

# Text extension dry-run (for text-capable adapters)
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --model trellis \
  --prompt dataset/prompts/robot_lowpoly.txt \
  --dry-run \
  --benchmark
```

## Dataset-driven Evaluation

Use `dataset/metadata/cases.json` as the canonical manifest and run by case, group, or all cases:

```bash
# 1) Run one case_id from cases.json
python scripts/run_dataset_eval.py --case-id robot_img --dry-run

# 2) Run one group from cases.json
python scripts/run_dataset_eval.py --group main_image_set --dry-run

# 3) Run all dataset cases
python scripts/run_dataset_eval.py --all --dry-run
```

Real GPU run example:

```bash
HF_ENDPOINT=https://hf-mirror.com .venvs/genlab/bin/python scripts/run_dataset_eval.py \
  --group main_image_set \
  --config configs/triposr_gpu.yaml \
  --no-dry-run \
  --benchmark
```

Behavior:

- Respects per-case `intended_models` from `dataset/metadata/cases.json`.
- Image cases use image paths from metadata; text cases read prompt files from metadata.
- Final outputs are normalized to:
  - `outputs/<model_name>/<case_id>_<model_name>.<ext>`
- Per-run report:
  - `outputs/reports/<case_id>_<model_name>.json`
- Batch aggregate reports:
  - `outputs/reports/comparison_summary.json`
  - `outputs/reports/comparison_summary.csv`
- Optional render outputs (when enabled):
  - `outputs/renders/<model_name>/<case_id>_<model_name>.png`
- Per-run report render fields:
  - `render_path`, `render_success`, `render_error`

Failed runs are also kept in per-run reports and comparison summaries, so one model failure does not stop the batch.

## Blender Rendering (Optional)

The dataset runner can render a preview image for each successful mesh by calling Blender in headless mode.

- Default mode is system executable (`blender`).
- Python `bpy` mode is also supported through `scripts/render_blender.py` and falls back to system mode when unavailable.
- Render settings are configured via `render_settings` in YAML.

Enable render during dataset evaluation:

```bash
/DATA/disk1/yjh/workspace0/model/3DGenLab/.venvs/trellis310/bin/python scripts/run_dataset_eval.py \
  --group main_image_set \
  --config configs/triposr_gpu.yaml \
  --no-dry-run \
  --benchmark \
  --render
```

Render one existing mesh directly:

```bash
python scripts/render_blender.py \
  --mesh outputs/trellis/robot_img_trellis.obj \
  --output outputs/renders/trellis/robot_img_trellis.png \
  --mode system \
  --blender-bin blender
```

Troubleshooting:

- If `blender` is not found, set `render_settings.blender_bin` to an absolute Blender path.
- For headless servers, always use `-b` mode (already used by integration).
- If render fails, mesh success is preserved and error details are recorded in `render_error`.

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

## Real Model Rollout Order (main/text/challenge)

Recommended execution order for stable server experiments:

1. Run `main_image_set` first for all four models (baseline matrix).
2. Run `text_extension_set` for text-capable models (`hunyuan3d`, `trellis`).
3. Run `challenge_set` for failure analysis (transparent/thin structures).
4. Aggregate reports from `outputs/reports` for cross-model summary.

## GPU Server Deployment

TripoSR, InstantMesh, Hunyuan3D-2.1, and TRELLIS real inference are integrated via subprocess calls into `external/*` checkouts. TRELLIS uses a lightweight headless helper (`scripts/trellis_genlab_infer.py`) and keeps dry-run behavior unchanged for local development.

This repository does not train foundation models from scratch and does not vendor full upstream model source code.

`external/` and `outputs/` are git-ignored by design.

## Supported Models (Current Status)

- TripoSR: dry-run + real image-to-3D (adapter integrated; server env currently blocked if `.venvs/triposr` was copied from another machine)
- InstantMesh: dry-run + real image-to-3D (adapter integrated; server env currently blocked if `.venvs/instantmesh` was copied from another machine)
- Hunyuan3D-2.1: dry-run + real image/text-to-3D (adapter integrated; server env currently needs complete Python deps such as `timm`)
- TRELLIS: dry-run + real image/text headless entrypoint (**verified on server**)

All adapters normalize final outputs into:

- `outputs/<model_name>/<input_stem>_<model_name>.obj`

## TripoSR Real Inference (GPU Only)

TripoSR real inference is integrated through the existing adapter-based pipeline.

- Local Mac workflow remains dry-run only (`configs/default.yaml`, `dry_run: true`).
- Real TripoSR inference is enabled with `configs/triposr_gpu.yaml` (`dry_run: false`).
- 3DGenLab does not install CUDA or TripoSR heavy dependencies automatically.
- Use a dedicated environment for `external/TripoSR` dependencies, separate from the lightweight 3DGenLab environment.
- Ensure the interpreter in `models.triposr.inference.command` points to a venv created on this server. If you copied `.venvs/triposr` from another machine, `bin/python` may be a broken absolute symlink.
- If `external/TripoSR` is missing or the configured command fails, the pipeline exits with a clear error message.
- Output mesh path convention is:
  - `outputs/triposr/<input_stem>_triposr.obj`

## InstantMesh Real Inference (GPU Only)

- Use a dedicated environment for `external/InstantMesh` dependencies (for example `.venvs/instantmesh`).
- Ensure the interpreter in `models.instantmesh.inference.command` points to a local, valid Python binary. Copied venvs can fail with `No such file or directory` even when the file exists.
- If your server cannot reach `huggingface.co` directly, set `HF_ENDPOINT` (for example `https://hf-mirror.com`) via `models.instantmesh.inference.env` in `configs/triposr_gpu.yaml`.
- InstantMesh runtime still needs online/offline access to model assets (for example `sudo-ai/zero123plus-v1.2` and `TencentARC/InstantMesh` files). If the server cannot reach Hugging Face or mirror endpoints, prepare local model cache first.
- Output mesh path convention is:
  - `outputs/instantmesh/<input_stem>_instantmesh.obj`
- Some diffusion stacks are sensitive to extremely small synthetic RGB images. Prefer a normal-resolution RGB input (or use `external/InstantMesh/examples/*.png` for smoke checks).

## Hunyuan3D-2.1 Real Inference (GPU Only)

Hunyuan3D-2.1 is wired like the other models: the adapter runs a **headless shape pipeline** (untreated mesh as OBJ) via [`scripts/hunyuan3d_genlab_infer.py`](scripts/hunyuan3d_genlab_infer.py) inside `external/Hunyuan3D-2.1`. This avoids the full Gradio stack and skips upstream **texture / PBR** compilation by default (only geometry for the benchmark path).

- **External checkout**: `bash scripts/setup_external_repos.sh` clones `external/Hunyuan3D-2.1` (SSH, same pattern as TripoSR / InstantMesh).
- **Dedicated venv**: use `.venvs/hunyuan3d` (or your own) with PyTorch + deps from upstream [`external/Hunyuan3D-2.1/requirements.txt`](external/Hunyuan3D-2.1/requirements.txt). Upstream documents Python 3.10 and CUDA PyTorch (see their README). Shape-only usage still needs `hy3dshape`, `diffusers`, `rembg` / `onnxruntime`, `trimesh`, etc.
- **HF mirror**: `models.hunyuan3d.inference.env.HF_ENDPOINT` is set to `https://hf-mirror.com` in YAML (same idea as InstantMesh).
- **Runtime isolation**: the default GPU config runs Hunyuan via `env -u LD_LIBRARY_PATH` to avoid host-level Qt/GL library pollution during `pymeshlab` import.
- **Common missing deps**: if you see `ModuleNotFoundError` (for example `timm`), install missing packages into `.venvs/hunyuan3d` following upstream requirements.
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

## Server Troubleshooting (Current)

- `No such file or directory` for `.venvs/<model>/bin/python` usually means the venv was copied from another host and contains broken absolute symlinks. Recreate that venv on this server.
- Hunyuan3D failures around `pymeshlab`/Qt are typically host runtime conflicts; keep `LD_LIBRARY_PATH` isolated for the inference command.
- If Hunyuan3D reports `ModuleNotFoundError` (for example `timm`), install missing dependencies in `.venvs/hunyuan3d`.
