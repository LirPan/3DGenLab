# Real Model Rollout Plan (Grouped by Dataset)

This plan uses `dataset/metadata/cases.json` as the source of truth.

## 0. Preconditions

- External repos are ready under `external/`.
- Model environments are ready (`.venvs/trellis310`, `.venvs/hunyuan3d`, etc.).
- Offline assets are in place for network-restricted servers (especially InstantMesh dependencies).

## 1. main_image_set (4 cases, all 4 models)

Cases:

- `dataset/images/robot.png`
- `dataset/images/chair.png`
- `dataset/images/sneaker.png`
- `dataset/images/mug.png`

Models:

- `triposr`
- `instantmesh`
- `hunyuan3d`
- `trellis`

Command template:

```bash
python3 scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml \
  --model <model_name> \
  --input <dataset_image_path> \
  --benchmark
```

Expected result:

- 16 real runs (4 cases x 4 models)
- per-run mesh in `outputs/<model>/`
- per-run report in `outputs/reports/`

## 2. text_extension_set (2 cases, text-capable models)

Cases:

- `dataset/prompts/robot_lowpoly.txt`
- `dataset/prompts/treasure_chest.txt`

Models:

- `hunyuan3d`
- `trellis`

Command templates:

```bash
# Hunyuan3D text mode
python3 scripts/run_pipeline.py \
  --config configs/hunyuan3d_text_gpu.yaml \
  --model hunyuan3d \
  --prompt <dataset_prompt_path> \
  --benchmark

# TRELLIS text mode (requires prefer=text in selected config)
python3 scripts/run_pipeline.py \
  --config configs/default.yaml \
  --model trellis \
  --prompt <dataset_prompt_path> \
  --benchmark
```

## 3. challenge_set (2 cases, all 4 models)

Cases:

- `dataset/images/glass.png`
- `dataset/images/lamp.png`

Focus:

- Transparent materials (`glass`)
- Thin structures (`lamp`)

Execution:

- Same command template as `main_image_set`
- Collect failures and compare error types across models

## 4. Suggested Reporting

For each run, capture:

- success/failure
- wall-clock time
- `num_vertices`, `num_faces`
- `is_watertight`
- output mesh path

Store final summary as a markdown table in `docs/` plus raw JSON under `outputs/reports/`.
