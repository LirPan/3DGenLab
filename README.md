# 3DGenLab

3DGenLab is a local scaffold for comparing four open-source 3D generation models through one consistent interface:

- TripoSR
- InstantMesh
- Hunyuan3D-2.1
- TRELLIS

Current status: dry-run only. Real model integration is intentionally deferred.

## Setup

Python 3.9+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package uses a `src/` layout and is defined in `pyproject.toml` as package `genlab`.

## Dry-Run Usage

Dry-run mode does not call external model code. It creates a dummy cube OBJ (8 vertices, 12 triangular faces).

Run one model:

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --model triposr \
  --input inputs/images/example.png \
  --prompt inputs/prompts/example.txt \
  --dry-run \
  --benchmark
```

Run all models:

```bash
python scripts/run_all_models.py \
  --config configs/default.yaml \
  --input inputs/images/example.png \
  --prompt inputs/prompts/example.txt \
  --dry-run \
  --benchmark
```

## Benchmark Usage

Generate a JSON report for one mesh:

```bash
python scripts/benchmark.py \
  --mesh outputs/triposr/triposr_result.obj \
  --output-report outputs/reports/triposr_metrics.json
```

## Smoke Test

Run an end-to-end local smoke check:

```bash
python scripts/smoke_test.py
```

What it does:

- creates `inputs/images/example.png` if missing
- creates `inputs/prompts/example.txt` if missing
- runs all 4 model adapters in dry-run mode
- benchmarks each generated mesh
- verifies 4 OBJ files and 4 JSON reports exist
- prints `SMOKE TEST PASSED` on success

## Expected Outputs

Dry-run outputs are written to model-specific folders:

- `outputs/triposr/`
- `outputs/instantmesh/`
- `outputs/hunyuan3d/`
- `outputs/trellis/`

Benchmark reports are written to `outputs/reports/`.

## External Repositories

Real model integration will be added later by wiring commands to repositories under `external/`.
This project does not vendor or copy upstream model source code.

`external/` and `outputs/` are git-ignored by design.
