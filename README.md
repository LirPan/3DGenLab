# 3DGenLab

3DGenLab is a personal Python project by [LirPan](https://github.com/LirPan) for running and comparing image-to-3D models. It takes an image or a text prompt, runs the selected model through a common interface, and saves a mesh, evaluation results and optional Blender preview.

I built it to compare model outputs on the same inputs without having to run each upstream project by hand. Model adapters keep the different GPU environments separate, while the evaluation runner gives meshes and reports consistent names. A failed model run is recorded alongside the successful runs, so the rest of the batch can finish.

## Models

| Model | Inputs supported by 3DGenLab |
| --- | --- |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | Image |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) | Image |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Image or text via text-to-image |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Image or text |

3DGenLab handles orchestration, rendering and evaluation. The model implementations and weights come from their upstream projects.

## Try it locally

Use Python 3.9 or later. The commands below run from the repository root on macOS or Linux.

```bash
git clone https://github.com/LirPan/3DGenLab.git
cd 3DGenLab
bash scripts/bootstrap.sh
.venv/bin/python scripts/smoke_test.py
```

The smoke test writes one placeholder OBJ mesh and one JSON metrics report per adapter, then prints `SMOKE TEST PASSED`. Dry runs use cube meshes to exercise the pipeline without a GPU, model weights or Blender.

To exercise the batch workflow, run the four adapters on a shared group of four images.

```bash
.venv/bin/python scripts/run_dataset_eval.py \
  --group main_image_set --dry-run --no-render --benchmark
```

This dry run writes 16 meshes and the CSV and JSON comparison summaries. Use `--case-id robot_img` to try just one input, or `--all` to include the text prompts and challenge images. The [dataset guide](dataset/README.md) describes the groups and image sources.

## Generate real meshes

Real inference uses Linux, a CUDA-capable GPU, and the upstream model environments and weights. See the [GPU setup guide](docs/GPU_SETUP.md) for model-specific configuration, a batch command and Blender preview options.

The batch runner saves the following under `outputs/`.

| Output | Location |
| --- | --- |
| Generated mesh | `<model>/<case_id>_<model>.<ext>` |
| Optional preview | `renders/<model>/<case_id>_<model>.png` |
| Per-run results | `reports/<case_id>_<model>.json` |
| Batch comparison | `reports/comparison_summary.csv` and `reports/comparison_summary.json` |

Reports record generation time, success or failure, and mesh properties such as vertex and face counts, watertightness and file size. OBJ meshes can be imported into Blender or used in a Unity asset workflow.

## Code layout

| Location | Purpose |
| --- | --- |
| [`src/genlab/models/`](src/genlab/models/) | Adapter interface, model registry and inference wrappers |
| [`src/genlab/render/`](src/genlab/render/) | Headless Blender rendering |
| [`src/genlab/eval/`](src/genlab/eval/) | Mesh metrics |
| [`scripts/`](scripts/) | Single-run, batch evaluation and smoke-test entry points |
| [`configs/`](configs/) and [`dataset/`](dataset/) | Model configurations and shared evaluation inputs |

## License

3DGenLab is released under the [MIT License](LICENSE). Upstream models and [dataset images](dataset/metadata/sources.md) retain their respective licences.
