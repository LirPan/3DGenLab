# GPU setup

Run these commands from the repository root on a Linux host with a CUDA-capable GPU. Keep the lightweight 3DGenLab environment separate from each model runtime.

## Install the runner and upstream projects

```bash
bash scripts/bootstrap.sh
bash scripts/setup_external_repos.sh
```

The upstream setup script clones the four public model repositories into `external/` over SSH. If your GitHub setup uses HTTPS, clone the [upstream repositories](../README.md#models) into `external/TripoSR`, `external/InstantMesh`, `external/Hunyuan3D-2.1` and `external/TRELLIS` respectively.

Install each model's CUDA and Python dependencies according to its upstream instructions. Prepare the model weights on the GPU host. Restricted hosts may also need an offline cache for assets such as InstantMesh's Zero123++ and model checkpoints.

## Configure the model commands

Use [`configs/triposr_gpu.yaml`](../configs/triposr_gpu.yaml) as the example configuration for image inference across all four adapters. Before running it on another host, update each model's inference command to point to that host's interpreter. The sample uses `.venvs/trellis310` for TripoSR, InstantMesh and TRELLIS and `.venvs/hunyuan3d` for Hunyuan3D. Recreate environments on the target host so the interpreters point to valid local paths.

- **InstantMesh** — adjust the `PYTHONPATH` entry if your nvdiffrast checkout is elsewhere. Its example command uses the Linux `timeout` utility.
- **Hunyuan3D-2.1** — the headless helper runs the upstream shape pipeline. For text input, it first generates a reference image with HunyuanDiT. The example unsets `LD_LIBRARY_PATH` for the subprocess to avoid conflicts with system Qt or GL libraries.
- **TRELLIS** — set `model_path` to the image or text checkpoint for the selected configuration. The example configs contain paths from the original GPU host.
- **Model downloads** — adjust `HF_ENDPOINT` in the configuration if you use a mirror, or make the model assets available in the local cache.

## Run a model or a batch

Run one image through TripoSR and collect mesh metrics.

```bash
.venv/bin/python scripts/run_pipeline.py \
  --config configs/triposr_gpu.yaml --model triposr \
  --input dataset/images/robot.png --benchmark
```

Run the shared image group across the four configured adapters.

```bash
.venv/bin/python scripts/run_dataset_eval.py \
  --config configs/triposr_gpu.yaml --group main_image_set \
  --no-dry-run --no-render --benchmark
```

For text cases, use `--group text_extension_set` with [`configs/hunyuan3d_text_gpu.yaml`](../configs/hunyuan3d_text_gpu.yaml) or [`configs/trellis_text_gpu.yaml`](../configs/trellis_text_gpu.yaml). To inspect harder cases such as transparent glass, run `--group challenge_set` with the image configuration.

## Render with Blender

Install Blender and put its executable on `PATH`, or set `render_settings.blender_bin` in the configuration. Use `--render` on the batch command and select the adapters to render in `render_settings.models`. The example image GPU configuration selects TRELLIS previews.

To render an existing mesh directly, adjust the paths in the following command.

```bash
.venv/bin/python scripts/render_blender.py \
  --mesh outputs/trellis/robot_img_trellis.obj \
  --output outputs/renders/trellis/robot_img_trellis.png \
  --mode system --blender-bin blender
```

The renderer can also use the Python `bpy` module and fall back to the Blender executable. The batch reports retain render errors alongside any mesh that was generated successfully.
