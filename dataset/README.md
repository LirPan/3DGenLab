# Evaluation inputs

The [case manifest](metadata/cases.json) defines the input file, case group and supported adapters for each evaluation case.

| Group | Inputs | Purpose |
| --- | --- | --- |
| `main_image_set` | Robot, chair, sneaker and mug photos | Compare all four models on the same images |
| `text_extension_set` | Low-poly robot and treasure chest prompts | Exercise the Hunyuan3D and TRELLIS text pipelines |
| `challenge_set` | Glass and desk lamp photos | Inspect failure cases involving transparency or thin structures |

See the [image sources and credits](metadata/sources.md) for the photos used in the dataset. The sample in [`inputs/`](../inputs/) is used by the local smoke test.
