# Evaluation Dataset (Planned, Small-Scale)

This folder defines a practical, server-friendly evaluation set for 3DGenLab.

- `images/`: shared image-conditioned cases for fair multi-model comparison.
- `prompts/`: text extension cases for models with text-to-3D support.
- `metadata/cases.json`: machine-readable case registry.

Current placeholder policy:

- `images/*.png` are temporary placeholders copied from `inputs/images/example.png`.
- You can replace each placeholder with a real image while keeping the same filename.
- Prompt files are lightweight starter prompts and can be refined later.
