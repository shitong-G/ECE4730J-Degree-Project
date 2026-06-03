# Third-Party Dependencies

## RT-DETR

This project uses [RT-DETR](https://github.com/lyuwenyu/RT-DETR) as an **upstream reference and export source**, not as vendored source in the repository root.

### Recommended: Git Clone

```bash
bash scripts/clone_rtdetr.sh
```

This creates:

```
third_party/RT-DETR/   # cloned repository (gitignored)
```

### Manual Clone

```bash
mkdir -p third_party
git clone https://github.com/lyuwenyu/RT-DETR.git third_party/RT-DETR
```

### Usage in This Project

1. **Training / export** — use RT-DETR tooling to export ONNX (see `scripts/export_model_onnx.sh`).
2. **Runtime inference** — this repo loads exported `.onnx` via `ONNXRTDETREngine` only.
3. **Fallback** — if full RT-DETR is too heavy on Pi, use RT-DETR-R18 or a lightweight detector with the same `BaseInferenceEngine` interface.

### Submodule Alternative (optional)

```bash
git submodule add https://github.com/lyuwenyu/RT-DETR.git third_party/RT-DETR
git submodule update --init --recursive
```

Do **not** copy RT-DETR Python sources into `src/scene_runtime/`. Keep a thin adapter in `src/scene_runtime/inference/`.
