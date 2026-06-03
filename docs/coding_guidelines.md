# Coding Guidelines

## Style

- Python 3.10+
- Type hints on public APIs
- `pathlib.Path` for file paths
- Dataclasses for structured records (`RuntimeAction`, `Detection`)

## Module Boundaries

- **Scene features**: `src/scene_runtime/scene/` — OpenCV + NumPy only
- **Hardware reads**: `src/scene_runtime/device/` — sysfs / vcgencmd isolated here
- **Policies**: `src/scene_runtime/controller/` — no direct hardware access
- **Inference**: `src/scene_runtime/inference/` — ONNX and postprocess only

## Import Safety

All modules must import on non-Pi development machines. Hardware reads return `None` or safe defaults when sysfs is unavailable.

## TODOs

Mark incomplete integrations with `# TODO:` and a short explanation. Do not leave empty stub files.

## Testing

Add unit tests under `tests/` for:

- Pure functions (scene features, policy decisions)
- Logger schema validation
- Config loading

Run: `pytest tests/ -v`

## Commits

Use conventional prefixes: `feat:`, `fix:`, `docs:`, `exp:`, `refactor:`

## Branches

- `feature/scene-estimator`
- `feature/device-monitor`
- `feature/controller`
- `feature/inference-engine`
- `experiment/pi4-baseline`
