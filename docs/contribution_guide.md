# Contribution Guide

## Getting Started

1. Fork / clone the repository
2. Create a virtual environment: `bash scripts/setup_env.sh`
3. Install in editable mode: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Verify dry-run: `python scripts/run_experiment.py --dry-run --duration-min 1`

## Branch Naming

| Pattern | Example |
|---------|---------|
| `feature/<module>` | `feature/scene-estimator` |
| `experiment/<topic>` | `experiment/pi4-baseline` |
| `fix/<issue>` | `fix/logger-schema` |

## Commit Messages

```
feat: add entropy-based scene classifier threshold
fix: handle missing thermal sysfs on x86 dev machines
docs: update Pi5 setup instructions
exp: pi4 30min sustained scene_thermal_coadaptive run
refactor: split visual feature extraction helpers
```

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] Type hints on new public APIs
- [ ] No secrets or large binary blobs committed
- [ ] RT-DETR source not copied into root (use `third_party/` clone)
- [ ] Docs updated if behavior or CLI changes

## Issue Labels

| Label | Use |
|-------|-----|
| `module:scene` | Scene workload estimator |
| `module:device` | Device state monitor |
| `module:controller` | Runtime decision controller |
| `module:inference` | RT-DETR / ONNX engine |
| `module:experiment` | Experiments and logging |
| `priority:high` | Blocking milestone |
| `good-first-issue` | Starter-friendly tasks |

## Code Review Focus

- Graceful degradation off-target hardware
- Config-driven thresholds (no magic numbers in call sites)
- Log schema stability for experiment reproducibility
