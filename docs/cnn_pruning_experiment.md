# CNN Structured Pruning Experiment

## Purpose

This experiment checks whether RT-DETRv2 R18-lite CNN backbone pruning can be
exported to ONNX and reused by the current Raspberry Pi ONNX Runtime pipeline.
The pruning is structural channel pruning, so channels are removed from the
graph instead of only setting weights to zero.

## Implementation

The experiment script is:

```bash
scripts/run_cnn_pruning_experiment.py
```

It builds the upstream RT-DETRv2 R18-lite model from:

```text
third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml
```

Then it:

1. loads the official checkpoint when provided;
2. applies dependency-aware CNN channel pruning with `torch-pruning`;
3. exports a single-file ONNX model;
4. checks the ONNX model with ONNX Runtime;
5. records PyTorch and ONNX Runtime latency.

The pruning methods tested were:

| Method | Criterion |
|---|---|
| L1 | prune filters with smallest sum of absolute weights |
| L2 | prune filters with smallest Euclidean weight norm |
| FPGM | prune geometrically redundant filters |
| BN-scale | prune channels with small BatchNorm gamma scale |
| Random | random structured pruning baseline |

For Raspberry Pi deployment, `configs/raspberry_pi4_pruned_l1_r20.yaml` points
to the generated L1 20% ONNX files for 320, 480, and 640 input sizes.

## Commands

Download RT-DETR and prepare dependencies:

```bash
bash scripts/clone_rtdetr.sh
.venv/bin/python -m pip install torch torchvision onnx onnxruntime torch-pruning \
  numpy opencv-python PyYAML pandas matplotlib faster-coco-eval scipy tensorboard onnxscript
```

Run the 320 experiment:

```bash
.venv/bin/python scripts/run_cnn_pruning_experiment.py \
  --checkpoint third_party/checkpoints/rtdetrv2_r18vd_sp1_120e_coco.pth \
  --input-size 320 \
  --out-dir experiments/results/cnn_pruning_checkpoint \
  --warmup 2 \
  --runs 8 \
  --plan baseline l1:0.10 l1:0.20 l2:0.20 fpgm:0.20 bn_scale:0.20 random:0.20
```

Run the 480 experiment:

```bash
.venv/bin/python scripts/run_cnn_pruning_experiment.py \
  --checkpoint third_party/checkpoints/rtdetrv2_r18vd_sp1_120e_coco.pth \
  --input-size 480 \
  --out-dir experiments/results/cnn_pruning_checkpoint_480 \
  --warmup 1 \
  --runs 5 \
  --plan baseline l1:0.20 fpgm:0.20 bn_scale:0.20
```

Run the 640 experiment:

```bash
.venv/bin/python scripts/run_cnn_pruning_experiment.py \
  --checkpoint third_party/checkpoints/rtdetrv2_r18vd_sp1_120e_coco.pth \
  --input-size 640 \
  --out-dir experiments/results/cnn_pruning_checkpoint_640 \
  --warmup 1 \
  --runs 3 \
  --plan baseline l1:0.20
```

## Results

### 320 Input

| Method | Ratio | Params | Conv Params | PyTorch ms | ONNX ms |
|---|---:|---:|---:|---:|---:|
| baseline | 0% | 20.02M | 15.55M | 33.824 | 66.426 |
| L1 | 10% | 17.58M | 13.11M | 29.379 | 36.332 |
| L1 | 20% | 15.76M | 11.29M | 28.684 | 30.330 |
| L2 | 20% | 15.76M | 11.29M | 28.138 | 30.405 |
| FPGM | 20% | 15.76M | 11.29M | 28.478 | 30.559 |
| BN-scale | 20% | 15.76M | 11.29M | 29.247 | 30.189 |
| Random | 20% | 15.76M | 11.29M | 29.070 | 31.468 |

### 480 Input

| Method | Ratio | Params | Conv Params | PyTorch ms | ONNX ms |
|---|---:|---:|---:|---:|---:|
| baseline | 0% | 20.02M | 15.55M | 51.474 | 85.518 |
| L1 | 20% | 15.76M | 11.29M | 45.165 | 47.575 |
| FPGM | 20% | 15.76M | 11.29M | 52.504 | 47.247 |
| BN-scale | 20% | 15.76M | 11.29M | 51.828 | 52.903 |

### 640 Input

| Method | Ratio | Params | Conv Params | PyTorch ms | ONNX ms |
|---|---:|---:|---:|---:|---:|
| baseline | 0% | 20.02M | 15.55M | 94.784 | 112.039 |
| L1 | 20% | 15.76M | 11.29M | 76.717 | 69.228 |

## Conclusion

Structured CNN channel pruning is compatible with the current ONNX Runtime
deployment path. L1 20% is the recommended first deployment candidate because it
is simple to explain, works for all three input sizes, and gives consistent ONNX
Runtime speedups:

| Input | Baseline ONNX | L1 20% ONNX | Speedup |
|---:|---:|---:|---:|
| 320 | 66.426 ms | 30.330 ms | 2.19x |
| 480 | 85.518 ms | 47.575 ms | 1.80x |
| 640 | 112.039 ms | 69.228 ms | 1.62x |

The next step should be quality evaluation on validation video or labeled data,
because latency alone does not measure detection accuracy after pruning.
