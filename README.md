# Frequency-Decoupled Structural Reformulation and Hierarchical Response Refinement for SAR Ship Detection

Core modules are protected with git-crypt. Before running the code, unlock the repository using your authorized key: `git-crypt unlock /path/to/FSHR-Net.git-crypt.key`. The full unencrypted code will be released after the paper is accepted.

## Prediction

Set the model, data and output paths in `predict.py`, then run:

```bash
python predict.py
```

The data directory should contain `images/` and `labels/` for the existing prediction visualization.

## Inference speed

```bash
python benchmark_fshr.py --weights path/to/FSHR.pt --source path/to/images
```

With **PyTorch 2.5.1+cu121**, FSHR achieved a reference latency of **19.19 ms/image (52.12 FPS)** for forward inference and postprocessing, using FP32, batch size 1 and single-channel 640 × 640 input. This excludes model loading, image preprocessing, data transfer and file I/O. The script measures the actual latency in your environment and saves it to `fshr_benchmark.json`.
