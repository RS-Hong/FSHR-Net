# Frequency-Decoupled Structural Reformulation and Hierarchical Response Refinement for SAR Ship Detection

Core source files are protected. Prediction runs without an unlock key. The full unencrypted code will be released after the paper is accepted.

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

With **PyTorch 2.5.1+cu121**, FSHR achieved a reference latency of **19.19 ms/image (52.12 FPS)** for forward inference and postprocessing, using FP32, batch size 1 and single-channel 640 × 640 input. This excludes model loading, image preprocessing, data transfer and file I/O.
