# Frequency-Decoupled Structural Reformulation and Hierarchical Response Refinement for SAR Ship Detection

Core modules are protected with git-crypt. Authorized users must unlock this checkout before importing or running it. The full unencrypted code will be released after the paper is accepted.

## Prediction

The initialized FSHR model now enables the M5 deployment path automatically through `YOLO(...).predict(...)`, including the existing `predict.py`. It uses a separate prediction model; training, validation, weights, confidence thresholds, NMS settings and visualization colors are unchanged. Other model architectures use their existing prediction path. Set `FSHR_FAST_PREDICT=0` **before creating the predictor** to use native FSHR prediction for comparison. The optimized path targets eager PyTorch inference; `compile=True` uses the existing path.

```python
from ultralytics import YOLO

model = YOLO("path/to/FSHR.pt")
results = model.predict(source="path/to/images", imgsz=640, batch=1,
                        device=0, half=False, rect=False, stream=True)
for result in results:
    print(result.speed)  # preprocessing, forward inference, postprocessing: separate timings
```

### Reproducing the timing

The reference M5 result was **19.19 ms/image (52.12 FPS)** for forward computation plus native postprocessing on an **RTX 4070 Ti**, PyTorch **2.5.1+cu121**, FP32, TF32 off, batch 1, single-channel 640 × 640 input, 4 CPU threads, eager mode and `cudnn.benchmark=False`. It used 32 real images resident on the GPU, 30 warmup passes per round and 3 rounds of 100 samples, with `conf=0.001`, `iou=0.6`, `max_det=300`. Forward-only time was **17.37 ms**. The reference checkpoint SHA256 was `24834a4e76bf444b0df738f8c38a33faf4a5164375652a59c2c5404d901c0e93`.

This is a measured reference, not a fixed latency or a guarantee on other GPUs/checkpoints. It excludes model loading, disk access, preprocessing, CPU-to-GPU transfer, drawing and saving; timing the entire `predict()` call measures additional work. Floating-point execution order changes slightly; bitwise-identical predictions are not promised.

Use your real checkpoint and images to record actual timings and the hardware/configuration:

```bash
python benchmark_fshr.py --weights path/to/FSHR.pt --source path/to/images --device 0 --output fshr_benchmark_fast.json
python benchmark_fshr.py --weights path/to/FSHR.pt --source path/to/images --device 0 --native --output fshr_benchmark_native.json
```

### Authorized checkout

The protection key was rotated with this prediction update. Obtain the new key privately from the repository owner, then run `git-crypt unlock /private/path/to/FSHR-Net.git-crypt.key`. Never commit or distribute that key with the public repository. A locked checkout contains encrypted Python modules and cannot run prediction.
