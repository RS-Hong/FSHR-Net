"""Reproduce warmed B1 FP32 prediction timing on real, GPU-resident images.

Run after unlocking the repository, with the same checkpoint as prediction.
No images, labels, weights or timings are downloaded or substituted.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Image directory or a single image")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--native", action="store_true", help="Disable FSHR M5 for comparison")
    parser.add_argument("--output", type=Path, default=Path("fshr_benchmark.json"))
    args = parser.parse_args()
    if min(args.imgsz, args.images, args.warmup, args.iterations, args.rounds) < 1:
        parser.error("Sizes and timing counts must be positive.")
    if not args.weights.is_file() or not args.source.exists():
        parser.error("Both weights and source must exist locally.")
    os.environ["FSHR_FAST_PREDICT"] = "0" if args.native else "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect import DetectionPredictor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = YOLO(str(args.weights))
    predictor = DetectionPredictor(overrides={
        "task": "detect", "imgsz": args.imgsz, "device": args.device,
        "half": False, "batch": 1, "rect": False, "conf": 0.001,
        "iou": 0.6, "max_det": 300, "save": False, "verbose": False,
    })
    predictor.setup_model(model.model, verbose=False)
    if predictor.device.type != "cuda":
        parser.error("This CUDA timing protocol requires a CUDA GPU.")
    report = getattr(predictor.model, "fshr_inference", None)
    if not args.native and not (report and report.get("enabled")):
        raise RuntimeError(f"The supplied checkpoint did not enable FSHR M5: {report}")
    deployment = predictor.model.model
    channels = predictor.model.ch
    if channels not in (1, 3):
        raise ValueError(f"Unsupported image channel count: {channels}")
    validator = model._smart_load("validator")(args={
        "task": "detect", "conf": 0.001, "iou": 0.6, "max_det": 300,
        "half": False, "plots": False, "save_json": False, "save_txt": False,
    })
    validator.nc = len(deployment.names)
    validator.end2end = getattr(deployment, "end2end", False)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in extensions) if args.source.is_dir() else [args.source]
    paths = paths[:args.images]
    if not paths:
        raise ValueError("No source images found.")
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    inputs, manifest = [], []
    for path in paths:
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        if channels == 1:
            image = image[..., None]
        image = letterbox(image=image)
        if channels == 3:
            image = image[..., ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
        inputs.append(tensor.to(predictor.device, dtype=torch.float32) / 255.0)
        manifest.append({"name": path.name, "sha256": sha256(path)})

    measurements = []
    with torch.inference_mode():
        for round_index in range(args.rounds):
            for i in range(args.warmup):
                validator.postprocess(deployment(inputs[i % len(inputs)]))
            torch.cuda.synchronize(predictor.device)
            forward, full = [], []
            for i in range(args.iterations):
                x = inputs[i % len(inputs)]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                prediction = deployment(x)
                end.record()
                end.synchronize()
                forward.append(start.elapsed_time(end))
                # A separate pass measures synchronized wall time, not a subtraction of timers.
                torch.cuda.synchronize(predictor.device)
                t0 = time.perf_counter()
                validator.postprocess(deployment(x))
                torch.cuda.synchronize(predictor.device)
                full.append((time.perf_counter() - t0) * 1000)
            measurements.append({"round": round_index + 1, "forward_ms": forward, "forward_postprocess_ms": full})
            print(f"Round {round_index + 1}: forward={statistics.mean(forward):.3f} ms; "
                  f"forward+postprocess={statistics.mean(full):.3f} ms", flush=True)

    mean_forward = [statistics.mean(r["forward_ms"]) for r in measurements]
    mean_full = [statistics.mean(r["forward_postprocess_ms"]) for r in measurements]
    result = {
        "gpu": torch.cuda.get_device_name(predictor.device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(), "platform": platform.platform(),
        "weights_sha256": sha256(args.weights), "adapter": report, "native": args.native,
        "protocol": {"batch": 1, "channels": channels, "imgsz": args.imgsz, "precision": "FP32",
                     "tf32": False, "cudnn_benchmark": False, "cpu_threads": 4,
                     "conf": 0.001, "iou": 0.6, "max_det": 300,
                     "warmup_per_round": args.warmup, "iterations_per_round": args.iterations,
                     "excluded": ["model setup", "disk IO", "preprocessing", "host-to-device transfer", "drawing", "saving"]},
        "images": manifest, "measurements": measurements,
        "forward_mean_ms": statistics.mean(mean_forward),
        "forward_postprocess_mean_ms": statistics.mean(mean_full),
        "forward_postprocess_round_sd_ms": statistics.stdev(mean_full) if len(mean_full) > 1 else None,
        "fps_forward_postprocess": 1000 / statistics.mean(mean_full),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Measured {result['forward_postprocess_mean_ms']:.3f} ms, "
          f"{result['fps_forward_postprocess']:.2f} FPS on {result['gpu']}. Report: {args.output}")


if __name__ == "__main__":
    main()
