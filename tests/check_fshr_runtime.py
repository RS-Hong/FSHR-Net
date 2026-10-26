"""Check a downloaded, still-encrypted checkout without a git-crypt key."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["FSHR_FAST_PREDICT"] = "1"

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = ROOT / "ultralytics/nn/modules"
    protected = sorted(source.glob("*.py"))
    assert len(protected) == 12
    for path in protected:
        assert path.read_bytes().startswith(b"\x00GITCRYPT\x00"), path
        name = "ultralytics.nn.modules" + ("." + path.stem if path.stem != "__init__" else "")
        imported = importlib.import_module(name)
        assert "_fshr_runtime" in imported.__file__, imported.__file__

    manifest = json.loads((ROOT / "ultralytics/_fshr_runtime/manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / "ultralytics/_fshr_runtime" / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    config = YAML.load(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml")
    config.update(nc=1, channels=1, scale="s")
    native = DetectionModel(config, ch=1, nc=1, verbose=False).float().eval()
    x = torch.rand(1, 1, 640, 640)
    with torch.inference_mode():
        reference = native(x)[0]
        assert torch.isfinite(reference).all()

    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "FSHR.pt"
        torch.save({"model": native, "train_args": {"task": "detect", "imgsz": 640}}, checkpoint)
        wrapper = YOLO(str(checkpoint))
        predictor = DetectionPredictor(overrides={"device": "cpu", "imgsz": 640, "half": False,
                                                   "rect": False, "save": False, "verbose": False})
        predictor.setup_model(wrapper.model, verbose=False)
        assert predictor.model.fshr_inference["enabled"]
        with torch.inference_mode():
            actual = predictor.model(x)[0]
        torch.testing.assert_close(actual, reference, atol=0.003, rtol=1e-4)
        image = (x[0, 0].numpy() * 255).astype(np.uint8)[..., None]
        result = wrapper.predict(image, device="cpu", imgsz=640, rect=False, save=False, verbose=False)
        assert len(result) == 1
        assert wrapper.predictor.model.fshr_inference["enabled"]
    print(json.dumps({"result": "passed", "encrypted_modules": len(protected), "python": sys.version.split()[0],
                      "torch": torch.__version__, "checkpoint_reload": True, "prediction": True,
                      "accelerated_vs_native_max_abs": (actual - reference).abs().max().item()}))


if __name__ == "__main__":
    main()
