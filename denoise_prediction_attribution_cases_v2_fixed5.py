#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction-related evidence script for DenoiseBlock in YOLOv8 SAR detection.

What this script does
---------------------
1) Compare baseline YOLOv8 vs YOLOv8 + DenoiseBlock on the same images/labels.
2) Select prediction-difference cases:
   - baseline-only false positives (FP suppressed by Denoise)
   - denoise-only true positives (TP recovered by Denoise)
3) For each selected case, export:
   - GT / baseline / denoise / denoise-bypass prediction panels
   - candidate score proxy bars under counterfactual bypasses
   - within the SAME denoise model, before/after DenoiseBlock support maps
     on the target ROI, to show what evidence the block suppresses or preserves.

This script is CONFIG-driven only. Edit CONFIG below and run directly.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Global plotting style for paper figures
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.unicode_minus"] = False

CONFIG = {
    # models / data
    "baseline_model": r"runs/detect/HRSID-YOLOv8-E300-B16-test/weights/best.pt",
    "denoise_model": r"runs/detect/HRSID-2denoise-300/weights/best.pt",
    "image_dir": r"images/HRSID/images",
    "label_dir": r"images/HRSID/labels",
    "outdir": r"绘图/1",
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    "max_images": 0,                 # 0 = all
    "max_cases_per_image": 1,

    # runtime
    "imgsz": 640,
    "conf": 0.3,
    "nms_iou": 0.7,
    "device": "0",
    "ultra_root": r"D:/code/ultralytics-SAR-V2",
    "list_modules_only": False,

    # matching / selection
    "target_class": None,            # e.g. 0, or None for all classes
    "match_iou": 0.5,
    "fp_suppress_iou": 0.3,
    "score_probe_iou": 0.1,          # when querying a candidate score near a target box
    "topk_fp_cases": 4,
    "topk_tp_cases": 4,
    "topk_support_channels": 6,
    "support_ring_scale": 1.5,

    # modules inside +Denoise model for same-model support analysis / bypass
    # set to your actual DenoiseBlock layers
    "denoise_layers": [
        {"name": "P4", "module": "model.12"},
        {"name": "P3", "module": "model.15"},
    ],
}


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_ultralytics_repo_to_path(repo_root: Optional[str]) -> None:
    if repo_root:
        repo_root = os.path.abspath(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)


def import_yolo_class():
    from ultralytics import YOLO
    return YOLO


def read_image_any(path: str | Path) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        x = img.astype(np.float32)
        if x.max() <= 1.0:
            x *= 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        if x.max() <= 1.0:
            x *= 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x
    return img


def to_gray_uint8_hwc1(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / (255.0 if gray.max() <= 255.0 else max(gray.max(), 1.0))
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return gray[..., None]


def percentile_normalize(x: np.ndarray, pmin: float = 1.0, pmax: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sym_percentile_normalize(x: np.ndarray, p: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    a = np.percentile(np.abs(x), p)
    if a < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / a, -1.0, 1.0).astype(np.float32)


def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm = np.clip(heatmap01, 0.0, 1.0)
    hm = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, hm_color, alpha, 0)


def resize_map_to_image(x: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def energy_map(feat: torch.Tensor) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat[0]
    x = feat.detach().float().cpu()
    return x.abs().mean(dim=0).numpy().astype(np.float32)


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: Optional[np.ndarray],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return canvas
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(canvas, labels[i], (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return canvas


def draw_gt_boxes(image_bgr: np.ndarray, gt_boxes: np.ndarray, gt_classes: np.ndarray) -> np.ndarray:
    canvas = image_bgr.copy()
    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(canvas, f"gt:{int(cls)}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def find_images(folder: Path, exts: Sequence[str]) -> List[Path]:
    exts_low = {e.lower() for e in exts}
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts_low]


# -----------------------------------------------------------------------------
# labels / boxes / matching
# -----------------------------------------------------------------------------
def xywhn_to_xyxy(row: Sequence[float], img_w: int, img_h: int) -> np.ndarray:
    cls, xc, yc, w, h = row[:5]
    bw = float(w) * img_w
    bh = float(h) * img_h
    cx = float(xc) * img_w
    cy = float(yc) * img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def read_yolo_label_file(path: Path, img_w: int, img_h: int, target_class: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    boxes: List[np.ndarray] = []
    classes: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        vals = [float(x) for x in s.split()[:5]]
        cls = int(vals[0])
        if target_class is not None and cls != target_class:
            continue
        boxes.append(xywhn_to_xyxy(vals, img_w, img_h))
        classes.append(cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(boxes, axis=0).astype(np.float32), np.asarray(classes, dtype=np.int32)


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    a = boxes1[:, None, :]
    b = boxes2[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-8
    return (inter / union).astype(np.float32)


@dataclass
class PredSummary:
    boxes: np.ndarray
    confs: np.ndarray
    classes: np.ndarray
    tp_flags: np.ndarray
    matched_gt: np.ndarray
    fp_flags: np.ndarray
    fn_gt_indices: List[int]


def extract_preds(result, target_class: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    if target_class is not None:
        keep = classes == int(target_class)
        boxes, confs, classes = boxes[keep], confs[keep], classes[keep]
    return boxes, confs, classes


def match_predictions_to_gt(
    boxes: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thr: float,
) -> PredSummary:
    n = len(boxes)
    tp = np.zeros((n,), dtype=bool)
    fp = np.zeros((n,), dtype=bool)
    matched_gt = np.full((n,), -1, dtype=np.int32)
    if len(boxes) == 0:
        return PredSummary(boxes, confs, classes, tp, matched_gt, np.zeros((0,), dtype=bool), list(range(len(gt_boxes))))
    order = np.argsort(-confs)
    used_gt = set()
    ious = box_iou_matrix(boxes, gt_boxes)
    for idx in order:
        best_j = -1
        best_iou = 0.0
        for j in range(len(gt_boxes)):
            if j in used_gt:
                continue
            if int(classes[idx]) != int(gt_classes[j]):
                continue
            iou = float(ious[idx, j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[idx] = True
            matched_gt[idx] = best_j
            used_gt.add(best_j)
        else:
            fp[idx] = True
    fn_gt = [j for j in range(len(gt_boxes)) if j not in used_gt]
    return PredSummary(boxes, confs, classes, tp, matched_gt, fp, fn_gt)


def best_overlap_same_class(box: np.ndarray, cls: int, boxes: np.ndarray, classes: np.ndarray) -> Tuple[float, int]:
    if len(boxes) == 0:
        return 0.0, -1
    keep = classes == int(cls)
    if not np.any(keep):
        return 0.0, -1
    cand = boxes[keep]
    idxs = np.where(keep)[0]
    ious = box_iou_matrix(box[None, :], cand)[0]
    k = int(np.argmax(ious))
    return float(ious[k]), int(idxs[k])


# -----------------------------------------------------------------------------
# YOLO helpers / hooks / patchers
# -----------------------------------------------------------------------------
def get_named_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    return list(model.model.named_modules())


def choose_module_by_name(modules: Sequence[Tuple[str, torch.nn.Module]], target_name: str) -> torch.nn.Module:
    for name, module in modules:
        if name == target_name:
            return module
    choices = ", ".join(name for name, _ in modules if name.startswith("model."))
    raise ValueError(f"Cannot find module {target_name}. Available names include: {choices}")


def get_named_modules_text(model, tag: str) -> str:
    lines = [f"==== {tag} named_modules ====\n"]
    idx = 0
    for name, module in getattr(model, "model", model).named_modules():
        if name.startswith("model.") and name.count(".") == 1:
            lines.append(f"[{idx:04d}] {name}    class={module.__class__.__name__}\n")
            idx += 1
    return "".join(lines)


def get_first_conv_in_channels(yolo_model) -> int:
    net = getattr(yolo_model, "model", None)
    if net is None:
        raise RuntimeError("Input model has no .model attribute")
    search_root = getattr(net, "model", net)
    for m in search_root.modules():
        if isinstance(m, torch.nn.Conv2d):
            return int(m.in_channels)
    raise RuntimeError("Cannot find first Conv2d")


def build_predict_source(img: np.ndarray, in_channels: int) -> np.ndarray:
    if in_channels == 1:
        return to_gray_uint8_hwc1(img)
    if in_channels == 3:
        return to_bgr_uint8(img)
    raise ValueError(f"Unsupported input channels: {in_channels}")


def run_predict(model, source_img_for_predict: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    results = model.predict(source=source_img_for_predict, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    if len(results) == 0:
        raise RuntimeError("No results returned by model.predict")
    return results[0]


class OutputHook:
    def __init__(self, module: torch.nn.Module):
        self.output: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            self.output = output.detach().cpu()
        elif isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
            self.output = output[0].detach().cpu()
        else:
            self.output = None

    def close(self):
        self.handle.remove()


class DenoiseIOCapture:
    """Capture input/output of a DenoiseBlock in the SAME trained denoise model."""
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.cache: Dict[str, torch.Tensor] = {}
        self._orig_forward = None

    def patch(self):
        m = self.module
        self._orig_forward = m.forward
        cache = self.cache

        def forward_hooked(x):
            b, c, _, _ = x.shape
            if m.fc is None:
                m.c = c
                m._init_layers(c, x.device)
            cache["input"] = x.detach().cpu()
            if m.shrink and m.tau is not None:
                stats = x.detach().abs().mean(dim=(2, 3))
                thr = (stats * m.tau.unsqueeze(0)).view(b, c, 1, 1)
                x_sh = m.soft_threshold(x, thr)
                cache["stats"] = stats.detach().cpu()
                cache["thr"] = thr.detach().cpu()
            else:
                x_sh = x
            cache["x_sh"] = x_sh.detach().cpu()
            avg = m.avg_pool(x_sh).view(b, c)
            maxv = m.max_pool(x_sh).view(b, c)
            y = torch.cat([avg, maxv], dim=1)
            gate = m.fc(y).view(b, c, 1, 1)
            out = x_sh * (1.0 + m.gamma * (gate - 1.0))
            cache["gate"] = gate.detach().cpu()
            cache["out"] = out.detach().cpu()
            cache["gamma"] = m.gamma.detach().cpu()
            return out

        m.forward = forward_hooked

    def restore(self):
        if self._orig_forward is not None:
            self.module.forward = self._orig_forward


class ModuleBypass:
    """Bypass a module with identity: y = x."""
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._orig_forward = None

    def patch(self):
        self._orig_forward = self.module.forward
        def forward_identity(x, *args, **kwargs):
            return x
        self.module.forward = forward_identity

    def restore(self):
        if self._orig_forward is not None:
            self.module.forward = self._orig_forward


# -----------------------------------------------------------------------------
# ROI support analysis within SAME denoise model
# -----------------------------------------------------------------------------
def box_to_feature_coords(box: np.ndarray, img_hw: Tuple[int, int], feat_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    ih, iw = img_hw
    fh, fw = feat_hw
    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    fx1 = int(np.floor(x1 / max(iw, 1) * fw))
    fy1 = int(np.floor(y1 / max(ih, 1) * fh))
    fx2 = int(np.ceil(x2 / max(iw, 1) * fw))
    fy2 = int(np.ceil(y2 / max(ih, 1) * fh))
    fx1 = max(0, min(fw - 1, fx1))
    fy1 = max(0, min(fh - 1, fy1))
    fx2 = max(fx1 + 1, min(fw, fx2))
    fy2 = max(fy1 + 1, min(fh, fy2))
    return fx1, fy1, fx2, fy2


def roi_ring_masks(feat_hw: Tuple[int, int], box: np.ndarray, img_hw: Tuple[int, int], scale: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    h, w = feat_hw
    fx1, fy1, fx2, fy2 = box_to_feature_coords(box, img_hw, feat_hw)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[fy1:fy2, fx1:fx2] = 1

    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0
    bw = (fx2 - fx1)
    bh = (fy2 - fy1)
    rx1 = max(int(round(cx - scale * bw / 2.0)), 0)
    ry1 = max(int(round(cy - scale * bh / 2.0)), 0)
    rx2 = min(int(round(cx + scale * bw / 2.0)), w)
    ry2 = min(int(round(cy + scale * bh / 2.0)), h)
    ring = np.zeros((h, w), dtype=np.uint8)
    ring[ry1:ry2, rx1:rx2] = 1
    ring = np.clip(ring - roi, 0, 1)
    return roi, ring


def per_channel_roi_ring_support(feat: torch.Tensor, box: np.ndarray, img_hw: Tuple[int, int], ring_scale: float = 1.5) -> Dict[str, np.ndarray]:
    """
    Local support proxy for one channel.

    support_c = mean_ROI(|x_c|) - mean_ring(|x_c|)

    Notes
    -----
    - We intentionally use absolute response magnitude because DenoiseBlock itself computes
      per-channel statistics from abs(x), and soft-thresholding acts on |x|.
    - Positive support means this channel is stronger inside the selected ROI than in the
      surrounding ring; negative support means the channel is background-dominant.
    - This is a local proxy, not a class-logit attribution.
    """
    if feat.ndim == 4:
        feat = feat[0]
    x = feat.detach().float().cpu().numpy().astype(np.float32)
    x_abs = np.abs(x)
    c, h, w = x_abs.shape
    roi, ring = roi_ring_masks((h, w), box, img_hw, scale=ring_scale)
    roi_mask = roi > 0
    ring_mask = ring > 0
    if roi_mask.sum() == 0 or ring_mask.sum() == 0:
        z = np.zeros((c,), dtype=np.float32)
        return {"roi": z, "ring": z, "support": z, "ratio": z, "roi_mask": roi, "ring_mask": ring}
    roi_mean = x_abs[:, roi_mask].mean(axis=1)
    ring_mean = x_abs[:, ring_mask].mean(axis=1)
    support = roi_mean - ring_mean
    ratio = roi_mean / (ring_mean + 1e-8)
    return {
        "roi": roi_mean.astype(np.float32),
        "ring": ring_mean.astype(np.float32),
        "support": support.astype(np.float32),
        "ratio": ratio.astype(np.float32),
        "roi_mask": roi,
        "ring_mask": ring,
    }


def weighted_channel_map(feat: torch.Tensor, weights: np.ndarray, top_idx: np.ndarray) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat[0]
    x = feat.abs().numpy().astype(np.float32)
    if len(top_idx) == 0:
        return np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)
    w = weights[top_idx].astype(np.float32)
    w = w / (w.sum() + 1e-8)
    return (x[top_idx] * w[:, None, None]).sum(axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# prediction panels / scoring
# -----------------------------------------------------------------------------
def make_pred_status_overlay(
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred: PredSummary,
    title_text: str = "",
) -> np.ndarray:
    canvas = img_bgr.copy()
    # FN GT in red
    for j in pred.fn_gt_indices:
        box = gt_boxes[j]
        cls = gt_classes[j]
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"FN gt:{int(cls)}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    # TP pred in green, FP pred in yellow
    for i, box in enumerate(pred.boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cls = int(pred.classes[i])
        conf = float(pred.confs[i])
        if pred.tp_flags[i]:
            color = (0, 255, 0)
            tag = f"TP {cls}:{conf:.2f}"
        else:
            color = (0, 255, 255)
            tag = f"FP {cls}:{conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, tag, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    if title_text:
        cv2.putText(canvas, title_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def find_candidate_score_near_box(result, target_box: np.ndarray, target_cls: int, iou_thr: float, target_class: Optional[int]) -> float:
    boxes, confs, classes = extract_preds(result, target_class)
    if len(boxes) == 0:
        return 0.0
    keep = classes == int(target_cls)
    if not np.any(keep):
        return 0.0
    cand_boxes = boxes[keep]
    cand_confs = confs[keep]
    ious = box_iou_matrix(target_box[None, :], cand_boxes)[0]
    valid = ious >= iou_thr
    if not np.any(valid):
        return 0.0
    return float(cand_confs[valid].max())


# -----------------------------------------------------------------------------
# case structures
# -----------------------------------------------------------------------------
@dataclass
class ImageCompare:
    image_path: str
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    baseline_pred: PredSummary
    denoise_pred: PredSummary
    baseline_result: Any
    denoise_result: Any


@dataclass
class CaseItem:
    case_type: str            # baseline_fp / denoise_tp
    image_path: str
    image_stem: str
    cls: int
    target_box: np.ndarray
    proxy_score: float
    desc: str


# -----------------------------------------------------------------------------
# list modules
# -----------------------------------------------------------------------------
def list_modules_for_both(cfg: dict) -> None:
    add_ultralytics_repo_to_path(cfg.get("ultra_root"))
    YOLO = import_yolo_class()
    base = YOLO(str(cfg["baseline_model"]))
    dn = YOLO(str(cfg["denoise_model"]))
    outdir = ensure_dir(cfg["outdir"])
    txt_b = get_named_modules_text(base, "baseline")
    txt_d = get_named_modules_text(dn, "+denoise")
    print(txt_b)
    print(txt_d)
    (outdir / "99_named_modules_baseline.txt").write_text(txt_b, encoding="utf-8")
    (outdir / "99_named_modules_denoise.txt").write_text(txt_d, encoding="utf-8")


# -----------------------------------------------------------------------------
# main comparison collection
# -----------------------------------------------------------------------------
def run_one_model(yolo_model, img_any: np.ndarray, cfg: dict):
    in_channels = get_first_conv_in_channels(yolo_model)
    source = build_predict_source(img_any, in_channels)
    return run_predict(
        yolo_model,
        source,
        imgsz=int(cfg["imgsz"]),
        conf=float(cfg["conf"]),
        iou=float(cfg["nms_iou"]),
        device=str(cfg["device"]),
    )


def build_image_compare(image_path: Path, base_model, dn_model, cfg: dict) -> ImageCompare:
    img_any = read_image_any(image_path)
    h, w = img_any.shape[:2]
    label_path = Path(cfg["label_dir"]) / (image_path.stem + ".txt")
    gt_boxes, gt_classes = read_yolo_label_file(label_path, w, h, cfg.get("target_class"))

    res_b = run_one_model(base_model, img_any, cfg)
    res_d = run_one_model(dn_model, img_any, cfg)

    boxes_b, confs_b, cls_b = extract_preds(res_b, cfg.get("target_class"))
    boxes_d, confs_d, cls_d = extract_preds(res_d, cfg.get("target_class"))

    pred_b = match_predictions_to_gt(boxes_b, confs_b, cls_b, gt_boxes, gt_classes, float(cfg["match_iou"]))
    pred_d = match_predictions_to_gt(boxes_d, confs_d, cls_d, gt_boxes, gt_classes, float(cfg["match_iou"]))

    return ImageCompare(
        image_path=str(image_path),
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        baseline_pred=pred_b,
        denoise_pred=pred_d,
        baseline_result=res_b,
        denoise_result=res_d,
    )


def collect_cases(comp: ImageCompare, cfg: dict) -> List[CaseItem]:
    cases: List[CaseItem] = []
    img_stem = Path(comp.image_path).stem

    # baseline-only FP that denoise no longer keeps nearby
    for i in np.where(comp.baseline_pred.fp_flags)[0].tolist():
        box = comp.baseline_pred.boxes[i]
        cls = int(comp.baseline_pred.classes[i])
        conf = float(comp.baseline_pred.confs[i])
        ov, idx = best_overlap_same_class(box, cls, comp.denoise_pred.boxes, comp.denoise_pred.classes)
        if idx < 0 or ov < float(cfg["fp_suppress_iou"]):
            cases.append(CaseItem(
                case_type="baseline_fp",
                image_path=comp.image_path,
                image_stem=img_stem,
                cls=cls,
                target_box=box.copy(),
                proxy_score=conf,
                desc=f"baseline FP suppressed; conf={conf:.3f}",
            ))

    # denoise-only TP that baseline missed
    used_gt_baseline = set(int(j) for j in comp.baseline_pred.matched_gt[comp.baseline_pred.tp_flags] if int(j) >= 0)
    for i in np.where(comp.denoise_pred.tp_flags)[0].tolist():
        gt_j = int(comp.denoise_pred.matched_gt[i])
        if gt_j not in used_gt_baseline:
            box = comp.denoise_pred.boxes[i]
            cls = int(comp.denoise_pred.classes[i])
            conf = float(comp.denoise_pred.confs[i])
            cases.append(CaseItem(
                case_type="denoise_tp",
                image_path=comp.image_path,
                image_stem=img_stem,
                cls=cls,
                target_box=box.copy(),
                proxy_score=conf,
                desc=f"denoise TP recovered; conf={conf:.3f}",
            ))
    return cases


def select_cases(all_cases: List[CaseItem], cfg: dict) -> List[CaseItem]:
    max_per_image = int(cfg.get("max_cases_per_image", 1))
    selected: List[CaseItem] = []
    for case_type, topk in [("baseline_fp", int(cfg["topk_fp_cases"])), ("denoise_tp", int(cfg["topk_tp_cases"]))]:
        subset = [c for c in all_cases if c.case_type == case_type]
        subset.sort(key=lambda c: -c.proxy_score)
        per_img: Dict[str, int] = {}
        count = 0
        for c in subset:
            if count >= topk:
                break
            k = c.image_stem
            if per_img.get(k, 0) >= max_per_image:
                continue
            selected.append(c)
            per_img[k] = per_img.get(k, 0) + 1
            count += 1
    return selected


# -----------------------------------------------------------------------------
# per-case rerun / analysis
# -----------------------------------------------------------------------------
def get_compare_from_path(comps: List[ImageCompare], path: str) -> ImageCompare:
    for c in comps:
        if c.image_path == path:
            return c
    raise KeyError(path)


def rerun_denoise_with_hooks_and_optional_bypass(
    model_path: str,
    ultra_root: Optional[str],
    img_any: np.ndarray,
    cfg: dict,
    capture_layers: Sequence[Dict[str, str]],
    bypass_layers: Optional[Sequence[str]] = None,
):
    add_ultralytics_repo_to_path(ultra_root)
    YOLO = import_yolo_class()
    model = YOLO(str(model_path))
    modules = get_named_modules(model)
    captures: Dict[str, DenoiseIOCapture] = {}
    bypassers: List[ModuleBypass] = []

    # patch captures on denoise layers
    for item in capture_layers:
        mod = choose_module_by_name(modules, item["module"])
        cap = DenoiseIOCapture(mod)
        cap.patch()
        captures[item["name"]] = cap

    # patch bypasses if requested
    if bypass_layers is not None:
        for mname in bypass_layers:
            mod = choose_module_by_name(modules, mname)
            bp = ModuleBypass(mod)
            bp.patch()
            bypassers.append(bp)

    try:
        result = run_one_model(model, img_any, cfg)
    finally:
        for cap in captures.values():
            cap.restore()
        for bp in bypassers:
            bp.restore()

    return result, {k: v.cache for k, v in captures.items()}


def summarize_support_change(cache: Dict[str, torch.Tensor], box: np.ndarray, img_hw: Tuple[int, int], case_type: str, topk: int, ring_scale: float = 1.5):
    inp = cache["input"]
    out = cache["out"]
    s_in = per_channel_roi_ring_support(inp, box, img_hw, ring_scale=ring_scale)
    s_out = per_channel_roi_ring_support(out, box, img_hw, ring_scale=ring_scale)

    # How top-k channels are chosen:
    # - For baseline_fp cases, we want channels whose local support for the false box DROPS most after Denoise.
    # - For denoise_tp cases, we want channels whose local support for the recovered true box INCREASES most after Denoise.
    if case_type == "baseline_fp":
        delta = s_out["support"] - s_in["support"]
        weights = np.maximum(-delta, 0.0)
        title = "channels with largest ROI-support drop (desired for FP suppression)"
    else:
        delta = s_out["support"] - s_in["support"]
        weights = np.maximum(delta, 0.0)
        title = "channels with largest ROI-support gain (desired for TP recovery)"
    top_idx = np.argsort(-weights)[:topk]
    in_map = weighted_channel_map(inp, weights + 1e-8, top_idx)
    out_map = weighted_channel_map(out, weights + 1e-8, top_idx)
    return {
        "support_in": s_in["support"],
        "support_out": s_out["support"],
        "roi_in": s_in["roi"],
        "roi_out": s_out["roi"],
        "ring_in": s_in["ring"],
        "ring_out": s_out["ring"],
        "ratio_in": s_in["ratio"],
        "ratio_out": s_out["ratio"],
        "delta": delta.astype(np.float32),
        "weights": weights.astype(np.float32),
        "top_idx": top_idx.astype(np.int32),
        "in_map": in_map.astype(np.float32),
        "out_map": out_map.astype(np.float32),
        "title": title,
    }

def save_support_figure(
    case: CaseItem,
    layer_name: str,
    cache: Dict[str, torch.Tensor],
    orig_bgr: np.ndarray,
    out_png: Path,
    topk: int,
    ring_scale: float = 1.5,
) -> None:
    """
    Visualize ROI-vs-ring local support change for one DenoiseBlock.

    support_c = mean_ROI(|x_c|) - mean_ring(|x_c|)

    - For denoise_tp cases, top-k channels are selected by largest support gain.
    - For baseline_fp cases, top-k channels are selected by largest support drop.
    """
    summ = summarize_support_change(
        cache=cache,
        box=case.target_box,
        img_hw=orig_bgr.shape[:2],
        case_type=case.case_type,
        topk=topk,
        ring_scale=ring_scale,
    )

    in_map = resize_map_to_image(percentile_normalize(summ["in_map"]), orig_bgr.shape[:2])
    out_map = resize_map_to_image(percentile_normalize(summ["out_map"]), orig_bgr.shape[:2])
    diff_map = resize_map_to_image(summ["out_map"] - summ["in_map"], orig_bgr.shape[:2])
    diff_vis = sym_percentile_normalize(diff_map)

    target_boxes = np.array([case.target_box], dtype=np.float32)
    target_labels = [f"target cls={case.cls}"]

    in_canvas = cv2.cvtColor(draw_boxes(overlay_heatmap_on_image(orig_bgr, in_map), target_boxes, target_labels, color=(255,255,255)), cv2.COLOR_BGR2RGB)
    out_canvas = cv2.cvtColor(draw_boxes(overlay_heatmap_on_image(orig_bgr, out_map), target_boxes, target_labels, color=(255,255,255)), cv2.COLOR_BGR2RGB)

    idx = summ["top_idx"]
    support_in = summ["support_in"][idx] if len(idx) else np.array([], dtype=np.float32)
    support_out = summ["support_out"][idx] if len(idx) else np.array([], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), constrained_layout=True)

    axes[0, 0].imshow(in_canvas)
    axes[0, 0].set_title(f"{layer_name} input support aggregate")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(out_canvas)
    axes[0, 1].set_title(f"{layer_name} output support aggregate")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(diff_vis, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_title(f"{layer_name} support change (out - in)")
    axes[1, 0].axis("off")
    cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Signed change")

    x = np.arange(len(idx))
    width = 0.35
    axes[1, 1].bar(x - width / 2, support_in, width=width, label="input")
    axes[1, 1].bar(x + width / 2, support_out, width=width, label="output")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(int(i)) for i in idx])
    axes[1, 1].set_xlabel("channel index")
    axes[1, 1].set_ylabel("ROI - ring support")
    if case.case_type == "baseline_fp":
        axes[1, 1].set_title(f"{layer_name} top-{len(idx)} support channels\n{summ['title']}")
    else:
        axes[1, 1].set_title(f"{layer_name} top-{len(idx)} support channels\n{summ['title']}")
    axes[1, 1].grid(alpha=0.25, axis="y")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(f"{case.case_type} | {case.image_stem} | cls={case.cls}", fontsize=14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# figure builders
# -----------------------------------------------------------------------------
def save_triptych_case(comp: ImageCompare, img_bgr: np.ndarray, out_png: Path) -> None:
    gt_img = draw_gt_boxes(img_bgr, comp.gt_boxes, comp.gt_classes)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    base_img = make_pred_status_overlay(
        img_bgr, comp.gt_boxes, comp.gt_classes, comp.baseline_pred,
        title_text=f"baseline | TP={int(comp.baseline_pred.tp_flags.sum())} FP={int(comp.baseline_pred.fp_flags.sum())} FN={len(comp.baseline_pred.fn_gt_indices)}"
    )
    den_img = make_pred_status_overlay(
        img_bgr, comp.gt_boxes, comp.gt_classes, comp.denoise_pred,
        title_text=f"+Denoise | TP={int(comp.denoise_pred.tp_flags.sum())} FP={int(comp.denoise_pred.fp_flags.sum())} FN={len(comp.denoise_pred.fn_gt_indices)}"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    for ax, arr, title in zip(axes, [gt_img, base_img, den_img], ["GT", "baseline", "+Denoise"]):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_image_level_score_groups(comp: ImageCompare, cfg: dict) -> Dict[str, List[Tuple[str, int, np.ndarray]]]:
    """
    Return per-image score groups:
    - fn_cases: GTs recovered only by +Denoise (baseline FN -> denoise TP)
    - fp_cases: baseline-only false positives suppressed by +Denoise
    Each item is (label, cls, target_box).
    """
    out = {"fn_cases": [], "fp_cases": []}
    # FN recovery cases: iterate denoise TPs whose matched GT was not matched by baseline
    used_gt_baseline = set(int(j) for j in comp.baseline_pred.matched_gt[comp.baseline_pred.tp_flags] if int(j) >= 0)
    fn_id = 1
    for i in np.where(comp.denoise_pred.tp_flags)[0].tolist():
        gt_j = int(comp.denoise_pred.matched_gt[i])
        if gt_j not in used_gt_baseline:
            out["fn_cases"].append((f"FN-{fn_id}", int(comp.denoise_pred.classes[i]), comp.denoise_pred.boxes[i].copy()))
            fn_id += 1
    # FP suppression cases: iterate baseline FPs that denoise no longer keeps nearby
    fp_id = 1
    for i in np.where(comp.baseline_pred.fp_flags)[0].tolist():
        box = comp.baseline_pred.boxes[i]
        cls = int(comp.baseline_pred.classes[i])
        ov, idx = best_overlap_same_class(box, cls, comp.denoise_pred.boxes, comp.denoise_pred.classes)
        if idx < 0 or ov < float(cfg["fp_suppress_iou"]):
            out["fp_cases"].append((f"FP-{fp_id}", cls, box.copy()))
            fp_id += 1
    return out

def draw_case_labels(canvas_bgr: np.ndarray, items: Dict[str, List[Tuple[str, int, np.ndarray]]], color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    canvas = canvas_bgr.copy()
    for group_items in items.values():
        for label, cls, box in group_items:
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            cv2.putText(canvas, label, (x1, min(canvas.shape[0] - 5, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def with_case_labels(rgb_or_bgr: np.ndarray, items: Dict[str, List[Tuple[str, int, np.ndarray]]], is_rgb: bool = True) -> np.ndarray:
    if is_rgb:
        bgr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_RGB2BGR)
        out = draw_case_labels(bgr, items)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return draw_case_labels(rgb_or_bgr, items)


def save_case_prediction_and_scores(
    case: CaseItem,
    comp: ImageCompare,
    img_bgr: np.ndarray,
    cfg: dict,
    out_png: Path,
    scores_png: Path,
) -> None:
    img_any = read_image_any(case.image_path)

    # normal denoise run with caches
    dn_normal_result, caches_normal = rerun_denoise_with_hooks_and_optional_bypass(
        cfg["denoise_model"], cfg.get("ultra_root"), img_any, cfg, cfg["denoise_layers"], bypass_layers=None
    )

    # bypass each layer and all layers
    bypass_results: Dict[str, Any] = {}
    bypass_specs = []
    for item in cfg["denoise_layers"]:
        bypass_specs.append((f"bypass_{item['name']}", [item["module"]]))
    bypass_specs.append(("bypass_all", [x["module"] for x in cfg["denoise_layers"]]))

    for key, mods in bypass_specs:
        res, _ = rerun_denoise_with_hooks_and_optional_bypass(
            cfg["denoise_model"], cfg.get("ultra_root"), img_any, cfg, cfg["denoise_layers"], bypass_layers=mods
        )
        bypass_results[key] = res

    # build prediction panels and annotate per-image FN/FP case ids
    groups = build_image_level_score_groups(comp, cfg)
    gt_img = draw_gt_boxes(img_bgr, comp.gt_boxes, comp.gt_classes)
    gt_img = cv2.cvtColor(draw_case_labels(gt_img, groups), cv2.COLOR_BGR2RGB)

    base_panel = make_pred_status_overlay(img_bgr, comp.gt_boxes, comp.gt_classes, comp.baseline_pred)
    base_panel = with_case_labels(base_panel, groups, is_rgb=True)
    den_pred = match_predictions_to_gt(*extract_preds(comp.denoise_result, cfg.get("target_class")), comp.gt_boxes, comp.gt_classes, float(cfg["match_iou"]))
    den_panel = make_pred_status_overlay(img_bgr, comp.gt_boxes, comp.gt_classes, den_pred)
    den_panel = with_case_labels(den_panel, groups, is_rgb=True)

    panels = [
        ("GT", gt_img),
        ("baseline", base_panel),
        ("+Denoise", den_panel),
    ]

    for key, res in bypass_results.items():
        boxes, confs, classes = extract_preds(res, cfg.get("target_class"))
        pred = match_predictions_to_gt(boxes, confs, classes, comp.gt_boxes, comp.gt_classes, float(cfg["match_iou"]))
        panel = make_pred_status_overlay(img_bgr, comp.gt_boxes, comp.gt_classes, pred)
        panel = with_case_labels(panel, groups, is_rgb=True)
        panels.append((key, panel))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 5.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{case.case_type} | {case.image_stem} | cls={case.cls} | {case.desc}")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # split score bars: FN recovery and FP suppression, potentially multiple per image
    score_series_order = [
        ("baseline", comp.baseline_result),
        ("+Denoise", dn_normal_result),
        *[(key, res) for key, res in bypass_results.items()],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.8), constrained_layout=True)
    for ax, group_key, title in zip(axes, ["fn_cases", "fp_cases"], ["FN recovery score proxy", "FP suppression score proxy"]):
        items = groups[group_key]
        if len(items) == 0:
            ax.text(0.5, 0.5, "No cases in this image", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        width = 0.12
        x = np.arange(len(items), dtype=np.float32)
        for j, (model_name, result_obj) in enumerate(score_series_order):
            vals = []
            has_box = []
            for label, cls, box in items:
                v = find_candidate_score_near_box(result_obj, box, cls, float(cfg["score_probe_iou"]), cfg.get("target_class"))
                vals.append(v)
                has_box.append(v > 0)
            xpos = x + (j - (len(score_series_order)-1)/2.0) * width
            bars = ax.bar(xpos, vals, width=width, label=model_name)
            for b, ok, v in zip(bars, has_box, vals):
                if not ok:
                    b.set_facecolor((0.75, 0.75, 0.75, 0.9))
                    b.set_edgecolor('black')
                    b.set_hatch('//')
                    ax.text(b.get_x() + b.get_width()/2, 0.01, 'none', ha='center', va='bottom', rotation=90, fontsize=8)
                else:
                    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in items])
        ax.set_ylabel("candidate score near selected box")
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(f"score proxy split by case type | {case.image_stem}")
    fig.savefig(scores_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return caches_normal



# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def validate_config(cfg: dict) -> None:
    required = ["baseline_model", "denoise_model", "image_dir", "label_dir", "outdir", "imgsz", "conf", "nms_iou", "device"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"CONFIG missing: {k}")


def main() -> None:
    cfg = CONFIG
    validate_config(cfg)
    outdir = ensure_dir(cfg["outdir"])

    if cfg.get("list_modules_only", False):
        list_modules_for_both(cfg)
        return

    add_ultralytics_repo_to_path(cfg.get("ultra_root"))
    YOLO = import_yolo_class()
    baseline_model = YOLO(str(cfg["baseline_model"]))
    denoise_model = YOLO(str(cfg["denoise_model"]))

    # save module lists too
    (outdir / "99_named_modules_baseline.txt").write_text(get_named_modules_text(baseline_model, "baseline"), encoding="utf-8")
    (outdir / "99_named_modules_denoise.txt").write_text(get_named_modules_text(denoise_model, "+denoise"), encoding="utf-8")

    image_paths = find_images(Path(cfg["image_dir"]), cfg["image_exts"])
    if int(cfg.get("max_images", 0)) > 0:
        image_paths = image_paths[: int(cfg["max_images"])]

    trip_dir = ensure_dir(outdir / "triptychs")
    case_dir = ensure_dir(outdir / "case_evidence")

    comps: List[ImageCompare] = []
    all_cases: List[CaseItem] = []

    # collect comparisons
    rows_csv: List[Dict[str, Any]] = []
    for i, img_path in enumerate(image_paths, 1):
        comp = build_image_compare(img_path, baseline_model, denoise_model, cfg)
        comps.append(comp)
        all_cases.extend(collect_cases(comp, cfg))
        rows_csv.append({
            "image": img_path.name,
            "baseline_tp": int(comp.baseline_pred.tp_flags.sum()),
            "baseline_fp": int(comp.baseline_pred.fp_flags.sum()),
            "baseline_fn": len(comp.baseline_pred.fn_gt_indices),
            "denoise_tp": int(comp.denoise_pred.tp_flags.sum()),
            "denoise_fp": int(comp.denoise_pred.fp_flags.sum()),
            "denoise_fn": len(comp.denoise_pred.fn_gt_indices),
        })
        # always export triptych
        save_triptych_case(comp, to_bgr_uint8(read_image_any(img_path)), trip_dir / f"{img_path.stem}_triptych.png")
        print(f"[{i}/{len(image_paths)}] {img_path.name}")

    # summary csv
    with (outdir / "01_per_image_compare.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else ["image"])
        writer.writeheader()
        for r in rows_csv:
            writer.writerow(r)

    # select representative cases
    selected_cases = select_cases(all_cases, cfg)
    with (outdir / "02_selected_cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["case_type", "image_path", "image_stem", "cls", "proxy_score", "desc"])
        writer.writeheader()
        for c in selected_cases:
            writer.writerow({
                "case_type": c.case_type,
                "image_path": c.image_path,
                "image_stem": c.image_stem,
                "cls": c.cls,
                "proxy_score": c.proxy_score,
                "desc": c.desc,
            })

    # build evidence for selected cases
    for idx, case in enumerate(selected_cases, 1):
        comp = get_compare_from_path(comps, case.image_path)
        img_bgr = to_bgr_uint8(read_image_any(case.image_path))

        case_root = ensure_dir(case_dir / f"{idx:02d}_{case.case_type}_{case.image_stem}_cls{case.cls}")
        caches_normal = save_case_prediction_and_scores(
            case,
            comp,
            img_bgr,
            cfg,
            case_root / "01_predictions.png",
            case_root / "02_score_proxy.png",
        )
        for item in cfg["denoise_layers"]:
            lname = item["name"]
            if lname in caches_normal:
                save_support_figure(
                    case,
                    lname,
                    caches_normal[lname],
                    img_bgr,
                    case_root / f"03_support_{lname}.png",
                    int(cfg["topk_support_channels"]),
                )

        with (case_root / "00_case_info.txt").open("w", encoding="utf-8") as f:
            f.write(f"case_type: {case.case_type}\n")
            f.write(f"image: {case.image_path}\n")
            f.write(f"class: {case.cls}\n")
            f.write(f"target_box: {case.target_box.tolist()}\n")
            f.write(f"proxy_score: {case.proxy_score}\n")
            f.write(f"desc: {case.desc}\n")

    # global summary text
    base_tp = sum(int(c.baseline_pred.tp_flags.sum()) for c in comps)
    base_fp = sum(int(c.baseline_pred.fp_flags.sum()) for c in comps)
    base_fn = sum(len(c.baseline_pred.fn_gt_indices) for c in comps)
    den_tp = sum(int(c.denoise_pred.tp_flags.sum()) for c in comps)
    den_fp = sum(int(c.denoise_pred.fp_flags.sum()) for c in comps)
    den_fn = sum(len(c.denoise_pred.fn_gt_indices) for c in comps)
    with (outdir / "00_dataset_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"images: {len(comps)}\n")
        f.write(f"baseline TP/FP/FN: {base_tp}/{base_fp}/{base_fn}\n")
        f.write(f"denoise  TP/FP/FN: {den_tp}/{den_fp}/{den_fn}\n")
        f.write(f"selected cases: {len(selected_cases)}\n")

    print("=" * 80)
    print("Saved:")
    for name in [
        "00_dataset_summary.txt",
        "01_per_image_compare.csv",
        "02_selected_cases.csv",
        "99_named_modules_baseline.txt",
        "99_named_modules_denoise.txt",
    ]:
        print(outdir / name)
    print(trip_dir)
    print(case_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()