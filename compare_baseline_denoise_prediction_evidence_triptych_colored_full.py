#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline YOLOv8 and +DenoiseBlock models on the same dataset and build
prediction-related evidence figures for DenoiseBlock.

This script is NOT command-line driven. Edit CONFIG below and run directly.

What it tries to show
---------------------
1) Which false positives (FPs) produced by the baseline are suppressed by the
   +DenoiseBlock model.
2) For those suppressed FPs, whether ROI-specific channel activations on a chosen
   fusion layer are reduced after DenoiseBlock.
3) Whether true positives are largely preserved while baseline-only FPs are reduced.

Important note
--------------
The per-channel/ROI analysis here is still a *proxy* for box-level contribution.
It is much more prediction-related than mean-abs feature maps over the whole image,
because it focuses on:
- actual baseline false-positive boxes, and
- the feature ROI projected from those boxes.
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


CONFIG = {
    # ----------------------------
    # models / data
    # ----------------------------
    "baseline_model": r"runs/detect/HRSID-YOLOv8-E300-B16-test/weights/best.pt",
    "denoise_model": r"runs/detect/HRSID-2denoise-300/weights/best.pt",
    "image_dir": r"images/HRSID/images",
    "label_dir": r"images/HRSID/labels",
    "outdir": r"绘图/1",
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    "max_images": 0,  # 0 means all

    # ----------------------------
    # runtime
    # ----------------------------
    "imgsz": 640,
    "conf": 0.25,
    "device": "0",
    "ultra_root": r"D:/code/ultralytics-SAR-V2",
    "list_modules_only": False,

    # ----------------------------
    # matching / comparison
    # ----------------------------
    "nms_iou": 0.5,            # NMS IoU used by model.predict()
    "match_iou": 0.5,          # TP/FP matching IoU against GT
    "fp_suppress_iou": 0.3,    # baseline FP is considered suppressed if denoise has no same-class box above this IoU
    "topk_fp_cases": 6,
    "topk_tp_cases": 4,
    "topk_channels": 6,

    # ----------------------------
    # always-export visual comparison panels
    # selection: "first", "fp_gap", "tp_gap", "fp_then_tp"
    # ----------------------------
    "export_triptychs": True,
    "triptych_selection": "fp_then_tp",
    "topk_triptychs": 12,
    "target_class": None,      # e.g. 0 ; None means all classes

    # ----------------------------
    # semantic layer pairs to compare
    # baseline and denoise layers should be semantically aligned
    # Standard YOLOv8 head:
    #   model.11 -> P4 Concat
    #   model.14 -> P3 Concat
    # Modified YAML:
    #   model.12 -> P4 DenoiseBlock
    #   model.16 -> P3 DenoiseBlock
    # ----------------------------
    "compare_pairs": [
        {
            "name": "P4",
            "baseline_layer": "model.11",
            "denoise_layer": "model.12",
        },
        {
            "name": "P3",
            "baseline_layer": "model.14",
            "denoise_layer": "model.15",
        },
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
# YOLO helpers / hooks
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


# -----------------------------------------------------------------------------
# ROI feature analysis for box-specific evidence
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


def per_channel_roi_mean_abs(feat: torch.Tensor, box: np.ndarray, img_hw: Tuple[int, int]) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat[0]
    c, h, w = feat.shape
    fx1, fy1, fx2, fy2 = box_to_feature_coords(box, img_hw, (h, w))
    roi = feat[:, fy1:fy2, fx1:fx2].abs().numpy().astype(np.float32)
    if roi.size == 0:
        return np.zeros((c,), dtype=np.float32)
    return roi.mean(axis=(1, 2))


def weighted_channel_map(feat: torch.Tensor, weights: np.ndarray, top_idx: np.ndarray) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat[0]
    x = feat.abs().numpy().astype(np.float32)
    if len(top_idx) == 0:
        return np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)
    w = weights[top_idx].astype(np.float32)
    w = w / (w.sum() + 1e-8)
    m = (x[top_idx] * w[:, None, None]).sum(axis=0)
    return m.astype(np.float32)


def summarize_top_channels(
    roi_base: np.ndarray,
    roi_dn: np.ndarray,
    topk: int,
) -> Dict[str, Any]:
    common_c = min(len(roi_base), len(roi_dn))
    roi_base = roi_base[:common_c]
    roi_dn = roi_dn[:common_c]
    suppression = np.maximum(roi_base - roi_dn, 0.0)
    scores = roi_base * suppression
    top_idx = np.argsort(-scores)[:topk]
    return {
        "common_c": common_c,
        "roi_base": roi_base,
        "roi_dn": roi_dn,
        "suppression": suppression,
        "scores": scores,
        "top_idx": top_idx,
    }


# -----------------------------------------------------------------------------
# figure builders
# -----------------------------------------------------------------------------
def make_pred_status_overlay(
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred: PredSummary,
    highlight_idx: Optional[int] = None,
    title_text: str = "",
) -> np.ndarray:
    """Prediction panel for triptychs.

    Rules:
    - Do NOT draw all GT boxes on prediction panels.
    - TP predictions: green boxes.
    - FP predictions: yellow boxes.
    - FN ground-truth boxes (missed GT): red boxes.
    """
    canvas = img_bgr.copy()

    # Draw missed GT boxes only (FN) in red.
    for j in pred.fn_gt_indices:
        box = gt_boxes[j]
        cls = int(gt_classes[j])
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"FN gt:{cls}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw predictions: TP in green, FP in yellow.
    for i, box in enumerate(pred.boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        color = (0, 255, 0) if pred.tp_flags[i] else (0, 255, 255)
        thick = 2
        if highlight_idx is not None and i == highlight_idx:
            thick = 3
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        tag = f"{int(pred.classes[i])}:{pred.confs[i]:.2f}"
        tag = ("TP " if pred.tp_flags[i] else "FP ") + tag
        cv2.putText(canvas, tag, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    if title_text:
        cv2.putText(canvas, title_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_dataset_summary_figure(summary_rows: List[Dict[str, Any]], out_png: Path) -> None:
    if not summary_rows:
        return
    total_base_tp = sum(r["baseline_tp"] for r in summary_rows)
    total_base_fp = sum(r["baseline_fp"] for r in summary_rows)
    total_base_fn = sum(r["baseline_fn"] for r in summary_rows)
    total_dn_tp = sum(r["denoise_tp"] for r in summary_rows)
    total_dn_fp = sum(r["denoise_fp"] for r in summary_rows)
    total_dn_fn = sum(r["denoise_fn"] for r in summary_rows)

    base_fp = np.array([r["baseline_fp"] for r in summary_rows], dtype=np.float32)
    dn_fp = np.array([r["denoise_fp"] for r in summary_rows], dtype=np.float32)
    base_tp = np.array([r["baseline_tp"] for r in summary_rows], dtype=np.float32)
    dn_tp = np.array([r["denoise_tp"] for r in summary_rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)

    names = ["TP", "FP", "FN"]
    base_vals = [total_base_tp, total_base_fp, total_base_fn]
    dn_vals = [total_dn_tp, total_dn_fp, total_dn_fn]
    x = np.arange(len(names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, base_vals, width=width, label="baseline")
    axes[0, 0].bar(x + width / 2, dn_vals, width=width, label="+Denoise")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names)
    axes[0, 0].set_title("dataset totals")
    axes[0, 0].grid(alpha=0.25, axis="y")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(base_fp, dn_fp, alpha=0.7)
    lim = max(float(base_fp.max()) if len(base_fp) else 0.0, float(dn_fp.max()) if len(dn_fp) else 0.0, 1.0)
    axes[0, 1].plot([0, lim], [0, lim], linestyle="--")
    axes[0, 1].set_xlabel("baseline FP / image")
    axes[0, 1].set_ylabel("+Denoise FP / image")
    axes[0, 1].set_title("per-image FP comparison")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].scatter(base_tp, dn_tp, alpha=0.7)
    lim2 = max(float(base_tp.max()) if len(base_tp) else 0.0, float(dn_tp.max()) if len(dn_tp) else 0.0, 1.0)
    axes[1, 0].plot([0, lim2], [0, lim2], linestyle="--")
    axes[1, 0].set_xlabel("baseline TP / image")
    axes[1, 0].set_ylabel("+Denoise TP / image")
    axes[1, 0].set_title("per-image TP comparison")
    axes[1, 0].grid(alpha=0.25)

    delta_fp = base_fp - dn_fp
    axes[1, 1].hist(delta_fp, bins=20)
    axes[1, 1].set_title("(baseline FP - +Denoise FP) histogram")
    axes[1, 1].set_xlabel("FP reduction per image")
    axes[1, 1].grid(alpha=0.25, axis="y")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_triptych_figure(
    out_png: Path,
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    base_pred: PredSummary,
    dn_pred: PredSummary,
    title_suffix: str = "",
) -> None:
    gt_panel = draw_gt_boxes(img_bgr, gt_boxes, gt_classes)
    base_panel = make_pred_status_overlay(img_bgr, gt_boxes, gt_classes, base_pred, highlight_idx=None, title_text="")
    dn_panel = make_pred_status_overlay(img_bgr, gt_boxes, gt_classes, dn_pred, highlight_idx=None, title_text="")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.8), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(gt_panel, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"GT{title_suffix}")
    axes[0].axis("off")

    axes[1].imshow(base_panel)
    axes[1].set_title(
        f"baseline\nTP={int(base_pred.tp_flags.sum())}, FP={int(base_pred.fp_flags.sum())}, FN={int(len(base_pred.fn_gt_indices))}"
    )
    axes[1].axis("off")

    axes[2].imshow(dn_panel)
    axes[2].set_title(
        f"+Denoise\nTP={int(dn_pred.tp_flags.sum())}, FP={int(dn_pred.fp_flags.sum())}, FN={int(len(dn_pred.fn_gt_indices))}"
    )
    axes[2].axis("off")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_triptych_rows(rows: List[Dict[str, Any]], mode: str, topk: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    mode = str(mode or "first")
    if mode == "first":
        return rows[:topk]
    if mode == "fp_gap":
        ordered = sorted(rows, key=lambda r: (r["baseline_fp"] - r["denoise_fp"], r["baseline_tp"] - r["denoise_tp"]), reverse=True)
        return ordered[:topk]
    if mode == "tp_gap":
        ordered = sorted(rows, key=lambda r: (r["denoise_tp"] - r["baseline_tp"], r["baseline_fp"] - r["denoise_fp"]), reverse=True)
        return ordered[:topk]
    if mode == "fp_then_tp":
        ordered = sorted(
            rows,
            key=lambda r: (
                r["baseline_fp"] - r["denoise_fp"],
                r["denoise_tp"] - r["baseline_tp"],
                -(r["denoise_fn"] - r["baseline_fn"]),
            ),
            reverse=True,
        )
        return ordered[:topk]
    return rows[:topk]


def save_fp_case_figure(
    out_png: Path,
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    base_pred: PredSummary,
    dn_pred: PredSummary,
    base_idx: int,
    dn_overlap_idx: int,
    pair_name: str,
    base_feat: torch.Tensor,
    dn_feat: torch.Tensor,
    top_summary: Dict[str, Any],
) -> Dict[str, float]:
    box = base_pred.boxes[base_idx]
    cls = int(base_pred.classes[base_idx])
    conf_b = float(base_pred.confs[base_idx])
    conf_dn = float(dn_pred.confs[dn_overlap_idx]) if dn_overlap_idx >= 0 else 0.0

    common_c = int(top_summary["common_c"])
    top_idx = top_summary["top_idx"]
    scores = top_summary["scores"]
    roi_base = top_summary["roi_base"]
    roi_dn = top_summary["roi_dn"]

    agg_base = weighted_channel_map(base_feat[:, :common_c], scores, top_idx)
    agg_dn = weighted_channel_map(dn_feat[:, :common_c], scores, top_idx)
    agg_change = agg_dn - agg_base

    p1 = overlay_heatmap_on_image(img_bgr, resize_map_to_image(percentile_normalize(agg_base), img_bgr.shape[:2]))
    p1 = cv2.cvtColor(draw_boxes(p1, np.array([box]), [f"baseline FP c{cls}:{conf_b:.2f}"], color=(0, 255, 255), thickness=3), cv2.COLOR_BGR2RGB)
    p2 = overlay_heatmap_on_image(img_bgr, resize_map_to_image(percentile_normalize(agg_dn), img_bgr.shape[:2]))
    p2 = cv2.cvtColor(draw_boxes(p2, np.array([box]), [f"denoise overlap:{conf_dn:.2f}"], color=(0, 255, 255), thickness=3), cv2.COLOR_BGR2RGB)
    ov1 = make_pred_status_overlay(img_bgr, gt_boxes, gt_classes, base_pred, highlight_idx=base_idx, title_text="baseline")
    ov2 = make_pred_status_overlay(img_bgr, gt_boxes, gt_classes, dn_pred, highlight_idx=None, title_text="+Denoise")

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(ov1)
    axes[0].set_title("baseline predictions")
    axes[0].axis("off")
    axes[1].imshow(ov2)
    axes[1].set_title("+Denoise predictions")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    axes[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='yellow', linewidth=3))
    axes[2].set_title(f"selected baseline-only FP\nclass={cls}, conf={conf_b:.2f}")
    axes[2].axis("off")

    axes[3].imshow(p1)
    axes[3].set_title(f"{pair_name} ROI-biased map\n(baseline)")
    axes[3].axis("off")
    axes[4].imshow(p2)
    axes[4].set_title(f"{pair_name} ROI-biased map\n(+Denoise)")
    axes[4].axis("off")
    im = axes[5].imshow(sym_percentile_normalize(resize_map_to_image(agg_change, img_bgr.shape[:2])), cmap="coolwarm", vmin=-1, vmax=1)
    axes[5].set_title(f"{pair_name} change\n(+Denoise - baseline)")
    axes[5].axis("off")
    fig.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # second figure: top harmful channels before/after
    fig2, axes2 = plt.subplots(2, len(top_idx), figsize=(3.1 * len(top_idx), 6.0), constrained_layout=True)
    if len(top_idx) == 1:
        axes2 = np.array(axes2).reshape(2, 1)
    for col, ch in enumerate(top_idx):
        base_map = base_feat[0, ch].numpy().astype(np.float32)
        dn_map = dn_feat[0, ch].numpy().astype(np.float32)
        b_overlay = overlay_heatmap_on_image(img_bgr, resize_map_to_image(percentile_normalize(np.abs(base_map)), img_bgr.shape[:2]))
        d_overlay = overlay_heatmap_on_image(img_bgr, resize_map_to_image(percentile_normalize(np.abs(dn_map)), img_bgr.shape[:2]))
        b_overlay = cv2.cvtColor(draw_boxes(b_overlay, np.array([box]), [f"ch{int(ch)}"], color=(0, 255, 255), thickness=3), cv2.COLOR_BGR2RGB)
        d_overlay = cv2.cvtColor(draw_boxes(d_overlay, np.array([box]), [f"ch{int(ch)}"], color=(0, 255, 255), thickness=3), cv2.COLOR_BGR2RGB)
        axes2[0, col].imshow(b_overlay)
        axes2[0, col].set_title(f"ch{int(ch)} before\nact={roi_base[ch]:.3f}")
        axes2[0, col].axis("off")
        axes2[1, col].imshow(d_overlay)
        axes2[1, col].set_title(f"ch{int(ch)} after\nact={roi_dn[ch]:.3f}")
        axes2[1, col].axis("off")
    fig2.suptitle(f"{pair_name} top suppressed channels on baseline-only FP", fontsize=14)
    fig2.savefig(out_png.with_name(out_png.stem + "_top_channels.png"), dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return {
        "baseline_fp_conf": conf_b,
        "denoise_overlap_conf": conf_dn,
        "mean_roi_before": float(np.mean(roi_base[top_idx])) if len(top_idx) else 0.0,
        "mean_roi_after": float(np.mean(roi_dn[top_idx])) if len(top_idx) else 0.0,
    }


# -----------------------------------------------------------------------------
# dataset case selection
# -----------------------------------------------------------------------------
@dataclass
class ImageCompareRow:
    image_path: str
    label_path: str
    baseline_tp: int
    baseline_fp: int
    baseline_fn: int
    denoise_tp: int
    denoise_fp: int
    denoise_fn: int


def find_images(root: Path, exts: Sequence[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def corresponding_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix('.txt')


def validate_config(cfg: dict) -> None:
    required = [
        "baseline_model", "denoise_model", "image_dir", "label_dir", "outdir", "imgsz",
        "conf", "device", "compare_pairs",
    ]
    for k in required:
        if k not in cfg:
            raise KeyError(f"CONFIG missing field: {k}")


def list_modules_for_both(cfg: dict) -> None:
    add_ultralytics_repo_to_path(cfg.get("ultra_root"))
    YOLO = import_yolo_class()
    base = YOLO(str(cfg["baseline_model"]))
    dn = YOLO(str(cfg["denoise_model"]))
    outdir = ensure_dir(cfg["outdir"])
    t1 = get_named_modules_text(base, "baseline")
    t2 = get_named_modules_text(dn, "+Denoise")
    print(t1)
    print(t2)
    (outdir / "99_named_modules_baseline.txt").write_text(t1, encoding="utf-8")
    (outdir / "99_named_modules_denoise.txt").write_text(t2, encoding="utf-8")


def build_models(cfg: dict):
    add_ultralytics_repo_to_path(cfg.get("ultra_root"))
    YOLO = import_yolo_class()
    base = YOLO(str(cfg["baseline_model"]))
    dn = YOLO(str(cfg["denoise_model"]))
    return base, dn


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = CONFIG
    validate_config(cfg)
    outdir = ensure_dir(cfg["outdir"])

    if cfg.get("list_modules_only", False):
        list_modules_for_both(cfg)
        return

    baseline_model, denoise_model = build_models(cfg)
    base_in_ch = get_first_conv_in_channels(baseline_model)
    dn_in_ch = get_first_conv_in_channels(denoise_model)

    image_root = Path(cfg["image_dir"])
    label_root = Path(cfg["label_dir"])
    image_paths = find_images(image_root, cfg.get("image_exts", []))
    if int(cfg.get("max_images", 0)) > 0:
        image_paths = image_paths[: int(cfg["max_images"])]

    # dataset pass: predictions + matching summary
    rows: List[Dict[str, Any]] = []
    fp_cases: List[Dict[str, Any]] = []
    tp_cases: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(image_paths):
        img_any = read_image_any(img_path)
        img_h, img_w = img_any.shape[:2]
        gt_boxes, gt_classes = read_yolo_label_file(corresponding_label_path(img_path, image_root, label_root), img_w, img_h, cfg.get("target_class"))

        base_source = build_predict_source(img_any, base_in_ch)
        dn_source = build_predict_source(img_any, dn_in_ch)
        base_res = run_predict(baseline_model, base_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
        dn_res = run_predict(denoise_model, dn_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
        base_boxes, base_confs, base_classes = extract_preds(base_res, cfg.get("target_class"))
        dn_boxes, dn_confs, dn_classes = extract_preds(dn_res, cfg.get("target_class"))

        base_sum = match_predictions_to_gt(base_boxes, base_confs, base_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))
        dn_sum = match_predictions_to_gt(dn_boxes, dn_confs, dn_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))

        rows.append({
            "image_path": str(img_path),
            "label_path": str(corresponding_label_path(img_path, image_root, label_root)),
            "baseline_tp": int(base_sum.tp_flags.sum()),
            "baseline_fp": int(base_sum.fp_flags.sum()),
            "baseline_fn": int(len(base_sum.fn_gt_indices)),
            "denoise_tp": int(dn_sum.tp_flags.sum()),
            "denoise_fp": int(dn_sum.fp_flags.sum()),
            "denoise_fn": int(len(dn_sum.fn_gt_indices)),
        })

        # baseline-only FP candidates
        for bi in np.where(base_sum.fp_flags)[0].tolist():
            iou_dn, di = best_overlap_same_class(base_sum.boxes[bi], int(base_sum.classes[bi]), dn_sum.boxes, dn_sum.classes)
            suppressed = (di < 0) or (iou_dn < float(cfg["fp_suppress_iou"]))
            if suppressed:
                fp_cases.append({
                    "image_path": str(img_path),
                    "baseline_idx": int(bi),
                    "baseline_conf": float(base_sum.confs[bi]),
                    "baseline_cls": int(base_sum.classes[bi]),
                    "denoise_overlap_iou": float(iou_dn),
                    "denoise_overlap_idx": int(di),
                })

        # preserved TP candidates
        for bi in np.where(base_sum.tp_flags)[0].tolist():
            gt_idx = int(base_sum.matched_gt[bi])
            preserved = any((dn_sum.tp_flags) & (dn_sum.matched_gt == gt_idx))
            if preserved:
                tp_cases.append({
                    "image_path": str(img_path),
                    "baseline_idx": int(bi),
                    "baseline_conf": float(base_sum.confs[bi]),
                    "matched_gt": gt_idx,
                })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] processed {img_path.name}")

    # save summary csv/text
    with (outdir / "01_per_image_compare.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    save_dataset_summary_figure(rows, outdir / "02_dataset_summary.png")

    total_base_tp = sum(r["baseline_tp"] for r in rows)
    total_base_fp = sum(r["baseline_fp"] for r in rows)
    total_base_fn = sum(r["baseline_fn"] for r in rows)
    total_dn_tp = sum(r["denoise_tp"] for r in rows)
    total_dn_fp = sum(r["denoise_fp"] for r in rows)
    total_dn_fn = sum(r["denoise_fn"] for r in rows)
    with (outdir / "00_dataset_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"images: {len(rows)}\n")
        f.write(f"baseline TP/FP/FN: {total_base_tp}/{total_base_fp}/{total_base_fn}\n")
        f.write(f"+Denoise TP/FP/FN: {total_dn_tp}/{total_dn_fp}/{total_dn_fn}\n")
        f.write(f"baseline-only FP candidates: {len(fp_cases)}\n")
        f.write(f"preserved TP candidates: {len(tp_cases)}\n")

    # always export a few GT / baseline / +Denoise triptychs first,
    # even if no baseline-only FP cases are found.
    if bool(cfg.get("export_triptychs", True)) and rows:
        trip_outdir = ensure_dir(outdir / "triptychs")
        selected = select_triptych_rows(rows, str(cfg.get("triptych_selection", "fp_then_tp")), int(cfg.get("topk_triptychs", 12)))
        for ti, row in enumerate(selected):
            img_path = Path(row["image_path"])
            img_any = read_image_any(img_path)
            img_h, img_w = img_any.shape[:2]
            img_bgr = to_bgr_uint8(img_any)
            gt_boxes, gt_classes = read_yolo_label_file(
                corresponding_label_path(img_path, image_root, label_root), img_w, img_h, cfg.get("target_class")
            )
            base_source = build_predict_source(img_any, base_in_ch)
            dn_source = build_predict_source(img_any, dn_in_ch)
            base_res = run_predict(baseline_model, base_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
            dn_res = run_predict(denoise_model, dn_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
            base_boxes, base_confs, base_classes = extract_preds(base_res, cfg.get("target_class"))
            dn_boxes, dn_confs, dn_classes = extract_preds(dn_res, cfg.get("target_class"))
            base_sum = match_predictions_to_gt(base_boxes, base_confs, base_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))
            dn_sum = match_predictions_to_gt(dn_boxes, dn_confs, dn_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))
            title_suffix = f"\nΔFP={row['baseline_fp'] - row['denoise_fp']}, ΔTP={row['denoise_tp'] - row['baseline_tp']}"
            save_triptych_figure(
                trip_outdir / f"{ti:02d}_{img_path.stem}_triptych.png",
                img_bgr,
                gt_boxes,
                gt_classes,
                base_sum,
                dn_sum,
                title_suffix=title_suffix,
            )

    # prepare hooks for case analysis
    base_modules = get_named_modules(baseline_model)
    dn_modules = get_named_modules(denoise_model)
    fp_outdir = ensure_dir(outdir / "fp_cases")

    # sort cases by baseline FP confidence descending
    fp_cases = sorted(fp_cases, key=lambda d: d["baseline_conf"], reverse=True)[: int(cfg["topk_fp_cases"])]

    pair_stats_rows: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(fp_cases):
        img_path = Path(case["image_path"])
        img_any = read_image_any(img_path)
        img_h, img_w = img_any.shape[:2]
        img_bgr = to_bgr_uint8(img_any)
        gt_boxes, gt_classes = read_yolo_label_file(corresponding_label_path(img_path, image_root, label_root), img_w, img_h, cfg.get("target_class"))

        base_source = build_predict_source(img_any, base_in_ch)
        dn_source = build_predict_source(img_any, dn_in_ch)

        for pair in cfg["compare_pairs"]:
            pair_name = str(pair["name"])
            base_layer = choose_module_by_name(base_modules, str(pair["baseline_layer"]))
            dn_layer = choose_module_by_name(dn_modules, str(pair["denoise_layer"]))
            base_hook = OutputHook(base_layer)
            dn_hook = OutputHook(dn_layer)
            try:
                base_res = run_predict(baseline_model, base_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
                dn_res = run_predict(denoise_model, dn_source, int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["nms_iou"]), str(cfg["device"]))
            finally:
                base_hook.close()
                dn_hook.close()

            if base_hook.output is None or dn_hook.output is None:
                print(f"[WARN] missing hook output for {img_path.name} / {pair_name}")
                continue

            base_boxes, base_confs, base_classes = extract_preds(base_res, cfg.get("target_class"))
            dn_boxes, dn_confs, dn_classes = extract_preds(dn_res, cfg.get("target_class"))
            base_sum = match_predictions_to_gt(base_boxes, base_confs, base_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))
            dn_sum = match_predictions_to_gt(dn_boxes, dn_confs, dn_classes, gt_boxes, gt_classes, float(cfg["match_iou"]))

            bi = int(case["baseline_idx"])
            if bi >= len(base_sum.boxes):
                continue
            iou_dn, di = best_overlap_same_class(base_sum.boxes[bi], int(base_sum.classes[bi]), dn_sum.boxes, dn_sum.classes)

            roi_base = per_channel_roi_mean_abs(base_hook.output, base_sum.boxes[bi], img_bgr.shape[:2])
            roi_dn = per_channel_roi_mean_abs(dn_hook.output, base_sum.boxes[bi], img_bgr.shape[:2])
            top_summary = summarize_top_channels(roi_base, roi_dn, int(cfg["topk_channels"]))
            fig_path = fp_outdir / f"{case_idx:02d}_{pair_name}_{img_path.stem}_fp_case.png"
            stat = save_fp_case_figure(
                out_png=fig_path,
                img_bgr=img_bgr,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                base_pred=base_sum,
                dn_pred=dn_sum,
                base_idx=bi,
                dn_overlap_idx=di,
                pair_name=pair_name,
                base_feat=base_hook.output,
                dn_feat=dn_hook.output,
                top_summary=top_summary,
            )
            pair_stats_rows.append({
                "case_index": case_idx,
                "pair": pair_name,
                "image": str(img_path),
                "baseline_cls": int(base_sum.classes[bi]),
                "baseline_conf": float(base_sum.confs[bi]),
                "denoise_overlap_iou": float(iou_dn),
                **stat,
            })

    if pair_stats_rows:
        with (outdir / "03_fp_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_stats_rows)

    # save module listings for convenience
    (outdir / "99_named_modules_baseline.txt").write_text(get_named_modules_text(baseline_model, "baseline"), encoding="utf-8")
    (outdir / "99_named_modules_denoise.txt").write_text(get_named_modules_text(denoise_model, "+Denoise"), encoding="utf-8")

    print("=" * 80)
    print("Saved:")
    print(outdir / "triptychs")
    for name in [
        "00_dataset_summary.txt",
        "01_per_image_compare.csv",
        "02_dataset_summary.png",
        "03_fp_case_metrics.csv",
        "99_named_modules_baseline.txt",
        "99_named_modules_denoise.txt",
    ]:
        p = outdir / name
        if p.exists():
            print(p)
    if fp_outdir.exists():
        print(fp_outdir)
    print("=" * 80)


if __name__ == "__main__":
    main()
