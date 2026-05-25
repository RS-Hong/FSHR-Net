import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

"""
根据test文件中的图像和标签，选择适合展示的图片
"""

# =========================================================
# 1. 配置区：直接在这里改参数
# =========================================================
TEST_DIR = r"YOLO_HRSID\test"  # 测试集根目录，内部应有 images/ 和 labels/
BASELINE_WEIGHT = r"runs\detect\HRSID-YOLOv8-E300-B16-test\weights\best.pt"  # baseline 权重
IMPROVED_WEIGHT = r"runs\detect\HRSID_YOLOv8-denoise-E300-B16\weights\best.pt"  # 改进模型权重
OUTPUT_CSV = r"output\compare_metrics_denoise.csv"  # 输出 csv 路径

CONF_THRESH = 0.25  # 预测置信度阈值
PRED_IOU_THRESH = 0.7  # 模型预测时 NMS 的 IoU 阈值
MATCH_IOU_THRESH = 0.5  # 预测框与 GT 匹配时的 IoU 阈值
IMGSZ = 640  # 推理尺寸
DEVICE = "0"  # "0" 表示第0块GPU，"cpu" 表示CPU
CLASS_AWARE = True  # True: 匹配时要求类别一致；单类别检测一般无影响


# 支持的图像后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def xywhn_to_xyxy(box, img_w, img_h):
    """YOLO标签格式: cls cx cy w h (归一化) -> 转为像素坐标 xyxy.
    """
    cls_id, cx, cy, w, h = box
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return [int(cls_id), x1, y1, x2, y2]


def read_yolo_label(label_path: str, img_w: int, img_h: int) -> list[list[float]]:
    """读取单个 YOLO txt 标签 返回格式: [[cls, x1, y1, x2, y2], ...].
    """
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        vals = list(map(float, parts[:5]))
        gt_boxes.append(xywhn_to_xyxy(vals, img_w, img_h))

    return gt_boxes


def compute_iou(box1, box2):
    """Box 格式: [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-16
    return inter / union


def greedy_match(
    gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_thr: float = 0.5, class_aware: bool = True
):
    """gt_boxes: [[cls, x1, y1, x2, y2], ...] pred_boxes: [[cls, x1, y1, x2, y2, conf], ...].

    返回:
        matches: [(gt_idx, pred_idx, iou), ...]
        unmatched_gt: 未匹配GT索引
        unmatched_pred: 未匹配预测索引
    """
    candidates = []

    for gi, gt in enumerate(gt_boxes):
        gt_cls, gx1, gy1, gx2, gy2 = gt
        for pi, pred in enumerate(pred_boxes):
            p_cls, px1, py1, px2, py2, _conf = pred

            if class_aware and int(gt_cls) != int(p_cls):
                continue

            iou = compute_iou([gx1, gy1, gx2, gy2], [px1, py1, px2, py2])
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))

    # 按 IoU 从大到小排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def evaluate_one_image(
    gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_thr: float = 0.5, class_aware: bool = True
) -> dict[str, float]:
    """单张图指标: - precision = TP / (TP + FP) - recall = TP / (TP + FN) - mIoU = 所有成功匹配对的平均 IoU.

    特殊情况:
    - 无GT且无预测 -> precision=1, recall=1, mIoU=1
    - 无GT但有预测 -> precision=0, recall=1, mIoU=0
    - 有GT但无预测 -> precision=0, recall=0, mIoU=0
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "miou": 1.0}

    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return {"tp": 0, "fp": len(pred_boxes), "fn": 0, "precision": 0.0, "recall": 1.0, "miou": 0.0}

    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return {"tp": 0, "fp": 0, "fn": len(gt_boxes), "precision": 0.0, "recall": 0.0, "miou": 0.0}

    matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes, iou_thr=iou_thr, class_aware=class_aware)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    miou = float(np.mean([m[2] for m in matches])) if tp > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "miou": miou}


def get_image_list(images_dir: str) -> list[str]:
    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    image_paths = sorted(list(set(image_paths)))
    return image_paths


def run_model_on_image(model, image_path, conf=0.25, iou=0.7, imgsz=640, device="0"):
    """返回预测框: [[cls, x1, y1, x2, y2, conf], ...].
    """
    results = model.predict(source=image_path, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

    pred_boxes = []
    if len(results) == 0:
        return pred_boxes

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return pred_boxes

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    boxes_cls = r.boxes.cls.cpu().numpy()
    boxes_conf = r.boxes.conf.cpu().numpy()

    for b, c, s in zip(boxes_xyxy, boxes_cls, boxes_conf):
        x1, y1, x2, y2 = b.tolist()
        pred_boxes.append([int(c), x1, y1, x2, y2, float(s)])

    return pred_boxes


def compare_two_models():
    images_dir = os.path.join(TEST_DIR, "images")
    labels_dir = os.path.join(TEST_DIR, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"未找到 images 路径: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"未找到 labels 路径: {labels_dir}")

    image_list = get_image_list(images_dir)
    if len(image_list) == 0:
        raise FileNotFoundError(f"在 {images_dir} 下未找到图片")

    print(f"共找到 {len(image_list)} 张图片")
    print("加载模型中...")
    baseline_model = YOLO(BASELINE_WEIGHT)
    improved_model = YOLO(IMPROVED_WEIGHT)

    rows = []

    for idx, image_path in enumerate(image_list, 1):
        image_name = os.path.basename(image_path)
        stem = Path(image_name).stem
        label_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[跳过] 无法读取图片: {image_path}")
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = read_yolo_label(label_path, img_w, img_h)

        baseline_preds = run_model_on_image(
            baseline_model, image_path, conf=CONF_THRESH, iou=PRED_IOU_THRESH, imgsz=IMGSZ, device=DEVICE
        )

        improved_preds = run_model_on_image(
            improved_model, image_path, conf=CONF_THRESH, iou=PRED_IOU_THRESH, imgsz=IMGSZ, device=DEVICE
        )

        baseline_metrics = evaluate_one_image(
            gt_boxes, baseline_preds, iou_thr=MATCH_IOU_THRESH, class_aware=CLASS_AWARE
        )

        improved_metrics = evaluate_one_image(
            gt_boxes, improved_preds, iou_thr=MATCH_IOU_THRESH, class_aware=CLASS_AWARE
        )

        row = {
            "image_name": image_name,
            "gt_num": len(gt_boxes),
            "baseline_pred_num": len(baseline_preds),
            "baseline_tp": baseline_metrics["tp"],
            "baseline_fp": baseline_metrics["fp"],
            "baseline_fn": baseline_metrics["fn"],
            "baseline_mIoU": baseline_metrics["miou"],
            "baseline_recall": baseline_metrics["recall"],
            "baseline_precision": baseline_metrics["precision"],
            "improved_pred_num": len(improved_preds),
            "improved_tp": improved_metrics["tp"],
            "improved_fp": improved_metrics["fp"],
            "improved_fn": improved_metrics["fn"],
            "improved_mIoU": improved_metrics["miou"],
            "improved_recall": improved_metrics["recall"],
            "improved_precision": improved_metrics["precision"],
        }

        rows.append(row)

        print(
            f"[{idx}/{len(image_list)}] {image_name} | "
            f"Baseline: mIoU={row['baseline_mIoU']:.4f}, "
            f"R={row['baseline_recall']:.4f}, "
            f"P={row['baseline_precision']:.4f} | "
            f"Improved: mIoU={row['improved_mIoU']:.4f}, "
            f"R={row['improved_recall']:.4f}, "
            f"P={row['improved_precision']:.4f}"
        )

    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)

    fieldnames = [
        "image_name",
        "gt_num",
        "baseline_pred_num",
        "baseline_tp",
        "baseline_fp",
        "baseline_fn",
        "baseline_mIoU",
        "baseline_recall",
        "baseline_precision",
        "improved_pred_num",
        "improved_tp",
        "improved_fp",
        "improved_fn",
        "improved_mIoU",
        "improved_recall",
        "improved_precision",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n结果已保存到: {OUTPUT_CSV}")

    if len(rows) > 0:
        baseline_mean_miou = np.mean([r["baseline_mIoU"] for r in rows])
        baseline_mean_recall = np.mean([r["baseline_recall"] for r in rows])
        baseline_mean_precision = np.mean([r["baseline_precision"] for r in rows])

        improved_mean_miou = np.mean([r["improved_mIoU"] for r in rows])
        improved_mean_recall = np.mean([r["improved_recall"] for r in rows])
        improved_mean_precision = np.mean([r["improved_precision"] for r in rows])

        print("\n===== 按图片平均后的结果 =====")
        print(
            f"Baseline : mIoU={baseline_mean_miou:.4f}, "
            f"Recall={baseline_mean_recall:.4f}, "
            f"Precision={baseline_mean_precision:.4f}"
        )
        print(
            f"Improved : mIoU={improved_mean_miou:.4f}, "
            f"Recall={improved_mean_recall:.4f}, "
            f"Precision={improved_mean_precision:.4f}"
        )


if __name__ == "__main__":
    compare_two_models()
