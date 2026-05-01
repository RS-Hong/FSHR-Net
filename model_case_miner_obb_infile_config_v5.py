import math
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

'''
保存对比结果
'''


# =========================
# 在这里集中修改参数
# =========================
CONFIG = {
    # 数据集根目录，目录结构要求：
    # dataset_root/
    #   images/
    #   labels/
    "dataset_root": r"D:\code\ultralytics-SAR-V2\YOLO_SSDD+\val",

    # 改进模型
    "improved_model_name": "Ours",
    "improved_model_weight": r"G:\对比权重\SSDD\Ours.pt",

    # 对比模型，按这里的顺序出现在 Excel 和拼图中
    "compare_models": OrderedDict(
        {
            "YOLOV8": r"G:\对比权重\SSDD\YOLOV8.pt",
            "YOLOV10": r"G:\对比权重\SSDD\YOLOV8.pt",
            "YOLO11": r"G:\对比权重\SSDD\YOLO11.pt",
            "YOLO12": r"G:\对比权重\SSDD\YOLO12.pt",
            "YOLO26": r"G:\对比权重\SSDD\YOLO26.pt",
        }
    ),

    # 输出目录
    "output_dir": r"D:\code\ultralytics-SAR-V2\绘图\SSDD",

    # 标签格式：
    # "auto" = 自动识别（推荐）：5列按YOLO水平框，9列按YOLO OBB四点框
    # "bbox" = 强制按  class cx cy w h  读取
    # "obb"  = 强制按  class x1 y1 x2 y2 x3 y3 x4 y4  读取
    "label_format": "auto",

    # 先按 GT 数量筛图，只保留 GT 数量 >= min_gt_count 的图片
    "min_gt_count": 4,

    # 选图时，要求改进模型的 AP50 至少达到这个阈值
    "improved_ap50_threshold": 0.80,

    # AP 差异比较用哪个指标："ap50" / "ap75" / "ap_mean"
    "ap_compare_metric": "ap50",

    # Precision 选图方式：
    # "absolute" = 选改进模型 Precision 最大的图（你这次要求的默认行为）
    # "delta"    = 选改进模型相对 baseline 的 Precision 差值最大的图
    "precision_selection_mode": "absolute",

    # 推理参数
    "imgsz": 640,
    "device": 0,            # 0 / '0' / 'cpu'
    "conf": 0.5,
    "iou_nms": 0.5,
    "max_det": 300,

    # 类别名（可选）
    # 如果为 None，则优先使用模型自带 names；若仍获取不到，就显示 cls_{id}
    "class_names": None,

    # 评估 IoU 阈值
    "eval_iou_thresholds": [0.5, 0.75],

    # 支持的图片后缀
    "image_suffixes": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],

    # 画图参数
    "panel_width": 720,
    "panel_height": 720,
    "title_height": 54,
    "margin": 16,
    "font_scale": 0.8,
    "font_thickness": 2,

    # 每行最左侧图片名标签面板
    "row_label_width": 280,
    "row_label_font_scale": 0.95,
    "row_label_font_thickness": 2,

    # 是否在预测框上方显示“类名 + 置信度”
    "show_pred_label_conf": False,

    # 是否限制处理图片数量；None 表示全量
    "limit_images": None,
}


# =========================
# 工具函数
# =========================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)



def list_images(images_dir, suffixes):
    suffixes = {s.lower() for s in suffixes}
    files = [p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files, key=lambda x: x.name)



def polygon_area(poly):
    """poly: (N,2)"""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)



def reorder_quad_clockwise(pts):
    """将四点重排成顺时针，便于计算与绘制。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -start, axis=0)
    return pts.astype(np.float32)



def quad_iou(quad1, quad2):
    """quad: (4,2), convex quadrilateral IoU"""
    q1 = reorder_quad_clockwise(quad1).astype(np.float32)
    q2 = reorder_quad_clockwise(quad2).astype(np.float32)
    a1 = polygon_area(q1)
    a2 = polygon_area(q2)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    ret, inter_poly = cv2.intersectConvexConvex(q1, q2)
    inter = float(ret) if ret is not None else 0.0
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))



def xyxy_to_quad(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)



def normalize_to_pixels(quad_norm, w, h):
    quad = np.asarray(quad_norm, dtype=np.float32).reshape(4, 2).copy()
    quad[:, 0] *= w
    quad[:, 1] *= h
    return quad



def bbox_cxcywh_to_quad(cx, cy, bw, bh, img_w, img_h):
    cx *= img_w
    cy *= img_h
    bw *= img_w
    bh *= img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy - bh / 2.0
    x3 = cx + bw / 2.0
    y3 = cy + bh / 2.0
    x4 = cx - bw / 2.0
    y4 = cy + bh / 2.0
    quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    return reorder_quad_clockwise(quad)


def load_label_as_quads(label_path, img_w, img_h, label_format="auto"):
    """
    统一读取标签并转成四点框 quad。

    支持三种模式：
    1) auto: 自动识别
       - 5列: class cx cy w h            (YOLO水平框)
       - 9列: class x1 y1 ... x4 y4      (YOLO OBB四点框)
    2) bbox: 强制按 YOLO 水平框读取
    3) obb : 强制按 YOLO OBB 四点框读取

    坐标默认按 YOLO 习惯视为归一化到 [0,1]。
    """
    gts = []
    if not Path(label_path).exists():
        return gts

    with open(label_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            fmt = label_format
            if fmt == "auto":
                if len(parts) == 5:
                    fmt = "bbox"
                elif len(parts) >= 9:
                    fmt = "obb"
                else:
                    continue

            cls_id = int(float(parts[0]))

            if fmt == "bbox":
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                quad = bbox_cxcywh_to_quad(cx, cy, bw, bh, img_w, img_h)
                gts.append({"cls": cls_id, "quad": quad})

            elif fmt == "obb":
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[1:9]))
                quad = np.array(coords, dtype=np.float32).reshape(4, 2)
                quad = normalize_to_pixels(quad, img_w, img_h)
                gts.append({"cls": cls_id, "quad": reorder_quad_clockwise(quad)})

            else:
                raise ValueError(f"不支持的 label_format: {label_format}")

    return gts



def get_class_name_map(models, fallback=None):
    if fallback is not None:
        return fallback
    for m in models.values():
        names = getattr(m, "names", None)
        if isinstance(names, dict) and names:
            return names
        if isinstance(names, (list, tuple)) and len(names) > 0:
            return {i: n for i, n in enumerate(names)}
    return None



def extract_predictions(result):
    preds = []
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        obb = result.obb
        xy = obb.xyxyxyxy.cpu().numpy() if hasattr(obb, "xyxyxyxy") else None
        conf = obb.conf.cpu().numpy() if hasattr(obb, "conf") else None
        cls = obb.cls.cpu().numpy() if hasattr(obb, "cls") else None
        if xy is not None:
            for i in range(len(xy)):
                quad = np.asarray(xy[i], dtype=np.float32).reshape(4, 2)
                preds.append(
                    {
                        "cls": int(cls[i]) if cls is not None else 0,
                        "conf": float(conf[i]) if conf is not None else 1.0,
                        "quad": reorder_quad_clockwise(quad),
                    }
                )
            return preds

    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None
        if xyxy is not None:
            for i in range(len(xyxy)):
                preds.append(
                    {
                        "cls": int(cls[i]) if cls is not None else 0,
                        "conf": float(conf[i]) if conf is not None else 1.0,
                        "quad": reorder_quad_clockwise(xyxy_to_quad(xyxy[i])),
                    }
                )
    return preds



def compute_pr_ap_for_image(gt_list, pred_list, iou_thr=0.5):
    gt_count = len(gt_list)
    pred_sorted = sorted(pred_list, key=lambda x: x["conf"], reverse=True)

    if gt_count == 0:
        if len(pred_sorted) == 0:
            return {"precision": 1.0, "recall": 1.0, "ap": 1.0, "tp": 0, "fp": 0, "fn": 0}
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "tp": 0, "fp": len(pred_sorted), "fn": 0}

    matched = [False] * gt_count
    tp_flags, fp_flags = [], []

    for pred in pred_sorted:
        best_iou = -1.0
        best_j = -1
        for j, gt in enumerate(gt_list):
            if matched[j]:
                continue
            if int(pred["cls"]) != int(gt["cls"]):
                continue
            iou = quad_iou(pred["quad"], gt["quad"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            matched[best_j] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum = np.cumsum(tp_flags) if len(tp_flags) else np.array([])
    fp_cum = np.cumsum(fp_flags) if len(fp_flags) else np.array([])

    if len(tp_cum) == 0:
        precision_curve = np.array([])
        recall_curve = np.array([])
        precision_final = 0.0
        recall_final = 0.0
        ap = 0.0
        tp_final = 0
        fp_final = 0
    else:
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recall_curve = tp_cum / max(gt_count, 1e-9)
        precision_final = float(precision_curve[-1])
        recall_final = float(recall_curve[-1])
        tp_final = int(tp_cum[-1])
        fp_final = int(fp_cum[-1])

        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], precision_curve, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    fn_final = int(gt_count - tp_final)
    return {
        "precision": precision_final,
        "recall": recall_final,
        "ap": ap,
        "tp": tp_final,
        "fp": fp_final,
        "fn": fn_final,
    }



def evaluate_image(gt_list, pred_list, iou_thresholds=(0.5, 0.75)):
    out = {}
    all_ap = []
    all_p = []
    all_r = []
    all_tp = []
    all_fp = []
    all_fn = []
    for thr in iou_thresholds:
        res = compute_pr_ap_for_image(gt_list, pred_list, iou_thr=thr)
        out[f"precision@{thr}"] = res["precision"]
        out[f"recall@{thr}"] = res["recall"]
        out[f"ap@{thr}"] = res["ap"]
        out[f"tp@{thr}"] = res["tp"]
        out[f"fp@{thr}"] = res["fp"]
        out[f"fn@{thr}"] = res["fn"]
        all_ap.append(res["ap"])
        all_p.append(res["precision"])
        all_r.append(res["recall"])
        all_tp.append(res["tp"])
        all_fp.append(res["fp"])
        all_fn.append(res["fn"])

    out["precision"] = out.get("precision@0.5", all_p[0])
    out["recall"] = out.get("recall@0.5", all_r[0])
    out["ap50"] = out.get("ap@0.5", all_ap[0])
    out["ap75"] = out.get("ap@0.75", all_ap[-1])
    out["ap_mean"] = float(np.mean(all_ap)) if all_ap else 0.0
    out["tp"] = out.get("tp@0.5", all_tp[0])
    out["fp"] = out.get("fp@0.5", all_fp[0])
    out["fn"] = out.get("fn@0.5", all_fn[0])
    return out



def run_inference(model, image_path, cfg):
    results = model.predict(
        source=str(image_path),
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=cfg["iou_nms"],
        max_det=cfg["max_det"],
        device=cfg["device"],
        verbose=False,
        save=False,
    )
    return extract_predictions(results[0])



def fit_to_canvas(img, width, height):
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas



def draw_quad(img, quad, color, thickness=2):
    pts = reorder_quad_clockwise(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)



def draw_label_box(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)



def class_name(cls_id, name_map):
    if isinstance(name_map, dict):
        return str(name_map.get(int(cls_id), f"cls_{int(cls_id)}"))
    return f"cls_{int(cls_id)}"



def match_predictions_for_display(gt_list, pred_list, iou_thr=0.5):
    pred_sorted_idx = sorted(range(len(pred_list)), key=lambda i: pred_list[i]["conf"], reverse=True)
    matched_gt = [False] * len(gt_list)
    pred_status = ["fp"] * len(pred_list)
    pred_to_gt = [-1] * len(pred_list)

    for pi in pred_sorted_idx:
        pred = pred_list[pi]
        best_iou = -1.0
        best_j = -1
        for j, gt in enumerate(gt_list):
            if matched_gt[j]:
                continue
            if int(pred["cls"]) != int(gt["cls"]):
                continue
            iou = quad_iou(pred["quad"], gt["quad"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt[best_j] = True
            pred_status[pi] = "tp"
            pred_to_gt[pi] = best_j

    gt_status = ["fn"] * len(gt_list)
    for gi, ok in enumerate(matched_gt):
        if ok:
            gt_status[gi] = "matched"

    return pred_status, gt_status, pred_to_gt



def render_panel(image_bgr, gt_list=None, pred_list=None, pred_status=None, gt_status=None, title="", name_map=None, cfg=None, mode="original"):
    if cfg is None:
        cfg = CONFIG
    panel = image_bgr.copy()
    gt_list = gt_list or []

    if mode == "gt":
        for gt in gt_list:
            draw_quad(panel, gt["quad"], color=(0, 200, 0), thickness=2)

    elif mode == "pred":
        gt_status = gt_status or ["fn"] * len(gt_list)
        for gt, status in zip(gt_list, gt_status):
            if status == "fn":
                draw_quad(panel, gt["quad"], color=(0, 0, 255), thickness=2)

        if pred_list is not None:
            if pred_status is None:
                pred_status = ["fp"] * len(pred_list)
            for pred, status in zip(pred_list, pred_status):
                if status == "tp":
                    color = (255, 0, 0)
                    tag = "TP"
                else:
                    color = (0, 255, 255)
                    tag = "FP"
                draw_quad(panel, pred["quad"], color=color, thickness=2)
                if cfg.get("show_pred_label_conf", True):
                    p = reorder_quad_clockwise(pred["quad"])[0]
                    text = f"{tag}:{class_name(pred['cls'], name_map)} {pred['conf']:.2f}"
                    draw_label_box(panel, text, p[0], p[1], color)

    panel = fit_to_canvas(panel, cfg["panel_width"], cfg["panel_height"])
    title_bar = np.full((cfg["title_height"], cfg["panel_width"], 3), 245, dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (12, cfg["title_height"] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg["font_scale"],
        (30, 30, 30),
        cfg["font_thickness"],
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, panel])



def wrap_text_by_width(text, max_width, font, scale, thickness):
    words = str(text).replace("\\", "/").split("/")
    segments = []
    current = ""
    for word in words:
        candidate = word if not current else current + "/" + word
        size = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if size[0] <= max_width or not current:
            current = candidate
        else:
            segments.append(current)
            current = word
    if current:
        segments.append(current)

    final_lines = []
    for seg in segments:
        if cv2.getTextSize(seg, font, scale, thickness)[0][0] <= max_width:
            final_lines.append(seg)
            continue

        current = ""
        for ch in seg:
            candidate = ch if not current else current + ch
            size = cv2.getTextSize(candidate, font, scale, thickness)[0]
            if size[0] <= max_width or not current:
                current = candidate
            else:
                final_lines.append(current)
                current = ch
        if current:
            final_lines.append(current)

    return final_lines


def render_row_label_panel(image_name, cfg):
    panel_h = cfg["title_height"] + cfg["panel_height"]
    panel_w = cfg.get("row_label_width", 280)
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cfg.get("row_label_font_scale", 0.95)
    thickness = cfg.get("row_label_font_thickness", 2)
    text_color = (20, 20, 20)
    border_color = (180, 180, 180)

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), border_color, 2)

    lines = wrap_text_by_width(image_name, panel_w - 24, font, scale, thickness)
    line_h = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 12
    total_h = len(lines) * line_h
    y = max(40, (panel_h - total_h) // 2 + line_h - 4)

    for line in lines:
        tw = cv2.getTextSize(line, font, scale, thickness)[0][0]
        x = max(12, (panel_w - tw) // 2)
        cv2.putText(panel, line, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
        y += line_h

    return panel



def make_comparison_figure(case_rows, compare_names, improved_name, output_path, image_cache, gt_cache, pred_cache, metrics_cache, name_map, cfg):
    cols = ["ImageName", "Original", "GT"] + list(compare_names) + [improved_name]
    ncols = len(cols)
    nrows = len(case_rows)
    panel_h = cfg["title_height"] + cfg["panel_height"]
    panel_w = cfg["panel_width"]
    label_w = cfg.get("row_label_width", 280)
    margin = cfg["margin"]

    H = margin + nrows * panel_h + (nrows - 1) * margin + margin
    W = margin + label_w + margin + (ncols - 1) * panel_w + (ncols - 2) * margin + margin
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    for r, case in enumerate(case_rows):
        image_name = case["image_name"]
        image_bgr = image_cache[image_name]
        gt_list = gt_cache[image_name]
        y0 = margin + r * (panel_h + margin)

        label_panel = render_row_label_panel(image_name, cfg)
        x_label = margin
        canvas[y0:y0 + label_panel.shape[0], x_label:x_label + label_panel.shape[1]] = label_panel

        row_models = ["__original__", "__gt__"] + list(compare_names) + [improved_name]
        for c, model_name in enumerate(row_models):
            if model_name == "__original__":
                title = f"Original | GT:{len(gt_list)} | {case['metric_tag']}"
                panel = render_panel(
                    image_bgr=image_bgr,
                    gt_list=[],
                    pred_list=None,
                    pred_status=None,
                    gt_status=None,
                    title=title,
                    name_map=name_map,
                    cfg=cfg,
                    mode="original",
                )
            elif model_name == "__gt__":
                title = f"GT | Count:{len(gt_list)}"
                panel = render_panel(
                    image_bgr=image_bgr,
                    gt_list=gt_list,
                    pred_list=None,
                    pred_status=None,
                    gt_status=None,
                    title=title,
                    name_map=name_map,
                    cfg=cfg,
                    mode="gt",
                )
            else:
                pred_list = pred_cache[(image_name, model_name)]
                pred_status, gt_status, _ = match_predictions_for_display(gt_list, pred_list, iou_thr=0.5)
                m = metrics_cache[(image_name, model_name)]
                title = f"{model_name} | P:{m['precision']:.3f} R:{m['recall']:.3f} AP50:{m['ap50']:.3f} AP75:{m['ap75']:.3f}"
                panel = render_panel(
                    image_bgr=image_bgr,
                    gt_list=gt_list,
                    pred_list=pred_list,
                    pred_status=pred_status,
                    gt_status=gt_status,
                    title=title,
                    name_map=name_map,
                    cfg=cfg,
                    mode="pred",
                )

            x0 = margin + label_w + margin + c * (panel_w + margin)
            canvas[y0:y0 + panel.shape[0], x0:x0 + panel.shape[1]] = panel

    cv2.imwrite(str(output_path), canvas)



def build_image_metric_row(image_name, gt_count, model_names, metrics_cache):
    row = {
        "image_name": image_name,
        "gt_count": gt_count,
    }
    for model_name in model_names:
        m = metrics_cache[(image_name, model_name)]
        row[f"{model_name}__precision"] = m["precision"]
        row[f"{model_name}__recall"] = m["recall"]
        row[f"{model_name}__ap50"] = m["ap50"]
        row[f"{model_name}__ap75"] = m["ap75"]
    return row



def select_case_rows(valid_images, compare_names, improved_name, metrics_cache, gt_cache, cfg):
    threshold = float(cfg["improved_ap50_threshold"])
    precision_mode = cfg.get("precision_selection_mode", "absolute")
    ap_metric = cfg.get("ap_compare_metric", "ap50")

    top_case_rows = []
    for baseline_name in compare_names:
        rows = []
        for image_name in valid_images:
            imp = metrics_cache[(image_name, improved_name)]
            if imp["ap50"] < threshold:
                continue
            base = metrics_cache[(image_name, baseline_name)]
            rows.append(
                {
                    "baseline_model": baseline_name,
                    "image_name": image_name,
                    "gt_count": len(gt_cache[image_name]),
                    "delta_precision": imp["precision"] - base["precision"],
                    "delta_recall": imp["recall"] - base["recall"],
                    "delta_ap50": imp["ap50"] - base["ap50"],
                    "delta_ap75": imp["ap75"] - base["ap75"],
                    "delta_ap_mean": imp["ap_mean"] - base["ap_mean"],
                    "improved_precision": imp["precision"],
                    "improved_recall": imp["recall"],
                    "improved_ap50": imp["ap50"],
                    "improved_ap75": imp["ap75"],
                    "baseline_precision": base["precision"],
                    "baseline_recall": base["recall"],
                    "baseline_ap50": base["ap50"],
                    "baseline_ap75": base["ap75"],
                }
            )

        if not rows:
            print(f"[WARN] 相对 {baseline_name} 没有满足 improved AP50 >= {threshold:.2f} 的候选图片，跳过出图。")
            continue

        df_delta = pd.DataFrame(rows)
        ap_delta_col = f"delta_{ap_metric}"
        if ap_delta_col not in df_delta.columns:
            ap_delta_col = "delta_ap50"

        top_ap = df_delta.sort_values([ap_delta_col, "improved_ap50", "improved_recall"], ascending=False).iloc[0].to_dict()
        top_recall = df_delta.sort_values(["delta_recall", "improved_ap50", "improved_precision"], ascending=False).iloc[0].to_dict()

        if precision_mode == "delta":
            top_precision = df_delta.sort_values(["delta_precision", "improved_ap50", "improved_recall"], ascending=False).iloc[0].to_dict()
            precision_desc = "max ΔPrecision"
        else:
            top_precision = df_delta.sort_values(["improved_precision", "improved_ap50", "delta_precision"], ascending=False).iloc[0].to_dict()
            precision_desc = "max Precision"

        top_ap["metric_tag"] = f"vs {baseline_name} | max Δ{ap_metric.upper()}"
        top_recall["metric_tag"] = f"vs {baseline_name} | max ΔRecall"
        top_precision["metric_tag"] = f"vs {baseline_name} | {precision_desc}"

        top_case_rows.extend([top_ap, top_recall, top_precision])

    return top_case_rows



def main():
    cfg = CONFIG
    dataset_root = Path(cfg["dataset_root"])
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    output_dir = Path(cfg["output_dir"])
    figures_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"未找到 images/ 或 labels/ 目录：{dataset_root}")

    image_files = list_images(images_dir, cfg["image_suffixes"])
    if cfg["limit_images"] is not None:
        image_files = image_files[: int(cfg["limit_images"])]
    if not image_files:
        raise RuntimeError(f"在 {images_dir} 中没有找到图片。")

    # 先按 GT 数量筛图
    image_cache = {}
    gt_cache = {}
    filtered_image_files = []
    gt_filter_rows = []

    print(f"[INFO] 共发现 {len(image_files)} 张图片，先按 GT 数量 >= {cfg['min_gt_count']} 过滤...")
    for img_path in tqdm(image_files, ncols=100):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 读取失败，跳过：{img_path}")
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_list = load_label_as_quads(label_path, w, h, cfg.get("label_format", "auto"))
        gt_count = len(gt_list)
        keep = gt_count >= int(cfg["min_gt_count"])
        gt_filter_rows.append(
            {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "label_path": str(label_path),
                "gt_count": gt_count,
                "kept": keep,
            }
        )
        if keep:
            image_cache[img_path.name] = img
            gt_cache[img_path.name] = gt_list
            filtered_image_files.append(img_path)

    if not filtered_image_files:
        raise RuntimeError(
            f"没有图片满足 GT 数量 >= {cfg['min_gt_count']} 的条件，请调小 min_gt_count。"
        )

    # 加载模型
    models = OrderedDict()
    for name, weight in cfg["compare_models"].items():
        models[name] = YOLO(weight)
    models[cfg["improved_model_name"]] = YOLO(cfg["improved_model_weight"])
    compare_names = list(cfg["compare_models"].keys())
    improved_name = cfg["improved_model_name"]
    all_model_names = compare_names + [improved_name]
    name_map = get_class_name_map(models, cfg.get("class_names", None))

    pred_cache = {}      # key: (image_name, model_name)
    metrics_cache = {}   # key: (image_name, model_name)
    long_rows = []

    print(f"[INFO] 过滤后保留 {len(filtered_image_files)} 张图片，开始逐图评估...")
    for img_path in tqdm(filtered_image_files, ncols=100):
        image_name = img_path.name
        gt_list = gt_cache[image_name]
        label_path = labels_dir / f"{img_path.stem}.txt"

        for model_name, model in models.items():
            pred_list = run_inference(model, img_path, cfg)
            metrics = evaluate_image(gt_list, pred_list, iou_thresholds=tuple(cfg["eval_iou_thresholds"]))
            pred_cache[(image_name, model_name)] = pred_list
            metrics_cache[(image_name, model_name)] = metrics

            row = {
                "image_name": image_name,
                "image_path": str(img_path),
                "label_path": str(label_path),
                "gt_count": len(gt_list),
                "model_name": model_name,
                "pred_count": len(pred_list),
                **metrics,
            }
            long_rows.append(row)

    if not long_rows:
        raise RuntimeError("没有得到任何评估结果，请检查路径、权重和标签格式。")

    df_long = pd.DataFrame(long_rows)
    df_summary = (
        df_long.groupby("model_name")[["precision", "recall", "ap50", "ap75", "ap_mean", "tp", "fp", "fn", "pred_count", "gt_count"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(by=["ap50", "ap75", "recall", "precision"], ascending=False)
    )

    # 用户要的主表：每行一张图，保存图片名、GT 数量、每个模型的 P/R/AP50/AP75
    valid_images = sorted([p.name for p in filtered_image_files])
    main_rows = [build_image_metric_row(name, len(gt_cache[name]), all_model_names, metrics_cache) for name in valid_images]
    df_main = pd.DataFrame(main_rows)

    # 可选：保留 image_path 方便回查
    df_main.insert(1, "image_path", [str(images_dir / name) for name in df_main["image_name"]])

    top_case_rows = select_case_rows(valid_images, compare_names, improved_name, metrics_cache, gt_cache, cfg)
    df_top_cases = pd.DataFrame(top_case_rows)

    if not df_top_cases.empty:
        for baseline_name in compare_names:
            sub = df_top_cases[df_top_cases["baseline_model"] == baseline_name].copy()
            if sub.empty:
                continue
            order = [
                f"vs {baseline_name} | max Δ{cfg.get('ap_compare_metric', 'ap50').upper()}",
                f"vs {baseline_name} | max ΔRecall",
                f"vs {baseline_name} | max Precision" if cfg.get("precision_selection_mode", "absolute") != "delta" else f"vs {baseline_name} | max ΔPrecision",
            ]
            sub["_order"] = pd.Categorical(sub["metric_tag"], categories=order, ordered=True)
            sub = sub.sort_values("_order")

            fig_path = figures_dir / f"compare_vs_{baseline_name}.jpg"
            make_comparison_figure(
                case_rows=sub.to_dict("records"),
                compare_names=compare_names,
                improved_name=improved_name,
                output_path=fig_path,
                image_cache=image_cache,
                gt_cache=gt_cache,
                pred_cache=pred_cache,
                metrics_cache=metrics_cache,
                name_map=name_map,
                cfg=cfg,
            )
            print(f"[INFO] 已保存对比图：{fig_path}")

    # 导出 Excel
    excel_path = output_dir / "per_image_metrics_filtered.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(gt_filter_rows).to_excel(writer, sheet_name="gt_filter", index=False)
        df_summary.to_excel(writer, sheet_name="summary_by_model", index=False)
        df_main.to_excel(writer, sheet_name="per_image_main", index=False)
        df_long.to_excel(writer, sheet_name="per_image_long", index=False)
        df_top_cases.to_excel(writer, sheet_name="top_cases", index=False)

    print(f"[INFO] 已保存 Excel：{excel_path}")
    print(f"[INFO] 全部完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
