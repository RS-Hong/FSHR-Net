from pathlib import Path

import cv2
import numpy as np

from ultralytics import RTDETR, YOLO

# =========================
# 配置区：直接改这里
# =========================
MODEL_PATH = r"G:\对比权重\HRSID\Ours.pt"
DATA_ROOT = r"images/HRSID"  # 里面必须包含 images 和 labels
SAVE_DIR = r"D:\code\ultralytics-SAR-V2\optimg\HRSID\OursHRSID"

USE_RTDETR = False  # RT-DETR权重就改成 True；YOLO权重就改成 False
IMGSZ = 640
CONF_THRES = 0.5  # 预测置信度阈值
IOU_NMS_THRES = 0.4  # 预测阶段 NMS 阈值
IOU_MATCH_THRES = 0.5  # 评估/显示 TP-FP-FN 时，预测与GT的匹配阈值
MAX_DET = 300
DEVICE = "0"  # "0" 表示 cuda:0, "cpu" 表示 CPU，空字符串表示自动

# 标签格式：
# "auto" = 自动识别（推荐）：5列按YOLO水平框，9列按YOLO OBB四点框
# "bbox" = 强制按  class cx cy w h  读取
# "obb"  = 强制按  class x1 y1 x2 y2 x3 y3 x4 y4  读取
LABEL_FORMAT = "auto"

# 是否递归读取 images 下所有子目录
RECURSIVE = True

# 是否显示文字（TP/FP/FN + 类别 + 置信度）
SHOW_TEXT = False

# 支持的图片后缀
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def polygon_area(poly):
    """Poly: (N,2)."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def reorder_quad_clockwise(pts):
    """将四点重排成顺时针，便于计算与绘制。."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -start, axis=0)
    return pts.astype(np.float32)


def quad_iou(quad1, quad2):
    """Quad: (4,2), convex quadrilateral IoU."""
    q1 = reorder_quad_clockwise(quad1).astype(np.float32)
    q2 = reorder_quad_clockwise(quad2).astype(np.float32)
    a1 = polygon_area(q1)
    a2 = polygon_area(q2)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    ret, _ = cv2.intersectConvexConvex(q1, q2)
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
    """统一读取标签并转成四点框 quad。.

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

    with open(label_path, encoding="utf-8") as f:
        for line in f:
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


def build_model(model_path, use_rtdetr=False):
    if use_rtdetr:
        return RTDETR(model_path)
    return YOLO(model_path)


def collect_images(images_dir):
    if RECURSIVE:
        files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES]
    else:
        files = [p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES]
    return sorted(files)


def get_class_name(class_names, cls_id):
    if isinstance(class_names, dict):
        return class_names.get(cls_id, str(cls_id))
    if isinstance(class_names, list) and 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return str(cls_id)


def extract_predictions(result):
    preds = []

    # 优先读取 OBB 结果，与 case miner 对齐
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

    # 没有 OBB 时退回水平框
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


def greedy_match_quads(preds, gts, iou_thr=0.5):
    """preds: [{"cls": int, "conf": float, "quad": (4,2)}, ...] gts: [{"cls": int, "quad": (4,2)}, ...].
    """
    candidates = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if int(p["cls"]) != int(g["cls"]):
                continue
            iou = quad_iou(p["quad"], g["quad"])
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matched_pairs = []

    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_pairs.append((pi, gi))

    unmatched_pred = [i for i in range(len(preds)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gts)) if i not in matched_gt]

    return matched_pairs, unmatched_pred, unmatched_gt


def draw_quad(img, quad, color, thickness=2):
    pts = reorder_quad_clockwise(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_label_box(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    data_root = Path(DATA_ROOT)
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    save_dir = Path(SAVE_DIR)

    save_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {images_dir}")
    if not labels_dir.exists():
        print(f"警告：未找到标签目录 {labels_dir}，将按空标签处理")

    model = build_model(MODEL_PATH, USE_RTDETR)
    class_names = model.names if hasattr(model, "names") else {}

    image_files = collect_images(images_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {images_dir} 下未找到图像文件")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"跳过无法读取的图像: {img_path}")
            continue

        h, w = img.shape[:2]

        rel_path = img_path.relative_to(images_dir)
        label_path = labels_dir / rel_path.with_suffix(".txt")

        gts = load_label_as_quads(label_path, w, h, LABEL_FORMAT)

        result = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_NMS_THRES,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False,
            save=False,
        )[0]

        preds = extract_predictions(result)

        matched_pairs, unmatched_pred_idxs, unmatched_gt_idxs = greedy_match_quads(preds, gts, iou_thr=IOU_MATCH_THRES)

        vis = img.copy()

        # TP：蓝色
        for pi, gi in matched_pairs:
            p = preds[pi]
            cls_name = get_class_name(class_names, int(p["cls"]))
            p0 = reorder_quad_clockwise(p["quad"])[0]
            text = f"TP {cls_name} {p['conf']:.2f}" if SHOW_TEXT else None
            draw_quad(vis, p["quad"], (255, 0, 0), thickness=2)
            if SHOW_TEXT and text:
                draw_label_box(vis, text, p0[0], p0[1], (255, 0, 0))

        # FP：黄色
        for pi in unmatched_pred_idxs:
            p = preds[pi]
            cls_name = get_class_name(class_names, int(p["cls"]))
            p0 = reorder_quad_clockwise(p["quad"])[0]
            text = f"FP {cls_name} {p['conf']:.2f}" if SHOW_TEXT else None
            draw_quad(vis, p["quad"], (0, 255, 255), thickness=2)
            if SHOW_TEXT and text:
                draw_label_box(vis, text, p0[0], p0[1], (0, 255, 255))

        # FN：红色
        for gi in unmatched_gt_idxs:
            g = gts[gi]
            cls_name = get_class_name(class_names, int(g["cls"]))
            g0 = reorder_quad_clockwise(g["quad"])[0]
            text = f"FN {cls_name}" if SHOW_TEXT else None
            draw_quad(vis, g["quad"], (0, 0, 255), thickness=2)
            if SHOW_TEXT and text:
                draw_label_box(vis, text, g0[0], g0[1], (0, 0, 255))

        total_tp += len(matched_pairs)
        total_fp += len(unmatched_pred_idxs)
        total_fn += len(unmatched_gt_idxs)

        save_path = save_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)

        print(
            f"[{img_path.name}] "
            f"Pred={len(preds)} "
            f"TP={len(matched_pairs)} "
            f"FP={len(unmatched_pred_idxs)} "
            f"FN={len(unmatched_gt_idxs)} -> {save_path}"
        )

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)

    print("\\n========== 汇总 ==========")
    print(f"CONF_THRES:      {CONF_THRES}")
    print(f"IOU_NMS_THRES:   {IOU_NMS_THRES}")
    print(f"IOU_MATCH_THRES: {IOU_MATCH_THRES}")
    print(f"MAX_DET:         {MAX_DET}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"结果保存目录: {save_dir}")


if __name__ == "__main__":
    main()
