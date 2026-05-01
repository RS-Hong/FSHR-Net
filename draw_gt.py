import cv2
import os
from pathlib import Path


def find_image_files(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def read_image_any(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def draw_yolo_boxes_on_image(img, label_path, color=(0, 255, 0), thickness=2, show_class=False):
    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x_center = xc * w
        y_center = yc * h
        box_w = bw * w
        box_h = bh * h

        x1 = int(round(x_center - box_w / 2))
        y1 = int(round(y_center - box_h / 2))
        x2 = int(round(x_center + box_w / 2))
        y2 = int(round(y_center + box_h / 2))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if show_class:
            cv2.putText(
                img,
                str(cls_id),
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return img


def draw_labels_from_dataset_root(
    dataset_root,
    save_root,
    color=(0, 255, 0),
    thickness=2,
    show_class=False,
    keep_subdir=False,
):
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    save_root = Path(save_root)

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到 images 文件夹: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"未找到 labels 文件夹: {labels_dir}")

    save_root.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(images_dir)
    if not image_files:
        print(f"images 文件夹下没有找到图片: {images_dir}")
        return

    total = 0
    missing_label = 0

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        try:
            img = read_image_any(image_path)
            img = draw_yolo_boxes_on_image(
                img=img,
                label_path=label_path,
                color=color,
                thickness=thickness,
                show_class=show_class,
            )

            if keep_subdir:
                out_path = save_root / image_path.name
            else:
                out_path = save_root / f"{image_path.stem}_gt_green{image_path.suffix}"

            cv2.imwrite(str(out_path), img)

            total += 1
            if not label_path.exists():
                missing_label += 1
                print(f"[无标签] {image_path.name}")
            else:
                print(f"[已保存] {out_path}")

        except Exception as e:
            print(f"[失败] {image_path.name}: {e}")

    print("=" * 60)
    print(f"处理完成，总图片数: {total}")
    print(f"缺少标签数: {missing_label}")
    print(f"保存目录: {save_root}")


if __name__ == "__main__":
    dataset_root = r"D:\code\ultralytics-SAR-V2\images\SSDD"
    save_root = r"D:\code\ultralytics-SAR-V2\optimg\GTSSDD"

    draw_labels_from_dataset_root(
        dataset_root=dataset_root,
        save_root=save_root,
        color=(0, 255, 0),   # 绿色
        thickness=2,
        show_class=False,
        keep_subdir=False,
    )