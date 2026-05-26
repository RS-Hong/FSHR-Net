import random
import shutil
from pathlib import Path


def split_dataset(
    dataset_root,
    output_root,
    seed=42,
    train_ratio=0.5,
    val_ratio=0.3,
    test_ratio=0.2,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    random.seed(seed)

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    assert images_dir.exists(), f"{images_dir} not found"
    assert labels_dir.exists(), f"{labels_dir} not found"

    # 读取 image 文件名（不含后缀）
    image_files = list(images_dir.iterdir())
    image_stems = [f.stem for f in image_files]

    # 打乱
    random.shuffle(image_stems)

    total = len(image_stems)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_ids = image_stems[:n_train]
    val_ids = image_stems[n_train : n_train + n_val]
    test_ids = image_stems[n_train + n_val :]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split, ids in splits.items():
        img_out = output_root / split / "images"
        lbl_out = output_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in ids:
            # image
            img_file = next(images_dir.glob(stem + ".*"), None)
            lbl_file = next(labels_dir.glob(stem + ".*"), None)

            if img_file is None or lbl_file is None:
                raise FileNotFoundError(f"Missing image or label for: {stem}")

            shutil.copy(img_file, img_out / img_file.name)
            shutil.copy(lbl_file, lbl_out / lbl_file.name)

    print("Dataset split completed.")
    print(f"Total: {total}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


if __name__ == "__main__":
    dataset_root = r"D:\code\ultralytics-SAR-V2\SAR_ship_dataset"  # ← 修改这里
    output_root = r"D:\code\ultralytics-SAR-V2\SAR_ship_dataset\split_dataset"  # ← 修改这里
    seed = 42  # ← 你的随机种子

    split_dataset(dataset_root=dataset_root, output_root=output_root, seed=seed)
