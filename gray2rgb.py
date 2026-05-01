'''
RT-DETR仅支持3通道，因此将SAR图像灰度图转3通道图像训练、预测
'''
from PIL import Image
from pathlib import Path

src_dir = Path("YOLO_SARship/test/images")
dst_dir = Path("YOLO_SARship3/test/images")
dst_dir.mkdir(parents=True, exist_ok=True)

for p in src_dir.glob("*.*"):
    img = Image.open(p).convert("RGB")
    img.save(dst_dir / p.name)