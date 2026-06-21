import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    model = YOLO(r"D:\code\ultralytics-SAR-V2\ultralytics\cfg\models\v8\yolov8s.yaml")
    results = model.train(
        data="RTS_data.yaml",
        imgsz=640,
        batch=16,
        epochs=300,
        device=0,
        pretrained=False,
        save_period=20,  # 每10轮保存一次权重
    )
