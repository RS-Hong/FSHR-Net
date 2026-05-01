from ultralytics import YOLO
import ultralytics
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(r"D:\code\ultralytics-SAR-V2\ultralytics\cfg\models\v8\yolov8s.yaml", verbose=True)
model.model.to(device)   # 这句必须有

x = torch.randn(1, 3, 640, 640, device=device)
_ = model.model.predict(x, profile=True)