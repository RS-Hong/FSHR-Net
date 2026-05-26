# 导入必要的库
import os
from pathlib import Path

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_named_modules_text(model, tag="model"):
    lines = []
    lines.append(f"==== {tag} named_modules ====\n")
    idx = 0
    for name, module in getattr(model, "model", model).named_modules():
        # 只保留一级层，例如 model.13    class=SCRM
        if name.startswith("model.") and name.count(".") == 1:
            lines.append(f"[{idx:04d}] {name}    class={module.__class__.__name__}\n")
            idx += 1
    return "".join(lines)


def print_model_report(model, tag="model", save_txt_path=None):
    modules_text = get_named_modules_text(model, tag)
    print(modules_text)

    if save_txt_path is not None:
        Path(save_txt_path).write_text(modules_text, encoding="utf-8")


if __name__ == "__main__":
    # 1. 加载训练好的权重文件（替换为你的权重路径）
    model = YOLO(r"G:\对比权重\SSDD\RT-DETR.pt")  # 训练后最优权重
    # 也可以用 last.pt（最后一轮权重）：model = YOLO("runs/detect/train/weights/last.pt")

    # -------------------------------------#
    print_model_report(
        model,
        tag="test_model",
    )

    # 2. 执行评估，关键参数可根据需求调整
    results = model.val(
        data="RTS_data.yaml",  # 必须：数据集配置文件路径（与训练时一致）
        batch=16,  # 批次大小，根据显卡显存调整（显存小则调小，如8、4）
        imgsz=640,  # 评估时的输入图片尺寸，与训练时一致即可
        save_json=False,  # 是否保存评估结果为 JSON 文件（用于COCO格式评估）
        save_txt=False,  # 是否保存预测结果为 TXT 文件
        plots=True,  # 是否生成评估可视化图表（如PR曲线、混淆矩阵）
        conf=0.001,  # 置信度阈值（默认即可，不建议随意修改）
        iou=0.6,  # NMS的IOU阈值（默认0.6）
        device=0,  # 使用的GPU编号（CPU则填"cpu"，多GPU填[0,1]）
        verbose=True,  # 是否打印详细评估日志
        save_period=10,  # 每10轮保存一次权重
    )

    # 3. 提取关键评估指标（可选，用于后续分析）
    print("mAP@0.5:", results.box.map50)  # 所有类别的mAP@0.5（核心指标）
    print("mAP@0.75:", results.box.map75)  # 所有类别的mAP@0.75（核心指标）
    print("mAP@0.5:0.95:", results.box.map)  # 所有类别的mAP@0.5:0.95（综合指标）
    print("Precision:", results.box.mp)  # 平均精确率
    print("Recall:", results.box.mr)  # 平均召回率

    # # 2. 读取速度（单位：ms）
    # speed = results.speed
    # preprocess = speed["preprocess"]
    # inference = speed["inference"]
    # postprocess = speed["postprocess"]
    # total_ms = preprocess + inference + postprocess
    #
    # # 3. 计算 FPS
    # fps = 1000 / total_ms
    # fps_infer = 1000 / inference
    #
    # print(f"总时间: {total_ms:.2f} ms/frame")
    # print(f"整体 FPS: {fps:.2f}")
    # print(f"纯推理 FPS: {fps_infer:.2f}")

    # 按类别查看指标（如果有多个类别）
    for i, name in enumerate(model.names):
        print(f"类别 {name} 的mAP@0.5: {results.box.map50[i]}")
