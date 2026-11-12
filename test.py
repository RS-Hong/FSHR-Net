
from ultralytics import YOLO
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_named_modules_text(model, tag="model"):
    lines = []
    lines.append(f"==== {tag} named_modules ====\n")
    idx = 0
    for name, module in getattr(model, "model", model).named_modules():

        if name.startswith("model.") and name.count(".") == 1:
            lines.append(f"[{idx:04d}] {name}    class={module.__class__.__name__}\n")
            idx += 1
    return "".join(lines)


def print_model_report(model, tag="model", save_txt_path=None):
    modules_text = get_named_modules_text(model, tag)
    print(modules_text)

    if save_txt_path is not None:
        Path(save_txt_path).write_text(modules_text, encoding="utf-8")

if __name__ == '__main__':

    model = YOLO(r"G:\对比权重\SSDD\RT-DETR.pt")



    print_model_report(
        model,
        tag="test_model",
    )


    results = model.val(
        data="RTS_data.yaml",
        batch=16,
        imgsz=640,
        save_json=False,
        save_txt=False,
        plots=True,
        conf=0.001,
        iou=0.6,
        device=0,
        verbose=True,
        save_period = 10,
    )


    print("mAP@0.5:", results.box.map50)
    print("mAP@0.75:", results.box.map75)
    print("mAP@0.5:0.95:", results.box.map)
    print("Precision:", results.box.mp)
    print("Recall:", results.box.mr)

















    for i, name in enumerate(model.names):
        print(f"类别 {name} 的mAP@0.5: {results.box.map50[i]}")