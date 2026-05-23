from ultralytics import YOLO

models = [
    # "runs/detect/results/yolov8s/weights/best.pt",
    "runs/detect/yolov8s_best/weights/best.pt"
]

for m in models:

    model = YOLO(m)

    metrics = model.val(
        data="dataset.yaml",
        split="val"
    )

    print(m)
    print(metrics.box.map)