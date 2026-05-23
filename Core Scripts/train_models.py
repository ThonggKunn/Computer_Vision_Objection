from ultralytics import YOLO
import os

models = [
"yolov8n.pt",
"yolov8s.pt",
"yolov8m.pt"
]

for model_name in models:

    print("Training:", model_name)

    model = YOLO(model_name)

    results = model.train(
        data="dataset.yaml",
        epochs=30,
        imgsz=480,
        batch=16,
        save_period=10,  # save mỗi 10 epochs
        project="results",
        name=model_name.split(".")[0],
        device= "mps"
    )