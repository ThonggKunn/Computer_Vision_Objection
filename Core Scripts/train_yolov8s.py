from ultralytics import YOLO

model = YOLO("../Models/yolov8s.pt")

model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    save_period=10,  # save mỗi 10 epochs
    # device="mps",
    workers=2,
    name="yolov8s_best"
)