from ultralytics import YOLO

# model = YOLO("runs/detect/results/yolov8s/weights/best.pt")
model = YOLO("../Models/runs/detect/yolov8s_best/weights/best.pt")

# test tất cả ảnh trong folder
# model.predict(
#     source="voc0712/images/val",
#     save=True
# )

# test bằng webcam
# model.predict(
#     source=1, #0 là cam ip, 1 là cam mac
#     show=True
# )

model.predict(
    source="anh_test/1.png",
    show=True,
    save=True
)