from ultralytics import YOLO
import cv2

# model segmentation
model = YOLO("../Models/yolov8s-seg.pt")

#test tất cả ảnh trong file

# model.predict(
#     source="voc0712/images/test",
#     save=True
# )

# test bằng webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # vẽ mask + box
    annotated = results[0].plot()

    cv2.imshow("Segmentation", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()