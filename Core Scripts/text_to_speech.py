from ultralytics import YOLO
import cv2
from gtts import gTTS
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===== TEXT TO SPEECH =====
def speak_vi(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("voice.mp3")
    os.system("afplay voice.mp3")
    os.remove("voice.mp3")

# ===== VẼ TEXT TIẾNG VIỆT =====
def draw_text_vi(frame, text, position=(20, 40)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # để file font cùng thư mục
    font = ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        30
    )

    draw.text(position, text, font=font, fill=(0, 255, 0))

    return np.array(img_pil)

# ===== LOAD MODEL SEGMENTATION =====
model = YOLO("../Models/yolov8s-seg.pt")

# ===== MAP LABEL TIẾNG VIỆT =====
label_map = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "xe hơi",
    "motorcycle": "xe máy",
    "airplane": "máy bay",
    "bus": "xe buýt",
    "train": "tàu hỏa",
    "truck": "xe tải",
    "boat": "thuyền",
    "traffic light": "đèn giao thông",
    "fire hydrant": "trụ nước",
    "stop sign": "biển dừng",
    "parking meter": "đồng hồ đỗ xe",
    "bench": "ghế dài",
    "bird": "chim",
    "cat": "mèo",
    "dog": "chó",
    "horse": "ngựa",
    "sheep": "cừu",
    "cow": "bò",
    "elephant": "voi",
    "bear": "gấu",
    "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ",
    "backpack": "ba lô",
    "umbrella": "ô",
    "handbag": "túi xách",
    "tie": "cà vạt",
    "suitcase": "vali",
    "frisbee": "đĩa bay",
    "skis": "ván trượt tuyết",
    "snowboard": "ván tuyết",
    "sports ball": "bóng",
    "kite": "diều",
    "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày",
    "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis",
    "bottle": "chai nước",
    "wine glass": "ly rượu",
    "cup": "cốc",
    "fork": "nĩa",
    "knife": "dao",
    "spoon": "muỗng",
    "bowl": "bát",
    "banana": "chuối",
    "apple": "táo",
    "sandwich": "bánh mì kẹp",
    "orange": "cam",
    "broccoli": "bông cải",
    "carrot": "cà rốt",
    "hot dog": "xúc xích",
    "pizza": "pizza",
    "donut": "bánh donut",
    "cake": "bánh",
    "chair": "ghế",
    "couch": "sofa",
    "potted plant": "cây cảnh",
    "bed": "giường",
    "dining table": "bàn ăn",
    "toilet": "bồn cầu",
    "tv": "tivi",
    "laptop": "máy tính",
    "mouse": "chuột",
    "remote": "điều khiển",
    "keyboard": "bàn phím",
    "cell phone": "điện thoại",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "book": "sách",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "scissors": "kéo",
    "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc",
    "toothbrush": "bàn chải"
}

# ===== CAMERA =====
cap = cv2.VideoCapture("http://192.168.1.4:4747/video")
# cap = cv2.VideoCapture(0)
last_text = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    results = model(frame)

    biggest_obj = None
    max_area = 0

    # ===== DUYỆT OBJECT =====
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                biggest_obj = box

    # ===== XỬ LÝ OBJECT =====
    if biggest_obj is not None:
        cls_id = int(biggest_obj.cls[0])
        label = model.names[cls_id]
        label_vi = label_map.get(label, label)

        x1, y1, x2, y2 = biggest_obj.xyxy[0]
        x_center = (x1 + x2) / 2

        # vị trí
        if x_center < w / 3:
            position = "bên trái"
        elif x_center < 2 * w / 3:
            position = "phía trước"
        else:
            position = "bên phải"

        # khoảng cách
        if max_area > 80000:
            distance = "rất gần"
            distance_m = "0.5m"
        elif max_area > 50000:
            distance = "gần"
            distance_m = "1m"
        elif max_area > 30000:
            distance = "xa"
            distance_m = "2m"
        else:
            distance = "rất xa"
            distance_m = "3m+"

        text = f"Có {label_vi} {position}, {distance}, cách khoảng {distance_m}"

    else:
        text = "Không có vật thể"

    # ===== HIỂN THỊ =====
    annotated = results[0].plot()

    #  FIX FONT TIẾNG VIỆT
    annotated = draw_text_vi(annotated, text)

    cv2.imshow("Segmentation AI", annotated)

    key = cv2.waitKey(1)

    # ===== SPACE để đọc =====
    if key == 32:
        if text != last_text:
            print("Voice:", text)
            speak_vi(text)
            last_text = text

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()