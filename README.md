# Computer Vision — AI Hỗ Trợ Người Khiếm Thị

> Ứng dụng thị giác máy tính sử dụng YOLOv8 để phát hiện vật cản và cảnh báo bằng giọng nói tiếng Việt, hỗ trợ người khiếm thị di chuyển an toàn và trải nghiệm du lịch độc lập.

---

## Tính năng chính

- **Phát hiện vật cản** — nhận diện người, xe, ghế, cửa... và ước lượng khoảng cách
- **Nhận diện địa danh** - nhận diện các địa danh nổi tiếng ở Hà Nội: Lăng Bác, Hồ Gươm, Hồ Tây,...
- **Xác định vị trí** — thông báo vật thể đang ở bên trái, phía trước, hay bên phải
- **Cảnh báo giọng nói** — đọc cảnh báo bằng tiếng Việt tự nhiên (gTTS)
- **Hai chế độ** — Nhận diện vật cản & Nhận diện địa danh Hà Nội
- **Tự động đọc** — tự động thông báo khi phát hiện vật thể mới

---

## Cấu trúc project

```
CV/
├── Core Scripts/
│   ├── main.py               ← Chạy app chính (tích hợp tất cả)
│   ├── seg_yolov8s.py        ← Segmentation đơn lẻ
│   ├── test_yolov8s.py       ← Test model với ảnh/webcam
│   ├── text_to_speech.py     ← Demo TTS
│   ├── train_models.py       ← Train model chung
│   └── train_yolov8s.py      ← Train YOLOv8 detection
│
├── Dataset Tools/
│   ├── dataset.yaml          ← Cấu hình dataset YOLO
│   ├── merge_voc.py          ← Gộp VOC2007 + VOC2012
│   └── prepare_dataset.py    ← Chuẩn bị dataset
│
├── Evaluation/
│   ├── compare_model.py      ← So sánh các model
│   ├── evaluate_models.py    ← Đánh giá độ chính xác
│   ├── mAP_learning_curve.py ← Vẽ learning curve mAP
│   └── plot_learning_curve.py← Vẽ biểu đồ quá trình train
│
├── Models/                   ← Chứa file .pt (không push lên git)
├── Data/                     ← Chứa dataset ảnh (không push lên git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cài đặt

### Yêu cầu
- Python 3.10+
- Webcam hoặc camera IP (DroidCam...)

### Bước 1 — Clone repo
```bash
git clone https://github.com/ThonggKunn/Computer_Vision_Objection.git
cd Computer_Vision_Objection
```

### Bước 2 — Tạo môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3 — Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 4 — Tải model weights
Tải các file model và đặt vào thư mục `Models/`:

| File | Dùng cho |
|------|----------|
| `yolov8s.pt` | Detection |
| `yolov8s-seg.pt` | Segmentation |
| `yolov8s-seg.pt` | Segmentation |

Tải tại: https://github.com/ultralytics/ultralytics

---

## Chạy ứng dụng

```bash
cd "Core Scripts"
python main.py
```

### Phím tắt

| Phím | Chức năng |
|------|-----------|
| `SPACE` | Đọc cảnh báo ngay lập tức |
| `M` | Đổi chế độ Segmentation ↔ Detection |
| `A` | Bật / tắt tự động đọc |
| `ESC` | Thoát |

### Đổi nguồn camera
Mở `main.py`, chỉnh dòng `CAMERA_SOURCE`:
```python
CAMERA_SOURCE = 0                                    # Webcam Mac
CAMERA_SOURCE = 1                                    # Camera ngoài
CAMERA_SOURCE = "http://192.168.1.4:4747/video"     # DroidCam IP
```

---

## Train model

### Train YOLOv8 Detection (VOC dataset)
```bash
# Bước 1 — Chuẩn bị dataset
python "Dataset Tools/merge_voc.py"
python "Dataset Tools/prepare_dataset.py"

# Bước 2 — Train
python "Core Scripts/train_yolov8s.py"
```



---

## Đánh giá model

```bash
python Evaluation/evaluate_models.py
python Evaluation/compare_model.py
python Evaluation/plot_learning_curve.py
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Segmentation | YOLOv8-seg |
| Text To Speech | gTTS (Google TTS) |
| Xử lý ảnh | OpenCV, Pillow |
| Deep Learning | PyTorch (MPS — Apple M1) |
| Dataset crawl | icrawler |

---

## Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | CPU (chậm) | Apple M1/M2 hoặc NVIDIA |
| Camera | Webcam 720p | 1080p |
| Python | 3.10 | 3.11+ |

---

## Tác giả

**ThonggKunn** — [GitHub](https://github.com/ThonggKunn)

---

## License

MIT License
