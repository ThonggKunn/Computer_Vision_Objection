"""
=============================================================
Vietnam Landmark Dataset — SETUP HOÀN CHỈNH
MacBook Pro M1 | 8GB RAM | venv | Ultralytics + PyTorch
=============================================================
Chạy file này một lần duy nhất để:
  1. Kiểm tra môi trường
  2. Cài thư viện còn thiếu
  3. Cấu hình đường dẫn
  4. Test icrawler với 5 ảnh
  5. In ra lệnh chạy từng bước
=============================================================
"""

import subprocess
import sys
import importlib
from pathlib import Path

from PIL.Image import Image

# ── Cấu hình project ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent   # ~/PycharmProjects/CV
DATA_DIR     = PROJECT_ROOT / "Data" / "vietnam_landmark"
TOOLS_DIR    = PROJECT_ROOT / "Dataset Tools" / "vietnam_landmark"

# ══════════════════════════════════════════════════════════
# BƯỚC 1 — Kiểm tra môi trường
# ══════════════════════════════════════════════════════════

def check_env():
    print("=" * 55)
    print("BƯỚC 1 — Kiểm tra môi trường")
    print("=" * 55)

    # Python version
    v = sys.version_info
    print(f"  Python : {v.major}.{v.minor}.{v.micro}", end="")
    if v.major == 3 and v.minor >= 9:
        print("  ✓")
    else:
        print("  ⚠ Nên dùng Python 3.9+")

    # PyTorch + MPS
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}  ✓")
        mps = torch.backends.mps.is_available()
        print(f"  MPS    : {'✓ Sẵn sàng dùng GPU M1' if mps else '✗ Không khả dụng — sẽ dùng CPU'}")
    except ImportError:
        print("  PyTorch: ✗ Chưa cài")

    # Ultralytics
    try:
        import ultralytics
        print(f"  YOLO   : ultralytics {ultralytics.__version__}  ✓")
    except ImportError:
        print("  YOLO   : ✗ Chưa cài")

    print()


# ══════════════════════════════════════════════════════════
# BƯỚC 2 — Cài thư viện còn thiếu
# ══════════════════════════════════════════════════════════

REQUIRED = {
    "icrawler":    "icrawler",
    "PIL":         "pillow",
    "cv2":         "opencv-python",
    "imagehash":   "imagehash",
    "tqdm":        "tqdm",
    "matplotlib":  "matplotlib",
    "albumentations": "albumentations",
}

def install_missing():
    print("=" * 55)
    print("BƯỚC 2 — Kiểm tra & cài thư viện còn thiếu")
    print("=" * 55)

    to_install = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
            print(f"  {package:<25} ✓ đã có")
        except ImportError:
            print(f"  {package:<25} ✗ chưa có → sẽ cài")
            to_install.append(package)

    if to_install:
        print(f"\n  Đang cài: {', '.join(to_install)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", *to_install
        ])
        print("  ✓ Cài xong!")
    else:
        print("\n  ✓ Tất cả đã có, không cần cài thêm.")
    print()


# ══════════════════════════════════════════════════════════
# BƯỚC 3 — Tạo cấu trúc thư mục
# ══════════════════════════════════════════════════════════

LANDMARKS = [
    "ha_long", "hue",
    "cau_vang", "hoi_an", "duc_ba", "ben_nha_rong", "ben_thanh",
]
# LANDMARKS = [
#     "ho_guom", "chua_mot_cot", "lang_bac", "van_mieu", "nha_tho_lon", "nha_hat_lon", "ho_tay", "trang_tien_plaza",
# ]

def create_dirs():
    print("=" * 55)
    print("BƯỚC 3 — Tạo cấu trúc thư mục")
    print("=" * 55)

    dirs = []
    # Data dirs
    for cls in LANDMARKS:
        dirs.append(DATA_DIR / "raw_images" / cls)
        dirs.append(DATA_DIR / "clean_images" / cls)
        dirs.append(DATA_DIR / "trash" / cls)
    for split in ("train", "val", "test"):
        for cls in LANDMARKS:
            dirs.append(DATA_DIR / "dataset" / split / cls)

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  ✓ Tạo {len(dirs)} thư mục tại:")
    print(f"    {DATA_DIR}")
    print()


# ══════════════════════════════════════════════════════════
# BƯỚC 4 — Tạo file config dùng chung cho toàn bộ pipeline
# ══════════════════════════════════════════════════════════

CONFIG_CONTENT = f'''"""
Config dùng chung cho toàn bộ Vietnam Landmark Pipeline.
Import file này ở đầu mỗi step:
    from config import *
"""
from pathlib import Path

PROJECT_ROOT = Path("{PROJECT_ROOT}")
DATA_DIR     = Path("{DATA_DIR}")

RAW_DIR      = DATA_DIR / "raw_images"
CLEAN_DIR    = DATA_DIR / "clean_images"
TRASH_DIR    = DATA_DIR / "trash"
DATASET_DIR  = DATA_DIR / "dataset"

TARGET_PER_CLASS = 1000
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO        = 0.15
TARGET_SIZE      = (224, 224)
SEED             = 42

LANDMARKS = {{
    
    "ha_long":      {{"id": 3, "name_vi": "V\\u1ecbnh H\\u1ea1 Long",        "name_en": "Ha Long Bay"}},
    "hue":          {{"id": 4, "name_vi": "C\\u1ed1 \\u0111\\u00f4 Hu\\u1ebf","name_en": "Hue Imperial City"}},
    "cau_vang":     {{"id": 5, "name_vi": "C\\u1ea7u V\\u00e0ng",            "name_en": "Golden Bridge"}},
    "hoi_an":       {{"id": 6, "name_vi": "Ph\\u1ed1 c\\u1ed5 H\\u1ed9i An", "name_en": "Hoi An Ancient Town"}},
    "duc_ba":       {{"id": 7, "name_vi": "Nh\\u00e0 th\\u1edd \\u0110\\u1ee9c B\\u00e0", "name_en": "Notre Dame Cathedral"}},
    "ben_nha_rong": {{"id": 8, "name_vi": "B\\u1ebfn Nh\\u00e0 R\\u1ed3ng",  "name_en": "Nha Rong Wharf"}},
    "ben_thanh":    {{"id": 9, "name_vi": "Ch\\u1ee3 B\\u1ebfn Th\\u00e0nh", "name_en": "Ben Thanh Market"}},
}}

# icrawler search queries mỗi class
QUERIES = {{
    "ha_long":      ["Ha Long Bay Vietnam", "Halong Bay karst", "Vinh Ha Long"],
    "hue":          ["Hue Imperial City", "Hue Citadel Vietnam", "Co do Hue"],
    "cau_vang":     ["Golden Bridge Da Nang", "Cau Vang Ba Na Hills"],
    "hoi_an":       ["Hoi An Ancient Town", "Hoi An lanterns", "Hoi An old town"],
    "duc_ba":       ["Notre Dame Cathedral Saigon", "Duc Ba church Ho Chi Minh"],
    "ben_nha_rong": ["Ben Nha Rong Saigon", "Dragon House Wharf Ho Chi Minh"],
    "ben_thanh":    ["Ben Thanh Market Saigon", "Cho Ben Thanh Ho Chi Minh"],
}}
'''

def create_config():
    print("=" * 55)
    print("BƯỚC 4 — Tạo file config.py dùng chung")
    print("=" * 55)

    config_path = TOOLS_DIR / "config.py"
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(CONFIG_CONTENT, encoding="utf-8")
    print(f"  ✓ Tạo: {config_path}")
    print()


# # ══════════════════════════════════════════════════════════
# # BƯỚC 5 — Test icrawler với 3 ảnh
# # ══════════════════════════════════════════════════════════
#
# def test_icrawler():
#     print("=" * 55)
#     print("BƯỚC 5 — Test icrawler (crawl thử 3 ảnh Hồ Gươm)")
#     print("=" * 55)
#
#     try:
#         from icrawler.builtin import BingImageCrawler
#         from PIL import Image
#         Image.MAX_IMAGE_PIXELS = None
#
#         test_dir = DATA_DIR / "raw_images" / "ho_guom"
#         test_dir.mkdir(parents=True, exist_ok=True)
#
#         crawler = BingImageCrawler(
#             storage={"root_dir": str(test_dir)},
#             log_level=50,
#         )
#         crawler.crawl(keyword="Ho Guom Hanoi lake", max_num=3)
#
#         count = len(list(test_dir.glob("*.jpg")))
#         if count > 0:
#             print(f"  ✓ icrawler hoạt động! Đã tải {count} ảnh vào:")
#             print(f"    {test_dir}")
#         else:
#             print("  ⚠ Không tải được ảnh — kiểm tra kết nối mạng")
#     except Exception as e:
#         print(f"  ✗ Lỗi: {e}")
#     print()


# ══════════════════════════════════════════════════════════
# BƯỚC 6 — In hướng dẫn chạy từng bước
# ══════════════════════════════════════════════════════════

def print_next_steps():
    tools = TOOLS_DIR
    print("=" * 55)
    print("XONG! Hướng dẫn chạy từng bước tiếp theo")
    print("=" * 55)
    steps = [
        ("Thu thập ảnh",        f"python '{tools}/step1_collect.py'",   "~2–4 giờ, 10.000 ảnh"),
        ("Làm sạch dữ liệu",    f"python '{tools}/step2_clean.py'",     "~20–30 phút"),
        ("Gán nhãn",            f"python '{tools}/step3_label.py'",     "~5 phút"),
        ("Chuẩn hóa cấu trúc",  f"python '{tools}/step4_structure.py'", "~5 phút"),
        ("Data augmentation",   f"python '{tools}/step5_augment.py'",   "~15–30 phút"),
        ("Kiểm tra chất lượng", f"python '{tools}/step6_verify.py --plot'", "~10 phút"),
        ("Train model",         f"python '{PROJECT_ROOT}/Core Scripts/train_landmark.py'", "~1–2 giờ"),
    ]
    for i, (name, cmd, est) in enumerate(steps, 1):
        print(f"\n  Bước {i} — {name}  ({est})")
        print(f"  $ {cmd}")

    print("\n" + "=" * 55)
    print("  Mẹo với M1 8GB:")
    print("  • Đóng Chrome + app khác trước khi train")
    print("  • Nếu crash → giảm batch=4 trong train_landmark.py")
    print("  • Dùng device='mps' để tận dụng GPU M1")
    print("=" * 55)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Vietnam Landmark Dataset — Setup M1 Mac\n")
    check_env()
    install_missing()
    create_dirs()
    create_config()
    # test_icrawler()
    print_next_steps()