"""
=============================================================
Hanoi Landmark Dataset — Bước 3: Gán nhãn
=============================================================
Tạo các file nhãn chuẩn:
  • labels.csv      — ánh xạ filename → class
  • classes.txt     — danh sách class
  • metadata.json   — thông tin chi tiết từng ảnh
  • labels_imagenet.txt — định dạng ImageNet (cho torchvision)

Cách dùng:
    python step3_label.py
=============================================================
"""

import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image

CLEAN_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "clean_images"

LANDMARKS = {
    "ho_guom":      {"id": 0, "name_vi": "Hồ Gươm",         "name_en": "Hoan Kiem Lake"},
    "chua_mot_cot": {"id": 1, "name_vi": "Chùa Một Cột",    "name_en": "One Pillar Pagoda"},
    "lang_bac":     {"id": 2, "name_vi": "Lăng Bác",         "name_en": "Ho Chi Minh Mausoleum"},
    "ho_tay":      {"id": 3, "name_vi": "Hồ Tây",     "name_en": "West Lake"},
    "nha_hat_lon":          {"id": 4, "name_vi": "Nhà Hát Lớn",        "name_en": "Hanoi Opera House"},
    "nha_tho_lon":     {"id": 5, "name_vi": "Nhà Thờ Lớn",         "name_en": "St. Joseph's Cathedral Hanoi"},
    "trang_tien_plaza":       {"id": 6, "name_vi": "Tràng Tiền Plaza",    "name_en": "Trang Tien Plaza"},
    "van_mieu":       {"id": 7, "name_vi": "Văn Miếu Quốc Tử Giám",   "name_en": "Temple of Literature Hanoi"},
}


def get_image_meta(path: Path) -> dict:
    """Lấy thông tin cơ bản của ảnh."""
    try:
        img = Image.open(path)
        w, h = img.size
        mode = img.mode
    except Exception:
        w, h, mode = 0, 0, "unknown"

    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    return {
        "width": w,
        "height": h,
        "mode": mode,
        "size_bytes": path.stat().st_size,
        "md5": md5,
    }


def generate_labels():
    print("Đang tạo file nhãn...\n")

    records = []       # cho CSV + JSON
    imagenet_lines = []

    for cls, info in LANDMARKS.items():
        cls_dir = CLEAN_DIR / cls
        if not cls_dir.exists():
            print(f"  [{cls}] Chưa có thư mục clean, bỏ qua.")
            continue

        files = sorted(cls_dir.glob("*.jpg"))
        print(f"  [{cls}] {len(files)} ảnh")

        for f in files:
            meta = get_image_meta(f)
            records.append({
                "filename":  f.name,
                "class":     cls,
                "class_id":  info["id"],
                "name_vi":   info["name_vi"],
                "name_en":   info["name_en"],
                "width":     meta["width"],
                "height":    meta["height"],
                "md5":       meta["md5"],
                "size_bytes":meta["size_bytes"],
            })
            # ImageNet format: relative/path/to/img.jpg <class_id>
            imagenet_lines.append(f"{cls}/{f.name} {info['id']}")

    # ── 1. labels.csv ─────────────────────────────────────
    csv_path = Path("labels.csv")
    fieldnames = ["filename", "class", "class_id", "name_vi", "name_en",
                  "width", "height", "md5", "size_bytes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n✓ labels.csv        ({len(records)} dòng)")

    # ── 2. classes.txt ────────────────────────────────────
    classes_path = Path("classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls, info in LANDMARKS.items():
            f.write(f"{info['id']}\t{cls}\t{info['name_vi']}\t{info['name_en']}\n")
    print(f"✓ classes.txt       ({len(LANDMARKS)} class)")

    # ── 3. metadata.json ──────────────────────────────────
    meta_path = Path("metadata.json")
    output = {
        "dataset_name": "Hanoi Landmark Dataset",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "num_classes": len(LANDMARKS),
        "total_images": len(records),
        "classes": LANDMARKS,
        "images": records,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✓ metadata.json     ({len(records)} ảnh)")

    # ── 4. labels_imagenet.txt ────────────────────────────
    imagenet_path = Path("labels_imagenet.txt")
    with open(imagenet_path, "w", encoding="utf-8") as f:
        f.write("\n".join(imagenet_lines))
    print(f"✓ labels_imagenet.txt ({len(imagenet_lines)} dòng)")

    # ── 5. In thống kê ────────────────────────────────────
    print("\n" + "=" * 50)
    print("THỐNG KÊ — Bước 3: Gán nhãn")
    print("=" * 50)
    print(f"  {'Class':<20} {'ID':>3}  {'Số ảnh':>7}")
    print("-" * 50)
    from collections import Counter
    counts = Counter(r["class"] for r in records)
    for cls, info in LANDMARKS.items():
        n = counts.get(cls, 0)
        warn = "  ⚠ thiếu" if n < 599 else ""
        print(f"  {cls:<20} {info['id']:>3}  {n:>7}{warn}")
    print("=" * 50)
    print(f"\nBước tiếp theo: python step4_structure.py")


if __name__ == "__main__":
    generate_labels()