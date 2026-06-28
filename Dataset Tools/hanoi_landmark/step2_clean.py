"""
=============================================================
Vietnam Landmark Dataset — Bước 2: Làm sạch dữ liệu
=============================================================
Tự động xóa:
  ✗ Ảnh trùng (hash-based + perceptual hash)
  ✗ Ảnh mờ (Laplacian variance)
  ✗ Ảnh quá nhỏ
  ✗ Ảnh có watermark (phát hiện text dày đặc)
  ✗ Ảnh AI-generated (EXIF + metadata check)
  ✗ File hỏng

Cài đặt:
    pip install pillow imagehash opencv-python tqdm
    pip install torch torchvision   # nếu dùng model detect ảnh AI

Cách dùng:
    python step2_clean.py
    python step2_clean.py --dry-run    # xem kết quả mà không xóa
=============================================================
"""

import os
import cv2
import shutil
import argparse
import numpy as np
import imagehash
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
from collections import defaultdict

# ── Cấu hình ──────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "raw_images"
CLEAN_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "clean_images"
TRASH_DIR   = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "trash"

MIN_WIDTH   = 200                     # pixel tối thiểu
MIN_HEIGHT  = 200
BLUR_THRESH = 80.0                    # Laplacian variance — thấp hơn = mờ hơn
HASH_THRESH = 8                       # perceptual hash distance (0=giống hệt)


#Landmark các địa điểm ở VN (làm sau)
# LANDMARKS = [
#     "ho_guom", "chua_mot_cot", "lang_bac", "ha_long", "hue",
#     "cau_vang", "hoi_an", "duc_ba", "ben_nha_rong", "ben_thanh",
# ]

LANDMARKS = [
    "ho_guom", "chua_mot_cot", "lang_bac", "van_mieu", "nha_tho_lon", "nha_hat_lon", "ho_tay", "trang_tien_plaza",
]

# Tên AI generator thường xuất hiện trong EXIF/metadata
AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "firefly",
    "dreamstudio", "novelai", "automatic1111", "comfyui",
]

# ── Tiện ích ──────────────────────────────────────────────
def _normalize_filenames(cls: str):
    """Đổi tên file về dạng cls_XXXX.jpg — bỏ tên tiếng Việt dài."""
    d     = RAW_DIR / cls
    files = sorted(f for f in d.glob("*")
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for i, f in enumerate(files):
        new_name = d / f"{cls}_{i:04d}.jpg"
        if f == new_name:
            continue
        if new_name.exists():
            new_name = d / f"{cls}_{i + 9000:04d}.jpg"
        try:
            f.rename(new_name)
        except Exception:
            pass

def clean_class(cls: str, dry_run: bool) -> dict:
    src_dir = RAW_DIR / cls

    # Đổi tên file có tên lạ về dạng chuẩn trước khi xử lý
    if not dry_run:
        _normalize_filenames(cls)

    files = sorted(src_dir.glob("*.jpg"))
    ...

def make_dirs(dry_run: bool):
    if not dry_run:
        for cls in LANDMARKS:
            (CLEAN_DIR / cls).mkdir(parents=True, exist_ok=True)
            (TRASH_DIR / cls).mkdir(parents=True, exist_ok=True)


def move_to_trash(src: Path, cls: str, reason: str, dry_run: bool):
    if not dry_run:
        # Giới hạn tên file tối đa 50 ký tự, bỏ ký tự đặc biệt
        safe_name = src.name[:50].encode("ascii", "ignore").decode("ascii")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = src.stem[:20]

        # Giới hạn reason tối đa 30 ký tự
        safe_reason = reason[:30].replace("/", "_").replace(":", "_")

        dst = TRASH_DIR / cls / f"[{safe_reason}]_{safe_name}.jpg"
        try:
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"  [WARN] Không move được {src.name[:30]}: {e}")
            # Xóa luôn nếu không move được
            try:
                src.unlink()
            except Exception:
                pass

def copy_to_clean(src: Path, cls: str, dry_run: bool):
    if not dry_run:
        dst = CLEAN_DIR / cls / src.name
        shutil.copy2(str(src), str(dst))


# ── Kiểm tra ảnh ─────────────────────────────────────────

def is_valid_image(path: Path) -> tuple[bool, str]:
    """Kiểm tra file có mở được không."""
    try:
        img = Image.open(path)
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"corrupt: {e}"


def is_too_small(path: Path) -> tuple[bool, str]:
    try:
        img = Image.open(path)
        w, h = img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return True, f"too_small_{w}x{h}"
    except Exception:
        pass
    return False, ""


def is_blurry(path: Path) -> tuple[bool, str]:
    """Dùng Laplacian variance để phát hiện ảnh mờ."""
    try:
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            return False, ""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < BLUR_THRESH:
            return True, f"blurry_{variance:.1f}"
    except Exception:
        pass
    return False, ""


def is_ai_generated(path: Path) -> tuple[bool, str]:
    """
    Kiểm tra EXIF metadata để phát hiện ảnh AI.
    Không phải tất cả ảnh AI đều có thể phát hiện cách này,
    nhưng đây là heuristic tốt nhất không cần model riêng.
    """
    try:
        img = Image.open(path)
        # Kiểm tra EXIF
        exif_data = img._getexif() or {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, "")
            if isinstance(value, str):
                val_lower = value.lower()
                for kw in AI_SOFTWARE_KEYWORDS:
                    if kw in val_lower:
                        return True, f"ai_{kw}"
        # Kiểm tra PNG metadata
        if img.format == "PNG":
            meta = img.info or {}
            meta_str = str(meta).lower()
            for kw in AI_SOFTWARE_KEYWORDS:
                if kw in meta_str:
                    return True, f"ai_{kw}"
        # Kiểm tra không có EXIF gì cả (heuristic nhẹ — không xóa tự động)
        # nếu muốn chặt hơn có thể bật dòng dưới:
        # if not exif_data and img.format == "JPEG":
        #     return True, "no_exif"
    except Exception:
        pass
    return False, ""


def has_watermark_heuristic(path: Path) -> tuple[bool, str]:
    """
    Phát hiện watermark đơn giản: nếu góc ảnh có vùng text-like
    (độ tương phản cao theo pattern ngang) thì cảnh báo.
    Đây là heuristic — không chính xác 100%.
    """
    try:
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            return False, ""
        h, w = img_cv.shape[:2]
        # Kiểm tra 4 góc (15% kích thước)
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        corners = [
            img_cv[:margin_h, :margin_w],           # trên-trái
            img_cv[:margin_h, w - margin_w:],        # trên-phải
            img_cv[h - margin_h:, :margin_w],        # dưới-trái
            img_cv[h - margin_h:, w - margin_w:],    # dưới-phải
        ]
        for corner in corners:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = edges.sum() / (255 * edges.size)
            if edge_ratio > 0.12:   # nhiều cạnh = khả năng có text/logo
                return True, "watermark_suspected"
    except Exception:
        pass
    return False, ""


# ── Phát hiện trùng lặp ───────────────────────────────────

def build_hash_index(cls: str) -> dict:
    """Tính perceptual hash cho tất cả ảnh trong 1 class."""
    index = {}
    src_dir = RAW_DIR / cls
    for f in sorted(src_dir.glob("*.jpg")):
        try:
            img = Image.open(f)
            h = imagehash.phash(img)
            index[f] = h
        except Exception:
            pass
    return index


def find_duplicates(hash_index: dict) -> set:
    """Tìm ảnh trùng — giữ lại 1, đánh dấu phần còn lại."""
    files = list(hash_index.keys())
    duplicates = set()
    for i in range(len(files)):
        if files[i] in duplicates:
            continue
        for j in range(i + 1, len(files)):
            if files[j] in duplicates:
                continue
            dist = hash_index[files[i]] - hash_index[files[j]]
            if dist <= HASH_THRESH:
                duplicates.add(files[j])
    return duplicates


# ── Pipeline chính ────────────────────────────────────────

def clean_class(cls: str, dry_run: bool) -> dict:
    src_dir = RAW_DIR / cls
    files = sorted(src_dir.glob("*.jpg"))
    stats = defaultdict(int)
    stats["total"] = len(files)

    print(f"\n  [{cls}] {len(files)} ảnh")

    # 1. Tìm ảnh trùng
    print(f"    Tính perceptual hash...")
    hash_index = build_hash_index(cls)
    duplicates = find_duplicates(hash_index)
    stats["duplicate"] = len(duplicates)

    # 2. Kiểm tra từng ảnh
    kept = []
    for f in tqdm(files, desc=f"    Kiểm tra", unit="img", leave=False):
        # Trùng lặp
        if f in duplicates:
            move_to_trash(f, cls, "duplicate", dry_run)
            continue

        # File hỏng
        ok, reason = is_valid_image(f)
        if not ok:
            stats["corrupt"] += 1
            move_to_trash(f, cls, reason, dry_run)
            continue

        # Quá nhỏ
        small, reason = is_too_small(f)
        if small:
            stats["too_small"] += 1
            move_to_trash(f, cls, reason, dry_run)
            continue

        # Mờ
        blurry, reason = is_blurry(f)
        if blurry:
            stats["blurry"] += 1
            move_to_trash(f, cls, reason, dry_run)
            continue

        # AI-generated
        ai, reason = is_ai_generated(f)
        if ai:
            stats["ai_generated"] += 1
            move_to_trash(f, cls, reason, dry_run)
            continue

        # Watermark (chỉ cảnh báo, không xóa tự động)
        wm, reason = has_watermark_heuristic(f)
        if wm:
            stats["watermark_suspected"] += 1
            # Để review thủ công — copy vào trash nhưng vẫn giữ bản gốc
            if not dry_run:
                dst = TRASH_DIR / cls / f"[review_watermark]_{f.name}"
                shutil.copy2(str(f), str(dst))

        # Ảnh hợp lệ → copy sang clean_images
        copy_to_clean(f, cls, dry_run)
        kept.append(f)

    stats["kept"] = len(kept)
    stats["removed"] = stats["total"] - len(kept)
    return stats


def print_report(all_stats: dict):
    print("\n" + "=" * 65)
    print("BÁO CÁO — Bước 2: Làm sạch dữ liệu")
    print("=" * 65)
    fmt = "  {:<20} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}"
    print(fmt.format("Class", "Total", "Kept", "Dup", "Blur", "Small", "AI", "WM?"))
    print("-" * 65)
    total_kept = 0
    for cls, s in all_stats.items():
        print(fmt.format(
            cls, s["total"], s["kept"],
            s["duplicate"], s["blurry"],
            s["too_small"], s["ai_generated"],
            s["watermark_suspected"],
        ))
        total_kept += s["kept"]
    print("=" * 65)
    print(f"  Tổng ảnh sạch: {total_kept}")
    print(f"\n  Ảnh bị loại → xem tại: {TRASH_DIR}/")
    print(f"  Ảnh sạch     → tại:    {CLEAN_DIR}/")
    print("\nBước tiếp theo: python step3_label.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Làm sạch Vietnam Landmark Dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ xem kết quả, không xóa/di chuyển file")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Sẽ không di chuyển / xóa file nào.")

    make_dirs(args.dry_run)

    all_stats = {}
    for cls in LANDMARKS:
        if not (RAW_DIR / cls).exists():
            print(f"  [{cls}] Thư mục không tồn tại, bỏ qua.")
            continue
        all_stats[cls] = clean_class(cls, args.dry_run)

    print_report(all_stats)