"""
=============================================================
Vietnam Landmark Dataset — Thu thập ảnh v2
=============================================================
Tích hợp 4 nguồn:
  1. Openverse  — không cần key, Creative Commons
  2. Wikimedia  — không cần key, ảnh lịch sử chất lượng cao
  3. Pexels     — cần key free (https://www.pexels.com/api/)
  4. Pixabay    — cần key free (https://pixabay.com/api/docs/)

Cài đặt:
    pip install requests pillow tqdm

Cách dùng:
    python step1_gallery.py                         # tất cả nguồn
    python step1_gallery.py --source openverse      # chỉ Openverse
    python step1_gallery.py --source wikimedia      # chỉ Wikimedia
    python step1_gallery.py --source pexels         # chỉ Pexels
    python step1_gallery.py --source pixabay        # chỉ Pixabay
    python step1_gallery.py --cls ho_guom           # chỉ 1 class
    python step1_gallery.py --cls ho_guom --source pexels
=============================================================
"""

import os
import time
import hashlib
import argparse
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from torch.export import export
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

Image.MAX_IMAGE_PIXELS = None

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════

PROJECT_ROOT     = Path(__file__).parent.parent.parent
RAW_DIR          = PROJECT_ROOT / "Data" / "vietnam_landmark" / "raw_images"
TARGET_PER_CLASS = 1000
MIN_SIZE         = 250

# Đặt key vào .env hoặc export trước khi chạy:
# export PEXELS_KEY="your_key"
#   export PIXABAY_KEY="your_key"
PEXELS_KEY  = os.environ.get("PEXELS_KEY",  "")
PIXABAY_KEY = os.environ.get("PIXABAY_KEY", "")

HEADERS = {
    "User-Agent": (
        "VietnamLandmarkDataset/1.0 "
        "(student research project; python-requests)"
    ),
    "Accept": "application/json",
}

# ══════════════════════════════════════════════════════════
# QUERIES
# ══════════════════════════════════════════════════════════

LANDMARKS = {
    "van_mieu": [
        "Temple of Literature Hanoi",
        "Văn Miếu Hà Nội",
        "Văn Miếu QUốc Tử Giám",
        "Temple of Literature Vietnam",
    ],

    "nha_tho_lon": [
        "Nhà Thờ Lớn",
        "Nhà Thờ Lớn Hà Nội",
        "St. Joseph's Cathedral Hanoi",
        "Notre Dame Cathedral Hanoi",
    ],

    "ho_guom": [
        "Hồ Hoàn Kiếm Hà Nội",
        "Hồ Gươm Hà Nội",
        "Hoan Kiem Lake Vietnam",
        "Ho Guom Hanoi",
        "Hoan Kiem lake aerial",
        "Tháp Rùa Hà Nội",
        "Turtle Tower Hanoi",
    ],
    "chua_mot_cot": [
        "One Pillar Pagoda Hanoi",
        "Chùa Một Cột Hà Nội",
        "chùa Một Cột",
        "Dien Huu temple Hanoi",
    ],
    "lang_bac": [
        "Lăng Bác Hà Nội",
        "Ho Chi Minh tomb Vietnam",
        "Ba Dinh square mausoleum",
        "lăng Chủ tịch Hồ Chí Minh",
        "Lang Bac Hanoi Vietnam",
    ],

    "nha_hat_lon": [
        "Hanoi Opera House Vietnam",
        "nhà hát lớn Hà Nội",
        "Hanoi Opera House architecture",
        "Grand Opera House Hanoi",
        "Nhà Hát Lớn Hà Nội",
    ],
    "ho_tay": [
        "Hồ Tây Hà Nội",
        "West Lake Hanoi Vietnam",
        "Ho Tay Hanoi sunset",
        "hồ Tây Hà Nội hoàng hôn",
        "West Lake Hanoi aerial",

    ],
    "trang_tien_plaza": [
        "Trang Tien Plaza Hanoi",
        "Tràng Tiền Plaza Hà Nội",
        "Trang Tien street Hanoi",
        "Trang Tien Plaza night Hanoi",
    ],
}


# ══════════════════════════════════════════════════════════
# TIỆN ÍCH DÙNG CHUNG
# ══════════════════════════════════════════════════════════

def make_dirs():
    for cls in LANDMARKS:
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)


def count_images(cls: str) -> int:
    return len(list((RAW_DIR / cls).glob("*.jpg")))


def save_image(data: bytes, cls: str, idx: int) -> bool:
    out_dir   = RAW_DIR / cls
    h         = hashlib.md5(data).hexdigest()
    hash_file = out_dir / f".hash_{h}"
    if hash_file.exists():
        return False
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        w, h_px = img.size
        if w < MIN_SIZE or h_px < MIN_SIZE:
            return False
        if w / h_px < 0.4:   # bỏ ảnh quá dọc (selfie, portrait)
            return False
        fname = out_dir / f"{cls}_{idx:04d}.jpg"
        img.save(fname, "JPEG", quality=95)
        (out_dir / f".hash_{hashlib.md5(data).hexdigest()}").touch()
        return True
    except Exception:
        return False


def rename_files(cls: str):
    d     = RAW_DIR / cls
    files = sorted(f for f in d.glob("*.jpg") if not f.name.startswith(cls))
    for i, f in enumerate(files):
        new_name = d / f"{cls}_{i:04d}.jpg"
        if not new_name.exists():
            f.rename(new_name)
        else:
            f.rename(d / f"{cls}_{i + 9000:04d}.jpg")


def download_image(url: str, cls: str) -> bool:
    """Download 1 ảnh từ URL và lưu vào class."""
    try:
        r = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": HEADERS["User-Agent"]},
            stream=True,
        )
        if r.status_code != 200:
            return False
        data = r.content
        return save_image(data, cls, count_images(cls))
    except Exception:
        return False


def print_progress(cls: str, source: str):
    n      = count_images(cls)
    status = "✓" if n >= TARGET_PER_CLASS else f"thiếu {TARGET_PER_CLASS - n}"
    print(f"  [{source}][{cls}] {n}/{TARGET_PER_CLASS} — {status}")


# ══════════════════════════════════════════════════════════
# NGUỒN 1 — OPENVERSE (không cần key)
# ══════════════════════════════════════════════════════════

def collect_openverse(target_classes: list):
    print("\n" + "=" * 52)
    print("[Openverse] Bắt đầu — không cần API key")
    print("=" * 52)

    for cls in target_classes:
        if count_images(cls) >= TARGET_PER_CLASS:
            print(f"  [{cls}] Đã đủ ảnh, bỏ qua.")
            continue

        queries = LANDMARKS.get(cls, [])
        print(f"\n  [{cls}] {count_images(cls)}/{TARGET_PER_CLASS}")

        for query in queries:
            if count_images(cls) >= TARGET_PER_CLASS:
                break

            for page in range(1, 20):
                if count_images(cls) >= TARGET_PER_CLASS:
                    break
                try:
                    resp = requests.get(
                        "https://api.openverse.org/v1/images/",
                        params={
                            "q":            query,
                            "page":         page,
                            "page_size":    50,
                            "license_type": "commercial,modification",
                            "mature":       "false",
                        },
                        headers=HEADERS,
                        timeout=12,
                    )
                    if resp.status_code == 429:
                        print(f"    Rate limit — chờ 30s...")
                        time.sleep(30)
                        continue
                    if resp.status_code != 200:
                        break

                    items = resp.json().get("results", [])
                    if not items:
                        break

                    saved = 0
                    for item in items:
                        if count_images(cls) >= TARGET_PER_CLASS:
                            break
                        url = item.get("url", "")
                        if url and download_image(url, cls):
                            saved += 1
                        time.sleep(0.25)

                    print(f"    '{query}' p{page}: +{saved} ảnh")
                    time.sleep(1.5)

                except Exception as e:
                    print(f"    Lỗi: {e}")
                    time.sleep(3)
                    break

        rename_files(cls)
        print_progress(cls, "Openverse")

    print("\n[Openverse] Xong!")


# ══════════════════════════════════════════════════════════
# NGUỒN 2 — WIKIMEDIA COMMONS (không cần key)
# ══════════════════════════════════════════════════════════

def collect_wikimedia(target_classes: list):
    print("\n" + "=" * 52)
    print("[Wikimedia] Bắt đầu — không cần API key")
    print("=" * 52)

    API_URL = "https://commons.wikimedia.org/w/api.php"

    for cls in target_classes:
        if count_images(cls) >= TARGET_PER_CLASS:
            print(f"  [{cls}] Đã đủ ảnh, bỏ qua.")
            continue

        queries = LANDMARKS.get(cls, [])
        print(f"\n  [{cls}] {count_images(cls)}/{TARGET_PER_CLASS}")

        for query in queries:
            if count_images(cls) >= TARGET_PER_CLASS:
                break

            # Tìm file ảnh
            for offset in range(0, 500, 50):
                if count_images(cls) >= TARGET_PER_CLASS:
                    break
                try:
                    resp = requests.get(
                        API_URL,
                        params={
                            "action":        "query",
                            "generator":     "search",
                            "gsrsearch":     f"File: {query}",
                            "gsrnamespace":  6,
                            "gsrlimit":      50,
                            "gsroffset":     offset,
                            "prop":          "imageinfo",
                            "iiprop":        "url|size",
                            "iiurlwidth":    1000,
                            "format":        "json",
                        },
                        headers=HEADERS,
                        timeout=12,
                    )
                    if resp.status_code != 200:
                        break
                    text = resp.text.strip()
                    if not text:
                        break

                    pages = resp.json().get("query", {}).get("pages", {})
                    if not pages:
                        break

                    saved = 0
                    for page in pages.values():
                        if count_images(cls) >= TARGET_PER_CLASS:
                            break
                        info = page.get("imageinfo", [{}])[0]
                        url  = info.get("thumburl") or info.get("url", "")
                        if not url:
                            continue
                        ext = url.split("?")[0].lower()
                        if not any(ext.endswith(e) for e in [".jpg", ".jpeg", ".png"]):
                            continue
                        # Bỏ ảnh quá nhỏ từ metadata
                        if info.get("thumbwidth", 999) < MIN_SIZE:
                            continue
                        if download_image(url, cls):
                            saved += 1
                        time.sleep(0.2)

                    print(f"    '{query}' offset{offset}: +{saved} ảnh")
                    time.sleep(1)

                except Exception as e:
                    print(f"    Lỗi: {e}")
                    time.sleep(3)
                    break

        rename_files(cls)
        print_progress(cls, "Wikimedia")

    print("\n[Wikimedia] Xong!")


# ══════════════════════════════════════════════════════════
# NGUỒN 3 — PEXELS (cần key free)
# ══════════════════════════════════════════════════════════

def collect_pexels(target_classes: list):
    print("\n" + "=" * 52)
    print("[Pexels] Bắt đầu")
    print("=" * 52)


    if not PEXELS_KEY:
        print("  Chưa có PEXELS_KEY!")
        print("  1. Đăng ký free tại: https://www.pexels.com/api/")
        print("  2. Chạy: export PEXELS_KEY='your_key'")
        return

    for cls in target_classes:
        if count_images(cls) >= TARGET_PER_CLASS:
            print(f"  [{cls}] Đã đủ ảnh, bỏ qua.")
            continue

        queries = LANDMARKS.get(cls, [])
        print(f"\n  [{cls}] {count_images(cls)}/{TARGET_PER_CLASS}")

        for query in queries:
            if count_images(cls) >= TARGET_PER_CLASS:
                break

            for page in range(1, 20):
                if count_images(cls) >= TARGET_PER_CLASS:
                    break
                try:
                    resp = requests.get(
                        "https://api.pexels.com/v1/search",
                        headers={**HEADERS, "Authorization": PEXELS_KEY},
                        params={
                            "query":    query,
                            "per_page": 80,
                            "page":     page,
                            "size":     "medium",
                        },
                        timeout=12,
                    )
                    if resp.status_code == 429:
                        print(f"    Rate limit — chờ 60s...")
                        time.sleep(60)
                        continue
                    if resp.status_code != 200:
                        break

                    photos = resp.json().get("photos", [])
                    if not photos:
                        break

                    saved = 0
                    for p in photos:
                        if count_images(cls) >= TARGET_PER_CLASS:
                            break
                        url = p.get("src", {}).get("large", "")
                        if url and download_image(url, cls):
                            saved += 1
                        time.sleep(0.15)

                    print(f"    '{query}' p{page}: +{saved} ảnh")
                    time.sleep(1)

                    # Kiểm tra còn trang không
                    total_results = resp.json().get("total_results", 0)
                    if page * 80 >= total_results:
                        break

                except Exception as e:
                    print(f"    Lỗi: {e}")
                    time.sleep(3)
                    break

        rename_files(cls)
        print_progress(cls, "Pexels")

    print("\n[Pexels] Xong!")


# ══════════════════════════════════════════════════════════
# NGUỒN 4 — PIXABAY (cần key free)
# ══════════════════════════════════════════════════════════

def collect_pixabay(target_classes: list):
    print("\n" + "=" * 52)
    print("[Pixabay] Bắt đầu")
    print("=" * 52)

    if not PIXABAY_KEY:
        print("  Chưa có PIXABAY_KEY!")
        print("  1. Đăng ký free tại: https://pixabay.com/api/docs/")
        print("  2. Chạy: export PIXABAY_KEY='your_key'")
        return

    for cls in target_classes:
        if count_images(cls) >= TARGET_PER_CLASS:
            print(f"  [{cls}] Đã đủ ảnh, bỏ qua.")
            continue

        queries = LANDMARKS.get(cls, [])
        print(f"\n  [{cls}] {count_images(cls)}/{TARGET_PER_CLASS}")

        for query in queries:
            if count_images(cls) >= TARGET_PER_CLASS:
                break

            for page in range(1, 20):
                if count_images(cls) >= TARGET_PER_CLASS:
                    break
                try:
                    resp = requests.get(
                        "https://pixabay.com/api/",
                        params={
                            "key":          PIXABAY_KEY,
                            "q":            query,
                            "image_type":   "photo",
                            "per_page":     200,
                            "page":         page,
                            "safesearch":   "true",
                            "min_width":    MIN_SIZE,
                            "min_height":   MIN_SIZE,
                            "order":        "popular",
                        },
                        headers=HEADERS,
                        timeout=12,
                    )
                    if resp.status_code == 429:
                        print(f"    Rate limit — chờ 60s...")
                        time.sleep(60)
                        continue
                    if resp.status_code != 200:
                        break

                    hits = resp.json().get("hits", [])
                    if not hits:
                        break

                    saved = 0
                    for hit in hits:
                        if count_images(cls) >= TARGET_PER_CLASS:
                            break
                        # Ưu tiên ảnh lớn nhất có sẵn
                        url = (hit.get("largeImageURL")
                               or hit.get("webformatURL", ""))
                        if url and download_image(url, cls):
                            saved += 1
                        time.sleep(0.15)

                    print(f"    '{query}' p{page}: +{saved} ảnh")
                    time.sleep(1)

                    total = resp.json().get("totalHits", 0)
                    if page * 200 >= total:
                        break

                except Exception as e:
                    print(f"    Lỗi: {e}")
                    time.sleep(3)
                    break

        rename_files(cls)
        print_progress(cls, "Pixabay")

    print("\n[Pixabay] Xong!")


# ══════════════════════════════════════════════════════════
# TỔNG KẾT
# ══════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 58)
    print("TỔNG KẾT — Vietnam Landmark Dataset")
    print("=" * 58)
    total = 0
    for cls in LANDMARKS:
        n      = count_images(cls)
        total += n
        filled = int(n / TARGET_PER_CLASS * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        status = "✓" if n >= TARGET_PER_CLASS else f"⚠ thiếu {TARGET_PER_CLASS - n}"
        print(f"  {cls:<20} {n:>4}/{TARGET_PER_CLASS} [{bar}] {status}")
    print("-" * 58)
    print(f"  {'TỔNG':<20} {total:>5} / {TARGET_PER_CLASS * len(LANDMARKS)} ảnh")
    print("=" * 58)

    missing = [c for c in LANDMARKS if count_images(c) < TARGET_PER_CLASS]
    if missing:
        print(f"\n  Class còn thiếu: {missing}")
        print("  → Chạy lại với --cls <tên_class> để bổ sung")
    else:
        print("\n  ✓ Đã đủ ảnh tất cả class!")
        print("  → Bước tiếp: python step2_clean.py")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

SOURCE_MAP = {
    "openverse": collect_openverse,
    "wikimedia": collect_wikimedia,
    "pexels":    collect_pexels,
    "pixabay":   collect_pixabay,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thu thập ảnh Vietnam Landmark Dataset"
    )
    parser.add_argument(
        "--source",
        choices=["openverse", "wikimedia", "pexels", "pixabay", "all"],
        default="all",
        help="Nguồn ảnh (mặc định: all)",
    )
    parser.add_argument(
        "--cls",
        default="",
        help="Chỉ crawl 1 class, ví dụ: --cls ho_guom",
    )
    args = parser.parse_args()

    # Xác định class cần crawl
    if args.cls:
        if args.cls not in LANDMARKS:
            print(f"  ✗ Class '{args.cls}' không tồn tại.")
            print(f"  Các class: {list(LANDMARKS.keys())}")
            exit(1)
        target = [args.cls]
    else:
        # Ưu tiên class còn thiếu nhiều nhất
        target = sorted(
            LANDMARKS.keys(),
            key=lambda c: count_images(c),
        )

    make_dirs()

    print(f"\nCrawl {len(target)} class")
    print(f"Mục tiêu: {TARGET_PER_CLASS} ảnh/class")
    if PEXELS_KEY:
        print("Pexels  : ✓ key đã cấu hình")
    else:
        print("Pexels  : ✗ chưa có key (export PEXELS_KEY=...)")
    if PIXABAY_KEY:
        print("Pixabay : ✓ key đã cấu hình")
    else:
        print("Pixabay : ✗ chưa có key (export PIXABAY_KEY=...)")

    # Chạy theo nguồn
    if args.source == "all":
        # Thứ tự tối ưu: không cần key trước, cần key sau
        collect_openverse(target)
        collect_wikimedia(target)
        collect_pexels(target)
        collect_pixabay(target)
    else:
        SOURCE_MAP[args.source](target)

    print_summary()