"""
=============================================================
Vietnam Landmark Dataset — Bước 6 & 7: Kiểm tra chất lượng
=============================================================
Kiểm tra:
  ✓ Số lượng ảnh mỗi class (cân bằng)
  ✓ Phân phối train/val/test
  ✓ Kích thước ảnh
  ✓ Tỷ lệ label chính xác
  ✓ Không có class rỗng
  ✓ Không có ảnh trùng giữa các split
  ✓ In báo cáo đầy đủ

Cài đặt:
    pip install pillow imagehash tqdm matplotlib

Cách dùng:
    python step6_verify.py
    python step6_verify.py --plot    # vẽ biểu đồ phân phối
=============================================================
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DATASET_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "dataset"
SPLITS      = ["train", "val", "test"]

LANDMARKS = {
    "ho_guom":          "Hồ Gươm",
    "chua_mot_cot":     "Chùa Một Cột",
    "lang_bac":         "Lăng Bác",
    "ho_tay":           "Hồ Tây",
    "nha_hat_lon":      "Nhà hát lớn",
    "nha_tho_lon":      "Nhà thờ lớn",
    "trang_tien_plaza": "Tràng Tiền Plaza",
    "van_mieu":         "Văn Miếu",
}

MIN_IMAGES_PER_CLASS = 599   # ngưỡng cảnh báo
EXPECTED_SIZE        = (224, 224)


# ── Kiểm tra ──────────────────────────────────────────────

def count_images() -> dict:
    """Đếm ảnh theo split và class."""
    counts = defaultdict(lambda: defaultdict(int))
    for split in SPLITS:
        for cls in LANDMARKS:
            d = DATASET_DIR / split / cls
            if d.exists():
                counts[split][cls] = len(list(d.glob("*.jpg")))
    return counts


def check_size_consistency() -> dict:
    """Kiểm tra tất cả ảnh có đúng kích thước không."""
    issues = defaultdict(list)
    for split in SPLITS:
        for cls in LANDMARKS:
            d = DATASET_DIR / split / cls
            if not d.exists():
                continue
            for f in d.glob("*.jpg"):
                try:
                    img = Image.open(f)
                    if img.size != EXPECTED_SIZE:
                        issues[f"{split}/{cls}"].append(f"{f.name}: {img.size}")
                except Exception as e:
                    issues[f"{split}/{cls}"].append(f"{f.name}: ERROR {e}")
    return issues


def check_cross_split_leakage() -> list:
    """
    Phát hiện ảnh trùng giữa train/val/test dùng MD5.
    Nếu có cùng ảnh trong train và test → data leakage!
    """
    hashes = defaultdict(list)   # hash → [(split, cls, filename)]
    for split in SPLITS:
        for cls in LANDMARKS:
            d = DATASET_DIR / split / cls
            if not d.exists():
                continue
            for f in d.glob("*.jpg"):
                with open(f, "rb") as fh:
                    h = hashlib.md5(fh.read()).hexdigest()
                hashes[h].append((split, cls, f.name))

    leaks = []
    for h, locations in hashes.items():
        splits_seen = set(loc[0] for loc in locations)
        if len(splits_seen) > 1:
            leaks.append(locations)
    return leaks


def compute_split_ratios(counts: dict) -> dict:
    """Tính tỷ lệ train/val/test thực tế."""
    ratios = {}
    for cls in LANDMARKS:
        total = sum(counts[s].get(cls, 0) for s in SPLITS)
        if total == 0:
            ratios[cls] = {"train": 0, "val": 0, "test": 0, "total": 0}
            continue
        ratios[cls] = {
            split: round(counts[split].get(cls, 0) / total * 100, 1)
            for split in SPLITS
        }
        ratios[cls]["total"] = total
    return ratios


# ── Báo cáo ───────────────────────────────────────────────

def print_report(counts, ratios, size_issues, leaks, args):
    print("\n" + "=" * 70)
    print("BÁO CÁO CHẤT LƯỢNG — Hanoi Landmark Dataset")
    print("=" * 70)

    # 1. Phân phối ảnh
    print("\n[1] Phân phối ảnh theo split\n")
    hdr = f"  {'Class':<22} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}  {'Train%':>7} {'Val%':>5} {'Test%':>6}"
    print(hdr)
    print("-" * 70)

    warnings = []
    total_train = total_val = total_test = 0

    for cls, name_vi in LANDMARKS.items():
        tr = counts["train"].get(cls, 0)
        vl = counts["val"].get(cls, 0)
        ts = counts["test"].get(cls, 0)
        tt = tr + vl + ts
        r  = ratios[cls]

        flag = ""
        if tt < MIN_IMAGES_PER_CLASS:
            flag = "  ⚠ THIẾU"
            warnings.append(f"  {cls}: chỉ có {tt} ảnh (cần ≥ {MIN_IMAGES_PER_CLASS})")
        if tr == 0:
            flag = "  ✗ RỖNG"

        print(f"  {cls:<22} {tr:>6} {vl:>5} {ts:>5} {tt:>6}  {r['train']:>6}% {r['val']:>4}% {r['test']:>5}%{flag}")
        total_train += tr
        total_val   += vl
        total_test  += ts

    total = total_train + total_val + total_test
    print("-" * 70)
    tr_pct = round(total_train / total * 100, 1) if total else 0
    vl_pct = round(total_val   / total * 100, 1) if total else 0
    ts_pct = round(total_test  / total * 100, 1) if total else 0
    print(f"  {'TỔNG':<22} {total_train:>6} {total_val:>5} {total_test:>5} {total:>6}  {tr_pct:>6}% {vl_pct:>4}% {ts_pct:>5}%")

    # 2. Class balance
    print("\n[2] Cân bằng class\n")
    vals = [counts["train"].get(c, 0) for c in LANDMARKS]
    if vals:
        mn, mx = min(vals), max(vals)
        imbalance = round((mx - mn) / mx * 100, 1) if mx else 0
        print(f"  Min ảnh/class (train): {mn}")
        print(f"  Max ảnh/class (train): {mx}")
        print(f"  Mức độ mất cân bằng:   {imbalance}%")
        if imbalance > 30:
            warnings.append(f"  Mất cân bằng cao ({imbalance}%) — cân nhắc oversample class nhỏ")

    # 3. Kích thước ảnh
    print("\n[3] Kiểm tra kích thước ảnh\n")
    if not size_issues:
        print(f"  ✓ Tất cả ảnh đúng kích thước {EXPECTED_SIZE}")
    else:
        total_issues = sum(len(v) for v in size_issues.values())
        print(f"  ⚠ {total_issues} ảnh sai kích thước:")
        for loc, items in list(size_issues.items())[:5]:
            print(f"    {loc}: {items[0]}")
        warnings.append(f"  {total_issues} ảnh sai kích thước → chạy lại step5_augment.py")

    # 4. Data leakage
    print("\n[4] Kiểm tra data leakage (trùng giữa splits)\n")
    if not leaks:
        print("  ✓ Không có ảnh trùng giữa train/val/test")
    else:
        print(f"  ✗ {len(leaks)} ảnh trùng giữa các split!")
        for loc_list in leaks[:3]:
            print(f"    {loc_list}")
        warnings.append(f"  {len(leaks)} ảnh bị data leakage — xem chi tiết ở trên")

    # 5. Warnings tổng hợp
    print("\n[5] Tổng hợp cảnh báo\n")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ Dataset sạch, không có vấn đề!")

    # 6. Checklist chất lượng
    print("\n[6] Checklist chất lượng dataset\n")
    checks = [
        ("Đa góc chụp",          "Cần review thủ công hoặc dùng diversity metric"),
        ("Điều kiện sáng/tối",   "Cần review thủ công"),
        ("Nhiều thời tiết",      "Cần review thủ công"),
        ("Cân bằng class",       "✓ Đã kiểm tra tự động ở mục [2]"),
        ("Label chính xác",      "✓ Kiểm tra qua cấu trúc thư mục"),
        ("Không data leakage",   "✓ Đã kiểm tra tự động ở mục [4]"),
        ("Kích thước chuẩn",     "✓ Đã kiểm tra tự động ở mục [3]"),
    ]
    for item, note in checks:
        print(f"  {'[ ]' if 'thủ công' in note else '[✓]'} {item:<25} — {note}")

    # 7. Lưu report
    report = {
        "total_images": total,
        "splits": {"train": total_train, "val": total_val, "test": total_test},
        "per_class": {cls: {"train": counts["train"].get(cls, 0),
                             "val":   counts["val"].get(cls, 0),
                             "test":  counts["test"].get(cls, 0)}
                      for cls in LANDMARKS},
        "warnings": warnings,
        "leakage_count": len(leaks),
        "size_issues": sum(len(v) for v in size_issues.values()),
    }
    with open("dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ Báo cáo đã lưu vào dataset_report.json")
    print("=" * 70)

    # 8. Vẽ biểu đồ (nếu --plot)
    if args.plot:
        plot_distribution(counts)


def plot_distribution(counts: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.family"] = "DejaVu Sans"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#4C9BE8", "#F5A623", "#7ED321"]

        for ax, split, color in zip(axes, SPLITS, colors):
            cls_names = list(LANDMARKS.keys())
            values = [counts[split].get(c, 0) for c in cls_names]
            bars = ax.barh(cls_names, values, color=color, alpha=0.85)
            ax.set_title(f"{split.upper()} ({sum(values)} ảnh)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Số ảnh")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=9)

        plt.suptitle("Hanoi Landmark Dataset — Phân phối ảnh", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("dataset_distribution.png", dpi=150, bbox_inches="tight")
        print("✓ Biểu đồ đã lưu: dataset_distribution.png")
        plt.show()
    except ImportError:
        print("  Cài matplotlib để vẽ biểu đồ: pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng Vietnam Landmark Dataset")
    parser.add_argument("--plot", action="store_true", help="Vẽ biểu đồ phân phối")
    args = parser.parse_args()

    print("Đang kiểm tra dataset...\n")

    counts      = count_images()
    ratios      = compute_split_ratios(counts)
    size_issues = check_size_consistency()

    print("Đang kiểm tra data leakage (có thể mất vài phút)...")
    leaks = check_cross_split_leakage()

    print_report(counts, ratios, size_issues, leaks, args)