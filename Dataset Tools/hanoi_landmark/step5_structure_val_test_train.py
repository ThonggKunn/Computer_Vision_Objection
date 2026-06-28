"""
=============================================================
Vietnam Landmark Dataset — Bước 4: Chuẩn hóa cấu trúc
=============================================================
Tạo cấu trúc thư mục chuẩn:

    dataset/
    ├── train/
    │   ├── ho_guom/
    │   ├── chua_mot_cot/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...

Cách dùng:
    python step4_structure.py
=============================================================
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm

CLEAN_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "clean_images"
DATASET_DIR = Path(__file__).parent.parent.parent / "Data" / "hanoi_landmark" / "dataset"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

LANDMARKS = [
    "ho_guom", "chua_mot_cot", "lang_bac", "ho_tay", "nha_hat_lon",
    "nha_tho_lon", "trang_tien_plaza", "van_mieu",
]

SEED = 42


def make_dirs():
    for split in ("train", "val", "test"):
        for cls in LANDMARKS:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def structure_class(cls: str) -> dict:
    src_dir = CLEAN_DIR / cls
    if not src_dir.exists():
        return {"train": 0, "val": 0, "test": 0}

    files = sorted(src_dir.glob("*.jpg"))
    random.seed(SEED)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train: n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    counts = {}
    for split, file_list in splits.items():
        dst_dir = DATASET_DIR / split / cls
        for f in tqdm(file_list, desc=f"  {cls}/{split}", leave=False, unit="img"):
            shutil.copy2(str(f), str(dst_dir / f.name))
        counts[split] = len(file_list)

    return counts


if __name__ == "__main__":
    make_dirs()
    print("Đang tổ chức cấu trúc dataset...\n")

    total = {"train": 0, "val": 0, "test": 0}
    for cls in LANDMARKS:
        counts = structure_class(cls)
        print(f"  {cls:<20} train={counts['train']:>4}  val={counts['val']:>3}  test={counts['test']:>3}")
        for split in total:
            total[split] += counts[split]

    print("\n" + "=" * 55)
    print(f"  {'TỔNG':<20} train={total['train']:>4}  val={total['val']:>3}  test={total['test']:>3}")
    print("=" * 55)
    print(f"\nDataset đã tổ chức tại: {DATASET_DIR}/")
    print("Bước tiếp theo: python step5_augment.py")