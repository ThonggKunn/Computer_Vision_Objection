"""
Tăng cường dữ liệu thông minh — multiplier khác nhau cho từng class
dựa trên số ảnh hiện có để đạt cân bằng.
"""

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEAN_DIR    = PROJECT_ROOT / "Data" / "hanoi_landmark" / "clean_images"
TARGET_SIZE  = (224, 224)
TARGET_MIN   = 600    # số ảnh tối thiểu mỗi class sau augment
SEED         = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Augmentation functions ────────────────────────────────

def aug_rotate(img):
    angle = random.uniform(-25, 25)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_flip(img):
    return cv2.flip(img, 1)

def aug_brightness(img):
    alpha = random.uniform(0.65, 1.4)
    beta  = random.randint(-40, 40)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aug_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def aug_noise(img):
    std   = random.uniform(5, 20)
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def aug_crop(img):
    h, w      = img.shape[:2]
    ratio     = random.uniform(0.78, 0.95)
    new_h     = int(h * ratio)
    new_w     = int(w * ratio)
    y         = random.randint(0, h - new_h)
    x         = random.randint(0, w - new_w)
    cropped   = img[y:y+new_h, x:x+new_w]
    return cv2.resize(cropped, (w, h))

def aug_color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] += random.uniform(-12, 12)
    hsv[:,:,1] *= random.uniform(0.75, 1.25)
    hsv[:,:,2] *= random.uniform(0.75, 1.25)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def aug_perspective(img):
    """Biến đổi phối cảnh nhẹ — giúp model robust hơn."""
    h, w  = img.shape[:2]
    shift = int(min(h, w) * 0.05)
    src   = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst   = np.float32([
        [random.randint(0, shift),      random.randint(0, shift)],
        [w - random.randint(0, shift),  random.randint(0, shift)],
        [w - random.randint(0, shift),  h - random.randint(0, shift)],
        [random.randint(0, shift),      h - random.randint(0, shift)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

AUG_FUNCS = [
    aug_rotate, aug_flip, aug_brightness, aug_blur,
    aug_noise, aug_crop, aug_color_jitter, aug_perspective,
]

def augment_image(img: np.ndarray, n_ops: int = 3) -> np.ndarray:
    ops = random.sample(AUG_FUNCS, min(n_ops, len(AUG_FUNCS)))
    for op in ops:
        img = op(img)
    return img


# ── Main augment ──────────────────────────────────────────

def augment_class(cls: str, current: int) -> int:
    cls_dir = CLEAN_DIR / cls
    if not cls_dir.exists():
        print(f"  [{cls}] Không tìm thấy thư mục, bỏ qua.")
        return 0

    originals = sorted(cls_dir.glob("*.jpg"))
    if not originals:
        return 0

    # Resize ảnh gốc về TARGET_SIZE trước
    for f in originals:
        img = cv2.imread(str(f))
        if img is not None:
            cv2.imwrite(str(f), cv2.resize(img, TARGET_SIZE))

    # Tính số ảnh cần tạo thêm
    need    = max(0, TARGET_MIN - current)
    n_aug   = 0

    if need == 0:
        print(f"  [{cls}] Đã đủ {current} ảnh, bỏ qua.")
        return 0

    print(f"  [{cls}] {current} ảnh → cần thêm {need} ảnh")

    # Tạo ảnh augmented bằng cách lặp qua originals
    pbar = tqdm(total=need, desc=f"    {cls}", unit="img")
    idx  = current

    while n_aug < need:
        for src_file in originals:
            if n_aug >= need:
                break
            img = cv2.imread(str(src_file))
            if img is None:
                continue

            # Số ops tỷ lệ với mức thiếu — thiếu nhiều → aug mạnh hơn
            multiplier = need / max(current, 1)
            n_ops      = 4 if multiplier > 4 else 3

            aug  = augment_image(img.copy(), n_ops=n_ops)
            out  = cls_dir / f"{cls}_aug_{idx:05d}.jpg"
            cv2.imwrite(str(out), aug)
            idx   += 1
            n_aug += 1
            pbar.update(1)

    pbar.close()
    return n_aug


def main():
    print("\nTăng cường dữ liệu thông minh")
    print(f"Mục tiêu: {TARGET_MIN} ảnh/class\n")

    # Đếm số ảnh hiện có
    classes = sorted(CLEAN_DIR.iterdir()) if CLEAN_DIR.exists() else []
    if not classes:
        print(f"Không tìm thấy dữ liệu tại {CLEAN_DIR}")
        return

    total_added = 0
    results     = {}

    for cls_dir in classes:
        if not cls_dir.is_dir():
            continue
        cls     = cls_dir.name
        current = len(list(cls_dir.glob("*.jpg")))
        added   = augment_class(cls, current)
        total_added       += added
        results[cls]       = current + added

    # Báo cáo
    print("\n" + "=" * 45)
    print("KẾT QUẢ TĂNG CƯỜNG")
    print("=" * 45)
    for cls, total in results.items():
        bar    = "█" * (total // 30) + "░" * max(0, (TARGET_MIN - total) // 30)
        status = "✓" if total >= TARGET_MIN else f"⚠ {total}"
        print(f"  {cls:<22} {total:>4}  {status}")
    print("-" * 45)
    print(f"  Tổng ảnh thêm : +{total_added}")
    print(f"  Tổng ảnh hiện : {sum(results.values())}")
    print("=" * 45)
    print("\nBước tiếp theo: python step4_structure.py")


if __name__ == "__main__":
    main()