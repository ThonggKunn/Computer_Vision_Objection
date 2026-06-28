from pathlib import Path

# Đường dẫn dataset
DATASET_DIR = Path("/Users/1pro/PycharmProjects/CV/Data/hanoi_landmark/clean_images")

# Các định dạng ảnh được chấp nhận
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0

print("=" * 50)
print("THỐNG KÊ DATASET")
print("=" * 50)

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    count = sum(
        1 for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

    total_images += count
    print(f"{class_dir.name:25} : {count:5d} ảnh")

print("=" * 50)
print(f"TỔNG SỐ ẢNH: {total_images}")
print("=" * 50)