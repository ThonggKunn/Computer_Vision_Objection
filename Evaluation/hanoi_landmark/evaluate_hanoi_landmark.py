"""
=====================================================
Evaluate Hanoi Landmark Model
=====================================================
Cách dùng:
    python evaluate_hanoi_landmark.py
=====================================================
"""

from pathlib import Path
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "Data" / "hanoi_landmark" / "dataset"
MODEL_PATH   = PROJECT_ROOT / "Models" / "efficientnetv2_s_hanoi_best.pt"

DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 8
IMG_SIZE    = 224
BATCH_SIZE  = 8


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():

    # ── Transform ─────────────────────────────────────────
    test_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    # ── Dataset ───────────────────────────────────────────
    test_ds = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=test_tf,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,      # fix lỗi multiprocessing trên macOS
    )
    class_names = test_ds.classes

    print(f"Test images : {len(test_ds)}")
    print(f"Classes     : {class_names}")
    print(f"Device      : {DEVICE}")

    # ── Model ─────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Không tìm thấy model: {MODEL_PATH}")
        print("  → Chạy train_landmark.py trước!")
        return

    print(f"\n[Model] Load từ: {MODEL_PATH.name}")
    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    print("  ✓ Load xong\n")

    # ── Evaluate ──────────────────────────────────────────
    y_true = []
    y_pred = []
    y_prob = []

    print("Đang đánh giá...")
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # ── Accuracy tổng ─────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print(f"  TEST ACCURACY : {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 60)

    # ── Classification report ─────────────────────────────
    print("\nCLASSIFICATION REPORT\n")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    ))

    # ── Per-class accuracy ────────────────────────────────
    print("PER-CLASS ACCURACY\n")
    cm = confusion_matrix(y_true, y_pred)
    for i, cls in enumerate(class_names):
        correct = cm[i, i]
        total   = cm[i].sum()
        cls_acc = correct / total if total > 0 else 0
        bar     = "█" * int(cls_acc * 25) + "░" * (25 - int(cls_acc * 25))
        print(f"  {cls:<22} {cls_acc:.4f}  [{bar}]  {correct}/{total}")

    # ── Confusion matrix ──────────────────────────────────
    print("\nCONFUSION MATRIX\n")
    # Header
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Đã lưu confusion_matrix.png")

    # ── Top-3 nhầm lẫn nhiều nhất ────────────────────────
    print("\nTOP NHẦM LẪN\n")
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:5]:
        print(f"  {true_cls:<22} → {pred_cls:<22} : {count} lần")

    print("\n" + "=" * 60)
    print(f"  Kết quả: {acc*100:.2f}%")
    if acc >= 0.90:
        print("  ✓ Xuất sắc — sẵn sàng tích hợp vào app")
    elif acc >= 0.80:
        print("  ✓ Tốt — có thể dùng được")
    elif acc >= 0.70:
        print("  ⚠ Trung bình — cần thêm data hoặc train lâu hơn")
    else:
        print("  ✗ Chưa đủ tốt — cần xem lại dataset và augmentation")
    print("=" * 60)


if __name__ == "__main__":
    main()