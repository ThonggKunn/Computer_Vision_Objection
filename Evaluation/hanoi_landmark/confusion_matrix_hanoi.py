"""
=====================================================
Vẽ Confusion Matrix — Hanoi Landmark
=====================================================
Cách dùng:
    python plot_confusion_matrix.py
=====================================================
"""

from pathlib import Path
import numpy as np
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# ── Config ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "Data" / "hanoi_landmark" / "dataset"
MODEL_PATH   = PROJECT_ROOT / "Models" / "efficientnetv2_s_hanoi_best.pt"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 8
IMG_SIZE    = 224
BATCH_SIZE  = 8

CLASS_NAMES_VI = {
    "chua_mot_cot":     "Chùa Một Cột",
    "ho_guom":          "Hồ Gươm",
    "ho_tay":           "Hồ Tây",
    "lang_bac":         "Lăng Bác",
    "nha_hat_lon":      "Nhà hát lớn",
    "nha_tho_lon":      "Nhà thờ lớn",
    "trang_tien_plaza": "Tràng Tiền Plaza",
    "van_mieu":         "Văn Miếu",
}


def get_predictions():
    test_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_tf)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    model = timm.create_model("tf_efficientnetv2_s",
                               pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs    = imgs.to(DEVICE)
            preds   = model(imgs).argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred, test_ds.classes


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm        = confusion_matrix(y_true, y_pred)
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels_vi = [CLASS_NAMES_VI.get(c, c) for c in class_names]
    n         = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Confusion Matrix — Hanoi Landmark", fontsize=16, fontweight="bold", y=1.01)

    for ax, data, title, fmt, cmap in [
        (axes[0], cm,      "Số lượng",   "d",    "Blues"),
        (axes[1], cm_norm, "Tỷ lệ (%)",  ".2f",  "Greens"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_vi, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(labels_vi, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val   = data[i, j]
                text  = f"{val:{fmt}}" if fmt == "d" else f"{val*100:.1f}%"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Lưu: {out}")


if __name__ == "__main__":
    print("Vẽ Confusion Matrix...")
    y_true, y_pred, class_names = get_predictions()
    plot_confusion_matrix(y_true, y_pred, class_names)