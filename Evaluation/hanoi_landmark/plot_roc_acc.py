"""
=====================================================
Vẽ ROC Curve + Per-class Accuracy — Hanoi Landmark
=====================================================
Cách dùng:
    python plot_roc_accuracy.py
=====================================================
"""

from pathlib import Path
import numpy as np
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

# ── Config ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
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

COLORS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]


def get_predictions_with_probs():
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

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return (np.array(y_true), np.array(y_pred),
            np.array(y_prob), test_ds.classes)


def plot_roc_curve(y_true, y_prob, class_names):
    """Vẽ ROC curve cho từng class (One-vs-Rest)."""
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    labels_vi = [CLASS_NAMES_VI.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("ROC Curve — Hanoi Landmark (One-vs-Rest)",
                 fontsize=14, fontweight="bold")

    auc_scores = []
    for i, (cls, color) in enumerate(zip(class_names, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{labels_vi[i]}  (AUC = {roc_auc:.4f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, "k--", linewidth=2.5,
            label=f"Micro-avg  (AUC = {auc_micro:.4f})")

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1.5,
            label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    out = RESULTS_DIR / "roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Lưu: {out}")
    print(f"  Micro-avg AUC: {auc_micro:.4f}")
    for cls, score in zip(class_names, auc_scores):
        print(f"  {cls:<22}: AUC = {score:.4f}")


def plot_per_class_accuracy(y_true, y_pred, class_names):
    """Vẽ bar chart accuracy từng class."""
    cm        = confusion_matrix(y_true, y_pred)
    labels_vi = [CLASS_NAMES_VI.get(c, c) for c in class_names]

    accs = []
    for i in range(len(class_names)):
        total = cm[i].sum()
        accs.append(cm[i, i] / total if total > 0 else 0)

    # Sắp xếp từ cao đến thấp
    sorted_idx = np.argsort(accs)[::-1]
    sorted_acc = [accs[i] for i in sorted_idx]
    sorted_lbl = [labels_vi[i] for i in sorted_idx]
    sorted_col = [COLORS[i % len(COLORS)] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(sorted_lbl, sorted_acc, color=sorted_col,
                   edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=11, fontweight="bold")

    # Đường trung bình
    avg_acc = np.mean(accs)
    ax.axvline(avg_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Trung bình: {avg_acc*100:.1f}%")

    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-class Accuracy — Hanoi Landmark",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "per_class_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Lưu: {out}")


if __name__ == "__main__":
    print("Đang tính toán predictions...")
    y_true, y_pred, y_prob, class_names = get_predictions_with_probs()

    print("\nVẽ ROC Curve...")
    plot_roc_curve(y_true, y_prob, class_names)

    print("\nVẽ Per-class Accuracy...")
    plot_per_class_accuracy(y_true, y_pred, class_names)