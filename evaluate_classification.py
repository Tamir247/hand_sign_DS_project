"""Evaluate the trained YOLO classification model and save metrics/plots.

Produces (in ./metrics/):
    - classification_report.txt
    - confusion_matrix.png
    - confusion_matrix_normalized.png
    - pr_curves.png            (per-class precision-recall, one-vs-rest)
    - roc_curves.png           (per-class ROC, one-vs-rest)
    - f1_vs_threshold.png      (per-class F1 vs decision threshold)
    - per_class_metrics.csv
"""

import os
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
)
from ultralytics import YOLO


WEIGHTS = Path("outputs_yolo_classification/best.pt").resolve()
TEST_DIR = Path("hand-sign-drive/test").resolve()
OUT_DIR = Path("metrics").resolve()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_predictions(model: YOLO, test_dir: Path):
    names = model.names                                  # {idx: class_name}
    class_to_idx = {v: k for k, v in names.items()}

    y_true, y_score = [], []
    class_dirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
    for cls_dir in class_dirs:
        if cls_dir.name not in class_to_idx:
            print(f"  Skipping unknown class folder: {cls_dir.name}")
            continue
        true_idx = class_to_idx[cls_dir.name]
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"  {cls_dir.name}: {len(images)} images")
        for img in images:
            r = model.predict(str(img), verbose=False)[0]
            y_true.append(true_idx)
            y_score.append(r.probs.data.cpu().numpy())

    return np.array(y_true), np.vstack(y_score), names


def plot_confusion(cm, class_names, path, normalize=False, title=""):
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if (normalize and v > 0.01) or (not normalize and v > 0):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(y_true, y_score, names, path):
    n_classes = y_score.shape[1]
    fig, ax = plt.subplots(figsize=(11, 8))
    aps = {}
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        p, r, _ = precision_recall_curve(y_bin, y_score[:, i])
        ap = average_precision_score(y_bin, y_score[:, i])
        aps[i] = ap
        ax.plot(r, p, label=f"{names[i]} (AP={ap:.2f})", linewidth=1.2)
    macro_ap = np.mean(list(aps.values())) if aps else float("nan")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Per-class Precision-Recall (macro AP = {macro_ap:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return aps


def plot_roc_curves(y_true, y_score, names, path):
    n_classes = y_score.shape[1]
    fig, ax = plt.subplots(figsize=(11, 8))
    aucs = {}
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        a = auc(fpr, tpr)
        aucs[i] = a
        ax.plot(fpr, tpr, label=f"{names[i]} (AUC={a:.2f})", linewidth=1.2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    macro_auc = np.mean(list(aucs.values())) if aucs else float("nan")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-class ROC (macro AUC = {macro_auc:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return aucs


def plot_f1_vs_threshold(y_true, y_score, names, path):
    n_classes = y_score.shape[1]
    thresholds = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(11, 8))
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        f1s = []
        for t in thresholds:
            y_hat = (y_score[:, i] >= t).astype(int)
            f1s.append(f1_score(y_bin, y_hat, zero_division=0))
        ax.plot(thresholds, f1s, label=names[i], linewidth=1.2)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 vs decision threshold (one-vs-rest)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_per_class_csv(y_true, y_pred, y_score, names, aps, aucs, path):
    n_classes = y_score.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1", "AP", "AUC"])
        for i in range(n_classes):
            y_bin_true = (y_true == i).astype(int)
            y_bin_pred = (y_pred == i).astype(int)
            support = int(y_bin_true.sum())
            tp = int(((y_bin_pred == 1) & (y_bin_true == 1)).sum())
            fp = int(((y_bin_pred == 1) & (y_bin_true == 0)).sum())
            fn = int(((y_bin_pred == 0) & (y_bin_true == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            w.writerow([
                names[i], support,
                f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
                f"{aps.get(i, float('nan')):.4f}",
                f"{aucs.get(i, float('nan')):.4f}",
            ])


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    print(f"Running inference on test set: {TEST_DIR}")
    y_true, y_score, names = collect_predictions(model, TEST_DIR)
    y_pred = y_score.argmax(axis=1)
    class_names = [names[i] for i in range(len(names))]

    print("\nClassification report:")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)
    (OUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, OUT_DIR / "confusion_matrix.png",
                   normalize=False, title="Confusion matrix (counts)")
    plot_confusion(cm, class_names, OUT_DIR / "confusion_matrix_normalized.png",
                   normalize=True, title="Confusion matrix (row-normalized)")

    aps = plot_pr_curves(y_true, y_score, names, OUT_DIR / "pr_curves.png")
    aucs = plot_roc_curves(y_true, y_score, names, OUT_DIR / "roc_curves.png")
    plot_f1_vs_threshold(y_true, y_score, names, OUT_DIR / "f1_vs_threshold.png")
    write_per_class_csv(y_true, y_pred, y_score, names, aps, aucs,
                        OUT_DIR / "per_class_metrics.csv")

    print(f"\nSaved metrics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
