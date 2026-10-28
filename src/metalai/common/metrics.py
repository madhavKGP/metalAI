"""Classification metrics + confusion-matrix plotting shared by all M1 models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def cls_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    for name, v in zip(class_names, per_class):
        out[f"f1_{name}"] = float(v)
    return out


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
