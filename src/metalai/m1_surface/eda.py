"""Exploratory data analysis figures for NEU.

Run:  python -m metalai.m1_surface.eda
Out:  reports/m1/eda/*.png + eda_summary.json
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metalai.common.utils import PROC, REPORTS, save_json
from metalai.m1_surface import CLASSES

OUT = REPORTS / "m1" / "eda"


def main(n_per_class: int = 6, seed: int = 42) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(PROC / "neu" / "manifest.csv")
    boxes = pd.read_csv(PROC / "neu" / "boxes.csv")
    rng = np.random.default_rng(seed)

    # 1) sample grid with boxes
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(2 * n_per_class, 2.1 * len(CLASSES)))
    for r, c in enumerate(CLASSES):
        sub = man[man.label == c]
        pick = sub.sample(min(n_per_class, len(sub)), random_state=seed)
        for j in range(n_per_class):
            ax = axes[r, j]
            ax.axis("off")
            if j >= len(pick):
                continue
            row = pick.iloc[j]
            img = cv2.imread(str(PROC / "neu" / row.path), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            for _, b in boxes[boxes.image_id == row.image_id].iterrows():
                ax.add_patch(plt.Rectangle((b.xmin, b.ymin), b.xmax - b.xmin, b.ymax - b.ymin,
                                           fill=False, edgecolor="red", linewidth=1))
            if j == 0:
                ax.set_title(c, fontsize=9, loc="left")
    fig.suptitle("NEU samples per class (red = ground-truth boxes)")
    fig.tight_layout()
    fig.savefig(OUT / "samples_grid.png", dpi=110)
    plt.close(fig)

    # 2) class counts
    counts = man.label.value_counts().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("images")
    ax.set_title("Images per class")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "class_counts.png", dpi=110)
    plt.close(fig)

    # 3) intensity statistics per class
    stats = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in CLASSES:
        sub = man[man.label == c]
        sub = sub.sample(min(100, len(sub)), random_state=seed)
        hist = np.zeros(256)
        means, stds = [], []
        for p in sub.path:
            img = cv2.imread(str(PROC / "neu" / p), cv2.IMREAD_GRAYSCALE)
            hist += np.bincount(img.ravel(), minlength=256)
            means.append(img.mean())
            stds.append(img.std())
        ax.plot(hist / hist.sum(), label=c)
        stats.append({"class": c, "mean_intensity": float(np.mean(means)), "mean_std": float(np.mean(stds))})
    ax.set_xlabel("grey level")
    ax.set_ylabel("fraction of pixels")
    ax.legend(fontsize=8)
    ax.set_title("Grey-level distribution per class (100 imgs/class)")
    fig.tight_layout()
    fig.savefig(OUT / "intensity_hist.png", dpi=110)
    plt.close(fig)

    # 4) box statistics
    box_stats = {}
    if len(boxes):
        b = boxes.merge(man[["image_id", "width", "height"]], on="image_id")
        b["rel_area"] = (b.xmax - b.xmin) * (b.ymax - b.ymin) / (b.width * b.height)
        b["aspect"] = (b.xmax - b.xmin) / (b.ymax - b.ymin)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        per_img = man.set_index("label").n_boxes
        axes[0].boxplot([np.atleast_1d(per_img.loc[c].values) for c in CLASSES])
        axes[0].set_title("boxes per image")
        axes[1].boxplot([b[b.label == c].rel_area.values for c in CLASSES])
        axes[1].set_title("box area / image area")
        axes[2].boxplot([np.log2(b[b.label == c].aspect.values) for c in CLASSES])
        axes[2].set_title("log2(box aspect w/h)")
        for ax in axes:
            ax.set_xticks(range(1, len(CLASSES) + 1))
            ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "box_stats.png", dpi=110)
        plt.close(fig)
        for c in CLASSES:
            s = b[b.label == c]
            box_stats[c] = {"n_boxes": int(len(s)),
                            "boxes_per_image_mean": float(per_img.loc[c].mean()),
                            "rel_area_median": float(s.rel_area.median()) if len(s) else None,
                            "rel_area_min": float(s.rel_area.min()) if len(s) else None}

    save_json({"intensity": stats, "boxes": box_stats}, OUT / "eda_summary.json")
    print(f"Saved EDA figures to {OUT}")


if __name__ == "__main__":
    main()
