"""Leakage check: near-duplicate images that ended up in DIFFERENT folds (would inflate test scores).

Exact duplicates are already removed by prepare_data (MD5). This finds *near* duplicates
(re-saved / slightly shifted crops) with a 64-bit difference hash (dHash) + Hamming distance.

Run:  python -m metalai.m1_surface.near_dups [--max-dist 6]
Out:  reports/m1/leakage/near_dups.csv, near_dups_examples.png, summary.json
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metalai.common.utils import PROC, REPORTS, save_json
from metalai.m1_surface.splits import load_folds

OUT = REPORTS / "m1" / "leakage"


def dhash(img: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=int, default=6, help="Hamming distance (of 64 bits) counted as near-duplicate")
    a = ap.parse_args()
    man = pd.read_csv(PROC / "neu" / "manifest.csv").merge(load_folds()[["image_id", "fold"]], on="image_id")
    imgs = {r.image_id: cv2.imread(str(PROC / "neu" / r.path), cv2.IMREAD_GRAYSCALE) for r in man.itertuples()}
    ids = man.image_id.tolist()
    H = np.array([dhash(imgs[i]) for i in ids], dtype=np.uint64)
    bits = np.unpackbits(H.view(np.uint8).reshape(-1, 8), axis=1)  # (N, 64)
    fold = man.fold.to_numpy()
    label = man.label.to_numpy()
    rows = []
    for i in range(len(ids)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.where(d <= a.max_dist)[0] + i + 1:
            rows.append({"a": ids[i], "b": ids[j], "hamming": int(d[j - i - 1]),
                         "fold_a": int(fold[i]), "fold_b": int(fold[j]), "same_fold": bool(fold[i] == fold[j]),
                         "label_a": label[i], "label_b": label[j],
                         "pixel_mae": float(np.abs(imgs[ids[i]].astype(int) - imgs[ids[j]].astype(int)).mean())})
    df = pd.DataFrame(rows, columns=["a", "b", "hamming", "fold_a", "fold_b", "same_fold", "label_a", "label_b",
                                     "pixel_mae"]).sort_values(["hamming", "pixel_mae"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "near_dups.csv", index=False)
    cross = df[~df.same_fold]
    summary = {"max_dist": a.max_dist, "pairs_total": len(df), "pairs_cross_fold": len(cross),
               "pairs_cross_fold_pixel_mae_lt_5": int((cross.pixel_mae < 5).sum()),
               "images_in_cross_fold_pairs": int(len(set(cross.a) | set(cross.b))),
               "pairs_with_different_labels": int((df.label_a != df.label_b).sum())}
    save_json(summary, OUT / "summary.json")
    print(summary)

    show = df.head(8)
    if len(show):
        fig, axes = plt.subplots(len(show), 2, figsize=(4.5, 2.2 * len(show)), squeeze=False)
        for r, row in enumerate(show.itertuples()):
            for c, (i, f) in enumerate([(row.a, row.fold_a), (row.b, row.fold_b)]):
                axes[r, c].imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
                axes[r, c].set_title(f"{i} (fold {f})\nham {row.hamming}, mae {row.pixel_mae:.1f}", fontsize=7)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "near_dups_examples.png", dpi=110)
        plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
