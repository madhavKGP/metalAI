"""Grad-CAM grids for trained CNN folds (raw heatmaps, no contrast tricks - decisions.md D-009).

Run:  python -m metalai.m1_surface.gradcam --exp E002 --model resnet50 --fold 0
Out:  reports/m1/gradcam/<exp>_<model>_fold<k>_correct.png   (3 correct per class)
      reports/m1/gradcam/<exp>_<model>_fold<k>_errors.png    (misclassified, up to 24)
      reports/m1/gradcam/<exp>_<model>_fold<k>_boxes.png     (heatmap + ground-truth boxes, 2 per class)
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from metalai.common.utils import PROC, REPORTS, current_exp
from metalai.m1_surface import CLASSES, CLASS_TO_IDX
from metalai.m1_surface.cnn import gradcam_target_layer, load_trained, preds_path
from metalai.m1_surface.preprocess import build_transform, display_rgb

OUT = REPORTS / "m1" / "gradcam"


class CamRunner:
    def __init__(self, model_name: str, tag: str, device: str, exp: str | None = None):
        self.model = load_trained(model_name, tag, device, exp)
        self.cam = GradCAM(model=self.model, target_layers=[gradcam_target_layer(self.model, model_name)])
        self.tf = build_transform(False)
        self.device = device

    def __call__(self, gray: np.ndarray, target_idx: int | None = None):
        x = self.tf(gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0].cpu().numpy()
        t = int(p.argmax()) if target_idx is None else target_idx
        heat = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(t)])[0]
        overlay = show_cam_on_image(display_rgb(gray), heat, use_rgb=True)
        return overlay, heat, p


def _grid(items, path, title, ncols=6):
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, ttl, boxes) in zip(axes.ravel(), items):
        ax.imshow(img)
        for (x0, y0, x1, y1) in boxes:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.2))
        ax.set_title(ttl, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="experiment id of the trained model (default: $env:METALAI_EXP)")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=3)
    a = ap.parse_args()
    exp = a.exp or current_exp()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"fold{a.fold}"
    run = CamRunner(a.model, tag, device, exp)
    a.model_tag = f"{exp}_{a.model}"

    man = pd.read_csv(PROC / "neu" / "manifest.csv").set_index("image_id")
    boxes = pd.read_csv(PROC / "neu" / "boxes.csv")
    preds = pd.read_csv(preds_path(exp, a.model))
    preds = preds[preds.fold == a.fold]
    if preds.empty:
        raise SystemExit("No predictions for this fold - train it first.")

    def load(i):
        return cv2.imread(str(PROC / "neu" / man.loc[i, "path"]), cv2.IMREAD_GRAYSCALE)

    def scaled_boxes(i, size=224):
        h, w = man.loc[i, "height"], man.loc[i, "width"]
        b = boxes[boxes.image_id == i]
        return [(r.xmin * size / w, r.ymin * size / h, r.xmax * size / w, r.ymax * size / h) for r in b.itertuples()]

    correct = []
    for c in CLASSES:
        sub = preds[(preds.true == c) & (preds.pred == c)].sort_values("conf", ascending=False)
        for r in sub.head(a.per_class).itertuples():
            ov, _, p = run(load(r.image_id))
            correct.append((ov, f"{r.image_id}\npred {r.pred} {p.max():.2f}", []))
    _grid(correct, OUT / f"{a.model_tag}_{tag}_correct.png", f"{a.model_tag} {tag}: correct (highest confidence)",
          ncols=a.per_class * 2)

    errs = []
    for r in preds[preds.true != preds.pred].head(24).itertuples():
        ov, _, p = run(load(r.image_id))
        errs.append((ov, f"{r.image_id}\ntrue {r.true}\npred {r.pred} {p.max():.2f}", []))
    if errs:
        _grid(errs, OUT / f"{a.model_tag}_{tag}_errors.png", f"{a.model_tag} {tag}: misclassified ({len(errs)} shown)")
    else:
        print("No misclassified test images in this fold.")

    withbox = []
    for c in CLASSES:
        sub = preds[(preds.true == c)].sample(min(2, (preds.true == c).sum()), random_state=0)
        for r in sub.itertuples():
            ov, _, p = run(load(r.image_id), CLASS_TO_IDX[r.true])
            withbox.append((ov, f"{r.image_id}\ncam for true class", scaled_boxes(r.image_id)))
    _grid(withbox, OUT / f"{a.model_tag}_{tag}_boxes.png",
          f"{a.model_tag} {tag}: Grad-CAM (true class) vs ground-truth boxes (green)", ncols=6)
    print(f"Saved Grad-CAM figures to {OUT}")


if __name__ == "__main__":
    main()
