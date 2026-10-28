"""Inference used by the web app. Models see ONLY pixels (decisions.md D-010).
Which weights are served is set in configs/app.yaml (falls back to the newest weights on disk)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from metalai.common.utils import MODELS, load_config
from metalai.m1_surface import CLASSES

UNCERTAIN_BELOW = 0.5


def resolve_weights(kind: str, sel: dict) -> Path | None:
    """kind = 'cnn' | 'yolo'. Weights live at models/m1/<kind>/<exp>/<model>_<tag>.pt"""
    base = MODELS / "m1" / kind
    if sel.get("exp") and sel.get("tag"):
        p = base / sel["exp"] / f"{sel['model']}_{sel['tag']}.pt"
        return p if p.exists() else None
    cands = sorted(base.glob(f"*/{sel['model']}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    full = [p for p in cands if p.stem.endswith("_full")]
    return (full or cands or [None])[0]


class SurfaceInspector:
    def __init__(self):
        try:
            sel = load_config("app")
        except FileNotFoundError:
            sel = {"cnn": {"model": "resnet50"}, "yolo": {"model": "yolov8n"}}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn_path = resolve_weights("cnn", sel["cnn"])
        self.cam = None
        if self.cnn_path:
            from metalai.m1_surface.gradcam import CamRunner

            exp, tag = self.cnn_path.parent.name, self.cnn_path.stem.split("_")[-1]
            self.cam = CamRunner(sel["cnn"]["model"], tag, self.device, exp)
        self.yolo_path = resolve_weights("yolo", sel["yolo"])
        self.yolo = None
        if self.yolo_path:
            from ultralytics import YOLO

            self.yolo = YOLO(str(self.yolo_path))

    def describe(self, p: Path | None) -> str:
        return f"{p.parent.name}/{p.name}" if p else "none"

    def classify(self, gray: np.ndarray):
        if self.cam is None:
            return None, None, None
        t = time.perf_counter()
        overlay, _, p = self.cam(gray)
        return {c: float(v) for c, v in zip(CLASSES, p)}, overlay, (time.perf_counter() - t) * 1000

    def detect(self, gray: np.ndarray, conf: float = 0.25):
        if self.yolo is None:
            return None, [], None
        rgb = np.repeat(gray[..., None], 3, axis=2)
        t = time.perf_counter()
        r = self.yolo.predict(rgb, conf=conf, verbose=False, device=0 if self.device == "cuda" else "cpu")[0]
        ms = (time.perf_counter() - t) * 1000
        dets = [{"class": CLASSES[int(b.cls)], "conf": float(b.conf), "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()]}
                for b in r.boxes]
        return r.plot()[..., ::-1], dets, ms  # plot() returns BGR
