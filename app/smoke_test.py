"""Headless app test (no browser needed): runs the same inspect() the Gradio button calls
on a few dataset images and saves what the UI would show.

Run:  python app\smoke_test.py
Out:  reports/m1/app_smoke/<image_id>_{det,cam}.png + app_smoke.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metalai_app", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)  # builds the UI + loads models, does NOT launch the server

from metalai.common.utils import PROC, REPORTS, save_json  # noqa: E402

OUT = REPORTS / "m1" / "app_smoke"
OUT.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(PROC / "neu" / "manifest.csv")
picks = man.groupby("label").head(1)
log = []
for r in picks.itertuples():
    rgb = cv2.cvtColor(cv2.imread(str(PROC / "neu" / r.path)), cv2.COLOR_BGR2RGB)
    _, det, cam, probs = app.inspect(rgb, 0.25)  # inspect() returns (verdict_html, det_img, cam, labels)
    if det is not None:
        cv2.imwrite(str(OUT / f"{r.image_id}_det.png"), cv2.cvtColor(det, cv2.COLOR_RGB2BGR))
    if cam is not None:
        cv2.imwrite(str(OUT / f"{r.image_id}_cam.png"), cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
    top = max(probs, key=probs.get) if probs else None
    log.append({"image_id": r.image_id, "true": r.label, "cnn_top": top,
                "cnn_conf": probs[top] if top else None, "text": probs})
    print(f"{r.image_id}: true={r.label} cnn={top}")
save_json(log, OUT / "app_smoke.json")
print(f"OK - outputs in {OUT}")
sys.exit(0 if all(x["cnn_top"] for x in log) else 1)
