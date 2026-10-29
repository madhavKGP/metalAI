"""Package the MetalAI app as a self-contained Hugging Face Space (free CPU hosting) and optionally push it.

Build only:   python -m metalai.deploy.build_space
Build + push: python -m metalai.deploy.build_space --push --repo-id <hf-username>/metalai
              (token read from .secrets/hf_token, or env HF_TOKEN; never printed)
Check build:  python -m metalai.deploy.build_space --check

Output folder: deploy_build/hf_space/  (git-ignored)
  app.py, README.md (Space config), requirements.txt (CPU torch), packages.txt (system libs),
  src/metalai/...   (only what inference needs), configs/app.yaml + model_card.json,
  models/m1/<kind>/<exp>/<weights>.pt (only the served models), examples/sample_XX.png
See decisions.md D-028.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from metalai.common.utils import CONFIGS, RESULTS, ROOT

OUT = ROOT / "deploy_build" / "hf_space"
SRC_FILES = [
    "metalai/__init__.py",
    "metalai/common/__init__.py", "metalai/common/utils.py", "metalai/common/metrics.py",
    "metalai/m1_surface/__init__.py", "metalai/m1_surface/preprocess.py", "metalai/m1_surface/cnn.py",
    "metalai/m1_surface/gradcam.py", "metalai/m1_surface/inference.py", "metalai/m1_surface/splits.py",
]


def _ver(mod: str, default: str) -> str:
    try:
        return __import__(mod).__version__.split("+")[0]
    except Exception:  # noqa: BLE001
        return default


def model_card() -> dict:
    """Numbers come ONLY from results/m1/*.csv (latest row per exp/model/fold)."""
    rows = []

    def latest(df, keys):
        return df.sort_values("timestamp").groupby(keys, as_index=False).tail(1)

    def pm(s):
        s = s.dropna() * 100
        return f"{s.mean():.1f}%" if len(s) == 1 else f"{s.mean():.2f} ± {s.std(ddof=1):.2f}% ({len(s)} folds)"

    f = RESULTS / "m1" / "classical_results.csv"
    if f.exists():
        c = latest(pd.read_csv(f), ["exp_id", "featset", "fold"])
        c = c[c.featset == "both"]
        if len(c):
            rows.append({"method": "LBP + GLCM + SVM", "task": "classify", "metric": "accuracy", "value": pm(c.accuracy)})
    f = RESULTS / "m1" / "cnn_results.csv"
    if f.exists():
        c = latest(pd.read_csv(f), ["exp_id", "model", "fold"])
        c = c[c.exp_id.isin(["E002", "E003"])]
        for m, sub in c.groupby("model"):
            name = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50"}.get(m, m)
            rows.append({"method": name, "task": "classify", "metric": "accuracy", "value": pm(sub.accuracy)})
    f = RESULTS / "m1" / "yolo_results.csv"
    if f.exists():
        y = latest(pd.read_csv(f), ["exp_id", "model", "fold"])
        y = y[y.exp_id == "E005"]
        if len(y):
            rows.append({"method": "YOLOv8n", "task": "detect", "metric": "mAP@0.5", "value": pm(y.mAP50)})
            rows.append({"method": "YOLOv8n", "task": "detect", "metric": "mAP@0.5:0.95", "value": pm(y.mAP50_95)})
    return {"data": "NEU surface defect database — 1,799 grayscale 200×200 images, 6 classes, stratified 5-fold CV",
            "rows": rows,
            "note": "Served models: EfficientNet-B0 (E003) and YOLOv8n (E005), fold-0 checkpoints; the sample gallery "
                    "is fold-0 test data, never seen in training."}


def build() -> Path:
    from metalai.m1_surface.inference import resolve_weights

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "configs").mkdir(parents=True)

    # app + examples
    shutil.copy2(ROOT / "app" / "app.py", OUT / "app.py")
    ex = ROOT / "app" / "examples"
    if not list(ex.glob("sample_*.png")):  # create the gallery via the app's own function
        spec = importlib.util.spec_from_file_location("metalai_app", ROOT / "app" / "app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    shutil.copytree(ex, OUT / "examples", ignore=shutil.ignore_patterns("answer_key.csv"))

    # source (inference-only subset)
    for rel in SRC_FILES:
        dst = OUT / "src" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / rel, dst)

    # configs + weights actually served
    sel = yaml.safe_load(open(CONFIGS / "app.yaml"))
    shutil.copy2(CONFIGS / "app.yaml", OUT / "configs" / "app.yaml")
    shutil.copy2(CONFIGS / "m1.yaml", OUT / "configs" / "m1.yaml")
    for kind in ("cnn", "yolo"):
        w = resolve_weights(kind, sel[kind])
        if w is None:
            raise SystemExit(f"weights for {kind} {sel[kind]} not found - train first / fix configs/app.yaml")
        dst = OUT / "models" / "m1" / kind / w.parent.name / w.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(w, dst)
        print(f"weights: {w.relative_to(ROOT)}  ({w.stat().st_size / 1e6:.1f} MB)")
    (OUT / "configs" / "model_card.json").write_text(json.dumps(model_card(), indent=2))

    # Space config
    torch_v, tv_v = _ver("torch", "2.6.0"), _ver("torchvision", "0.21.0")
    (OUT / "requirements.txt").write_text("\n".join([
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch_v}", f"torchvision=={tv_v}",
        f"ultralytics=={_ver('ultralytics', '8.3.0')}",
        "grad-cam", "opencv-python-headless", "pandas", "scikit-learn", "matplotlib", "pyyaml", ""]))
    (OUT / "packages.txt").write_text("libgl1\nlibglib2.0-0\n")
    (OUT / "README.md").write_text(f"""---
title: MetalAI Surface Defect Inspector
emoji: 🔩
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {_ver('gradio', '5.0.0')}
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Find, name and explain defects on hot-rolled steel
---

# MetalAI — Surface Defect Inspector
M.Tech project demo. EfficientNet-B0 classifier + YOLOv8n detector + Grad-CAM, trained on the NEU
surface defect database (6 classes). Tap a sample image to try it. Research prototype, not a certified tool.
""", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Built {OUT}  ({size:.1f} MB)")
    return OUT


def check() -> None:
    """Import the BUILT app in a fresh interpreter (only its own src/) and run inspect() on one sample."""
    code = (
        "import sys, glob, cv2; sys.path.insert(0, 'src'); import app; "
        "img = cv2.cvtColor(cv2.imread(sorted(glob.glob('examples/sample_*.png'))[0]), cv2.COLOR_BGR2RGB); "
        "v, det, cam, lab = app.inspect(img, 0.25); "
        "print('CHECK verdict ok:', 'mai-title' in v, '| det:', det is not None, '| cam:', cam is not None, "
        "'| top:', max(lab, key=lab.get) if lab else None)"
    )
    env = {**os.environ, "PYTHONPATH": str(OUT / "src")}
    r = subprocess.run([sys.executable, "-c", code], cwd=OUT, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    if r.returncode != 0 or "CHECK verdict ok: True" not in r.stdout:
        raise SystemExit("Built Space failed the check - see output above.")


def push(repo_id: str) -> None:
    from huggingface_hub import HfApi

    tok_file = ROOT / ".secrets" / "hf_token"
    token = os.environ.get("HF_TOKEN") or (tok_file.read_text().strip() if tok_file.exists() else None)
    if not token:
        raise SystemExit("No HF token: put it in .secrets/hf_token (git-ignored) or set $env:HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy MetalAI M1 (build_space.py)")
    print(f"Pushed. Space: https://huggingface.co/spaces/{repo_id}  (first build takes ~5-10 min)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default=None, help="<hf-username>/metalai")
    ap.add_argument("--check", action="store_true", help="only re-run the check on an existing build")
    a = ap.parse_args()
    if not a.check:
        build()
    check()
    if a.push:
        if not a.repo_id:
            raise SystemExit("--repo-id <hf-username>/metalai is required with --push")
        push(a.repo_id)


if __name__ == "__main__":
    main()
