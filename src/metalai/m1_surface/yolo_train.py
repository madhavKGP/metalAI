"""Train + test a YOLO detector on NEU-DET folds. GPU only.

Run:
  python -m metalai.m1_surface.yolo_train --folds 0                       # first fold
  python -m metalai.m1_surface.yolo_train --folds 1 2 3 4 --skip-existing
  python -m metalai.m1_surface.yolo_train --full                          # final model for the app
Out:
  runs/m1_yolo/<exp>_<model>_fold<k>/ ...           (Ultralytics run dir)
  models/m1/yolo/<exp>/<model>_fold<k>.pt          (best weights copy)
  results/m1/yolo_results.csv      (test-fold metrics; model selected on the inner val split only)
  reports/m1/yolo/<exp>_<model>_fold<k>/*.png  (curves, PR, confusion, prediction samples)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from metalai.common.utils import (MODELS, PROC, REPORTS, RESULTS, RUNS, append_result, load_config, require_cuda,
                                  save_json, start_experiment)
from metalai.m1_surface import CLASSES

YDIR = PROC / "neu_yolo"
WDIR = MODELS / "m1" / "yolo"


def copy_pngs(src: Path, dst: Path, prefix: str = ""):
    dst.mkdir(parents=True, exist_ok=True)
    for p in list(src.glob("*.png")) + list(src.glob("*.jpg")):
        shutil.copy2(p, dst / f"{prefix}{p.name}")


def main() -> None:
    cfg_all = load_config("m1")
    cfg = dict(cfg_all["yolo"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default=cfg["model"], help="yolov8n.pt | yolov8s.pt | yolo11n.pt ...")
    ap.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--patience", type=int, default=cfg["patience"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--workers", type=int, default=cfg["workers"])
    ap.add_argument("--cache", default=str(cfg.get("cache", "false")),
                    help="false | disk | ram. 'ram' crashed the 16 GB laptop (Windows workers copy the cache) - D-024")
    ap.add_argument("--resume", action="store_true",
                    help="continue a crashed/interrupted run from runs/.../weights/last.pt (same exp id + flags)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    cache = {"false": False, "none": False, "disk": "disk", "ram": "ram", "true": "ram"}[str(a.cache).lower()]
    device = require_cuda(a.allow_cpu)
    dev = 0 if device == "cuda" else "cpu"
    exp = start_experiment(f"m1_yolo_{Path(a.model).stem}", a, cfg_all)

    from ultralytics import YOLO

    tag_model = Path(a.model).stem
    folds = [-1] if a.full else a.folds
    for k in folds:
        ftag = "full" if k == -1 else f"fold{k}"
        name = f"{tag_model}_{ftag}"
        wpath = WDIR / exp / f"{name}.pt"
        run_name = f"{exp}_{name}"
        if a.skip_existing and wpath.exists():
            print(f"skip {name}")
            continue
        data_yaml = YDIR / f"{ftag}.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"{data_yaml} missing - run: python -m metalai.m1_surface.yolo_data")
        t0 = time.time()
        run_dir = RUNS / "m1_yolo" / run_name
        last = run_dir / "weights" / "last.pt"
        if a.resume and last.exists():
            import torch

            ck = torch.load(last, map_location="cpu", weights_only=False)
            ta = ck.get("train_args") or {}
            if ck.get("epoch", -1) == -1 or not ta.get("data"):
                # Ultralytics strips finished checkpoints (epoch=-1). Resuming those silently falls back to
                # default args (coco8!) - so: don't train, just evaluate the existing best.pt below.
                print(f"[resume] {last} is a finished run -> skipping training, evaluating existing weights")
                if not (run_dir / "weights" / "best.pt").exists():
                    raise SystemExit("finished run without best.pt - re-run without --resume")
            else:
                print(f"[resume] continuing {last} from epoch {ck.get('epoch')} (data={ta.get('data')})")
                YOLO(str(last)).train(resume=True)
        else:
            model = YOLO(a.model)
            model.train(data=str(data_yaml), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch,
                        device=dev, workers=a.workers, seed=cfg_all["seed"], deterministic=True, amp=True,
                        cache=cache, project=str(RUNS / "m1_yolo"), name=run_name, exist_ok=True, plots=True,
                        verbose=False)
        train_min = (time.time() - t0) / 60
        try:  # true total training time across resumes = Ultralytics' cumulative 'time' column (seconds)
            import pandas as pd

            rc = pd.read_csv(run_dir / "results.csv")
            rc.columns = [c.strip() for c in rc.columns]
            if "time" in rc:
                train_min = float(rc["time"].iloc[-1]) / 60
        except Exception:  # noqa: BLE001
            pass
        wpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "weights" / "best.pt", wpath)
        rep = REPORTS / "m1" / "yolo" / run_name
        copy_pngs(run_dir, rep, "train_")
        if (run_dir / "results.csv").exists():
            shutil.copy2(run_dir / "results.csv", rep / "train_results.csv")
        if k == -1:
            append_result(RESULTS / "m1" / "yolo_full_models.csv", {"model": tag_model, "train_minutes": round(train_min, 1)})
            continue

        best = YOLO(str(wpath))
        # inner-validation metrics: used for MODEL SELECTION between configs (never the test fold)
        mv = best.val(data=str(data_yaml), split="val", imgsz=a.imgsz, batch=a.batch, device=dev, plots=False,
                      workers=a.workers, project=str(RUNS / "m1_yolo"), name=f"{run_name}_val", exist_ok=True,
                      verbose=False)
        best_epoch = None
        if (run_dir / "results.csv").exists():
            import pandas as pd

            rc = pd.read_csv(run_dir / "results.csv")
            rc.columns = [c.strip() for c in rc.columns]
            col = "metrics/mAP50-95(B)"
            if col in rc:
                best_epoch = int(rc.loc[rc[col].idxmax(), "epoch"])
        m = best.val(data=str(data_yaml), split="test", imgsz=a.imgsz, batch=a.batch, device=dev, plots=True,
                     workers=a.workers, project=str(RUNS / "m1_yolo"), name=f"{run_name}_test", exist_ok=True,
                     verbose=False)
        copy_pngs(RUNS / "m1_yolo" / f"{run_name}_test", rep, "test_")
        row = {"model": tag_model, "fold": k, "imgsz": a.imgsz, "batch": a.batch, "epochs_max": a.epochs,
               "patience": a.patience, "cache": str(cache), "resumed": bool(a.resume),
               "train_minutes": round(train_min, 1), "best_epoch": best_epoch,
               "val_mAP50": float(mv.box.map50), "val_mAP50_95": float(mv.box.map),
               "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr),
               "ms_inference": float(m.speed.get("inference", float("nan")))}
        ap50 = dict(zip([int(i) for i in m.box.ap_class_index], [float(v) for v in m.box.ap50]))
        for i, c in enumerate(CLASSES):
            row[f"AP50_{c}"] = ap50.get(i, float("nan"))
        append_result(RESULTS / "m1" / "yolo_results.csv", row)
        save_json(row, rep / "test_metrics.json")
        print(f"==> {run_name}: test mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} ({row['train_minutes']} min)")


if __name__ == "__main__":
    main()
