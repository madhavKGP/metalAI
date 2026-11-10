"""Corrosion condition from surface photos: colour + texture features -> SVM. CPU only. decisions.md D-031.

PUBLIC data, not ours: Jackson (2025) steel corrosion dataset, MIT licence,
https://github.com/bennett-jackson/steel-corrosion-dataset (3,417 bridge-inspection photos, 512x512).
Classes used: good / fair / poor / severe (condition states). "Weathering" (weathering-steel patina)
is a different steel type, not a condition grade, so it is left out.

Two tasks
  4class : good vs fair vs poor vs severe
  binary : not corroded (good+fair) vs corroded (poor+severe)

Leakage guard: file names are camera photo numbers, so near-consecutive numbers are usually the same
bridge / visit. We group numbers with gaps <= GROUP_GAP and use StratifiedGroupKFold, so photos of the
same structure never sit in both train and test. A plain random split is also reported to show the gap.

Run:  METALAI_EXP=E009 python -m metalai.m1_surface.corrosion_baseline
Out:  results/m1/corrosion_results.csv, reports/m1/corrosion/*.png, summary.json
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metalai.common.metrics import confusion, plot_confusion
from metalai.common.utils import PROC, RAW, REPORTS, RESULTS, append_result, load_config, save_json, start_experiment
from metalai.m1_surface.classical import glcm_features, lbp_features

SRC = RAW / "corrosion_jackson"
FOLDERS = {"good": "ResizedIGoodmages", "fair": "ResizedFairImages",
           "poor": "ResizedPoorImages", "severe": "ResizedSevereImages"}
CONDITION = ["good", "fair", "poor", "severe"]
GROUP_GAP = 20
OUT = REPORTS / "m1" / "corrosion"


def colour_features(bgr: np.ndarray) -> np.ndarray:
    """Rust is orange/brown: hue-saturation histogram + simple colour stats."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256]).ravel()
    h = h / (h.sum() + 1e-9)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    stats = np.concatenate([lab.mean(0), lab.std(0)])
    rust = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 80) & (hsv[..., 2] > 40)).mean()
    return np.concatenate([h, stats, [rust]])


def one(path: str, cfg: dict) -> np.ndarray:
    bgr = cv2.resize(cv2.imread(path), (256, 256), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([colour_features(bgr), lbp_features(g, cfg["lbp_scales"]),
                           glcm_features(g, cfg["glcm_distances"], cfg["glcm_levels"])])


def manifest() -> tuple[pd.DataFrame, dict]:
    """Exact duplicates (MD5) are dropped: one copy kept if all copies share a label, ALL copies dropped
    if the same photo appears under different condition labels (label conflict)."""
    import hashlib
    import re

    rows = [{"image_id": p.stem, "label": c, "path": str(p), "md5": hashlib.md5(p.read_bytes()).hexdigest()}
            for c, f in FOLDERS.items() for p in sorted((SRC / f).glob("*.JPG"))]
    df = pd.DataFrame(rows)
    n_raw = len(df)
    labels_per_md5 = df.groupby("md5").label.nunique()
    conflict = labels_per_md5[labels_per_md5 > 1].index
    n_conflict_files = int(df.md5.isin(conflict).sum())
    df = df[~df.md5.isin(conflict)].drop_duplicates("md5").reset_index(drop=True)
    info = {"n_files": n_raw, "n_label_conflict_photos": len(conflict), "n_label_conflict_files_dropped": n_conflict_files,
            "n_exact_duplicates_dropped": n_raw - n_conflict_files - len(df), "n_kept": len(df)}
    num = df.image_id.map(lambda s: int(re.match(r"\d+", s).group()))   # '2712924 2' -> 2712924
    order = num.sort_values()
    grp = (order.diff().fillna(GROUP_GAP + 1) > GROUP_GAP).cumsum()
    df["group"] = grp.reindex(df.index)
    return df, info


def run_cv(X, y, groups, splitter, cfg, seed, n_jobs):
    pred = np.empty_like(y)
    folds = []
    split = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for k, (tr, te) in enumerate(split):
        grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
                            {"svc__C": cfg["svm_C"], "svc__gamma": cfg["svm_gamma"]},
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="f1_macro", n_jobs=n_jobs)
        grid.fit(X[tr], y[tr])
        pred[te] = grid.predict(X[te])
        folds.append((k, te, grid.best_params_))
    return pred, folds


def main() -> None:
    cfg_all = load_config("m1")
    cfg, seed = cfg_all["classical"], cfg_all["seed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    start_experiment("m1_corrosion_baseline", a, {"classical": cfg, "group_gap": GROUP_GAP})
    OUT.mkdir(parents=True, exist_ok=True)

    df, dup_info = manifest()
    cache = PROC / "corrosion_jackson_features.npz"
    if cache.exists() and list(np.load(cache, allow_pickle=True)["ids"]) == df.image_id.tolist():
        X = np.load(cache, allow_pickle=True)["X"]
    else:
        t = time.time()
        X = np.stack(Parallel(n_jobs=a.n_jobs)(delayed(one)(p, cfg) for p in df.path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(df.image_id, dtype=object), X=X)
        print(f"features {X.shape} in {time.time() - t:.0f}s")

    tasks = {"4class": (df.label.map({c: i for i, c in enumerate(CONDITION)}).values, CONDITION),
             "binary": (df.label.isin(["poor", "severe"]).astype(int).values, ["not corroded", "corroded"])}
    res_csv = RESULTS / "m1" / "corrosion_results.csv"
    summary = {**dup_info, "n_images": len(df), "per_class": df.label.value_counts().to_dict(), "n_groups": int(df.group.nunique())}
    for task, (y, names) in tasks.items():
        for split_name, splitter, groups in [
            ("grouped", StratifiedGroupKFold(5, shuffle=True, random_state=seed), df.group.values),
            ("random", StratifiedKFold(5, shuffle=True, random_state=seed), None),
        ]:
            pred, folds = run_cv(X, y, groups, splitter, cfg, seed, a.n_jobs)
            for k, te, bp in folds:
                append_result(res_csv, {"task": task, "split": split_name, "fold": k, "n_test": len(te),
                                        "accuracy": accuracy_score(y[te], pred[te]),
                                        "macro_f1": f1_score(y[te], pred[te], average="macro"),
                                        "best_C": bp["svc__C"], "best_gamma": bp["svc__gamma"]})
            accs = [accuracy_score(y[te], pred[te]) for _, te, _ in folds]
            f1s = [f1_score(y[te], pred[te], average="macro") for _, te, _ in folds]
            summary[f"{task}_{split_name}"] = {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs, ddof=1)),
                                               "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s, ddof=1)),
                                               "majority_baseline": float(np.bincount(y).max() / len(y))}
            if split_name == "grouped":
                plot_confusion(confusion(y, pred, len(names)), names, OUT / f"confusion_{task}.png",
                               f"Corrosion condition ({task}), colour+texture SVM, grouped 5-fold")
            print(task, split_name, {k: round(v, 4) for k, v in summary[f"{task}_{split_name}"].items()})
    save_json(summary, OUT / "summary.json")


if __name__ == "__main__":
    main()
