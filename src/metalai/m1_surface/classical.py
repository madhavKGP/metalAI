"""Classical texture baseline: LBP + GLCM features -> SVM (decisions.md D-008). CPU only.

Run:  python -m metalai.m1_surface.classical            # all 3 feature sets, 5 folds
      python -m metalai.m1_surface.classical --featsets both
Out:  results/m1/classical_results.csv            (one row per featset x fold)
      results/m1/classical_predictions_<fs>.csv
      reports/m1/classical/confusion_<fs>.png, summary.json
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metalai.common.metrics import cls_metrics, confusion, plot_confusion
from metalai.common.utils import PROC, REPORTS, RESULTS, append_result, load_config, save_json, start_experiment
from metalai.m1_surface import CLASSES
from metalai.m1_surface.splits import get_fold, load_folds

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def lbp_features(img: np.ndarray, scales) -> np.ndarray:
    feats = []
    for P, R in scales:
        lbp = local_binary_pattern(img, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(P + 3), density=True)
        feats.append(hist)
    return np.concatenate(feats)


def glcm_features(img: np.ndarray, distances, levels: int) -> np.ndarray:
    q = (img.astype(np.float32) * levels / 256).astype(np.uint8)
    g = graycomatrix(q, distances=distances, angles=ANGLES, levels=levels, symmetric=True, normed=True)
    feats = []
    for p in GLCM_PROPS:
        v = graycoprops(g, p)  # (n_dist, n_angles)
        feats.append(v.mean(axis=1))                 # rotation-averaged value
        feats.append(v.max(axis=1) - v.min(axis=1))  # anisotropy (range over angles)
    return np.concatenate(feats)


def extract_one(path, cfg):
    img = cv2.imread(str(PROC / "neu" / path), cv2.IMREAD_GRAYSCALE)
    return lbp_features(img, cfg["lbp_scales"]), glcm_features(img, cfg["glcm_distances"], cfg["glcm_levels"])


def get_features(cfg, n_jobs: int = -1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cache = PROC / "neu" / "features_classical.npz"
    man = pd.read_csv(PROC / "neu" / "manifest.csv").set_index("image_id")
    folds = load_folds()
    ids = folds.image_id.tolist()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == ids:
            print("Loaded cached features")
            return folds, z["lbp"], z["glcm"]
    t = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(extract_one)(man.loc[i, "path"], cfg) for i in ids)
    lbp = np.stack([o[0] for o in out])
    glcm = np.stack([o[1] for o in out])
    np.savez(cache, ids=np.array(ids, dtype=object), lbp=lbp, glcm=glcm)
    print(f"Extracted features for {len(ids)} images in {time.time() - t:.1f}s "
          f"(LBP dim {lbp.shape[1]}, GLCM dim {glcm.shape[1]})")
    return folds, lbp, glcm


def main() -> None:
    cfg_all = load_config("m1")
    cfg = cfg_all["classical"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--featsets", nargs="+", default=["lbp", "glcm", "both"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(cfg_all["n_folds"])))
    ap.add_argument("--n-jobs", type=int, default=-1, help="CPU cores (use ~half when a GPU job runs in parallel)")
    a = ap.parse_args()
    start_experiment("m1_classical", a, cfg_all)

    folds, lbp, glcm = get_features(cfg, a.n_jobs)
    id_to_row = {i: k for k, i in enumerate(folds.image_id)}
    y_all = folds.label_idx.values
    feats = {"lbp": lbp, "glcm": glcm, "both": np.hstack([lbp, glcm])}
    out_dir = REPORTS / "m1" / "classical"
    summary = {}

    for fs in a.featsets:
        X_all = feats[fs]
        cm_total = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        preds = []
        for k in a.folds:
            tr, va, te = get_fold(k, cfg_all["val_frac"], cfg_all["seed"])
            # SVM tunes with inner CV on train+val (no separate early stopping needed).
            trv = pd.concat([tr, va])
            itr = [id_to_row[i] for i in trv.image_id]
            ite = [id_to_row[i] for i in te.image_id]
            pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False))
            grid = GridSearchCV(
                pipe, {"svc__C": cfg["svm_C"], "svc__gamma": cfg["svm_gamma"]},
                cv=StratifiedKFold(cfg["inner_cv"], shuffle=True, random_state=cfg_all["seed"]),
                scoring="f1_macro", n_jobs=a.n_jobs,
            )
            t = time.time()
            grid.fit(X_all[itr], y_all[itr])
            fit_s = time.time() - t
            t = time.time()
            yp = grid.predict(X_all[ite])
            ms_per_img = (time.time() - t) / len(ite) * 1000
            m = cls_metrics(y_all[ite], yp, CLASSES)
            cm_total += confusion(y_all[ite], yp, len(CLASSES))
            append_result(RESULTS / "m1" / "classical_results.csv", {
                "featset": fs, "fold": k, "n_train": len(itr), "n_test": len(ite),
                "best_C": grid.best_params_["svc__C"], "best_gamma": grid.best_params_["svc__gamma"],
                "fit_seconds": round(fit_s, 1), "ms_per_img_svm_only": round(ms_per_img, 3), **m,
            })
            preds += [{"image_id": i, "fold": k, "true": CLASSES[t_], "pred": CLASSES[p_]}
                      for i, t_, p_ in zip(te.image_id, y_all[ite], yp)]
            print(f"[{fs}] fold {k}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                  f"best={grid.best_params_}")
        pd.DataFrame(preds).to_csv(RESULTS / "m1" / f"classical_predictions_{fs}.csv", index=False)
        plot_confusion(cm_total, CLASSES, out_dir / f"confusion_{fs}.png",
                       f"LBP/GLCM+SVM ({fs}) - summed over folds")
        res = pd.read_csv(RESULTS / "m1" / "classical_results.csv")
        r = res[(res.featset == fs) & (res.fold.isin(a.folds))].groupby("fold").tail(1)
        summary[fs] = {"accuracy_mean": r.accuracy.mean(), "accuracy_std": r.accuracy.std(),
                       "macro_f1_mean": r.macro_f1.mean(), "macro_f1_std": r.macro_f1.std(),
                       "confusion_summed": cm_total}
        print(f"==> {fs}: acc {r.accuracy.mean():.4f} +/- {r.accuracy.std():.4f}")
    save_json(summary, out_dir / "summary.json")


if __name__ == "__main__":
    main()
