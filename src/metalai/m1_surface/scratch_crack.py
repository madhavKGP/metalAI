"""Scratch vs crack from line SHAPE only (NEU scratches vs crazing). CPU only. decisions.md D-030.

Run:  METALAI_EXP=E008 python -m metalai.m1_surface.scratch_crack
Out:  results/m1/scratch_crack_results.csv        one row per (model, fold), plus the inversion check
      results/m1/scratch_crack_predictions.csv
      reports/m1/scratch_crack/feature_table.csv, summary.json, *.png
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from metalai.common.metrics import confusion, plot_confusion
from metalai.common.utils import PROC, REPORTS, RESULTS, append_result, load_config, save_json, start_experiment
from metalai.m1_surface.line_features import FEATURES, extract
from metalai.m1_surface.splits import load_folds

LABELS = {"scratches": 0, "crazing": 1}
NAMES = ["scratch", "crack (crazing)"]
OUT = REPORTS / "m1" / "scratch_crack"


def _read(path, invert=False):
    img = cv2.imread(str(PROC / "neu" / path), cv2.IMREAD_GRAYSCALE)
    return 255 - img if invert else img


def _feat(path, invert=False):
    f, _, pol = extract(_read(path, invert))
    return {**f, "polarity": pol}


def features(df, invert=False, n_jobs=-1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs)(delayed(_feat)(p, invert) for p in df.path)
    return pd.DataFrame(rows, index=df.index)


def models(seed):
    return {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed)}


def main() -> None:
    cfg = load_config("m1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    start_experiment("m1_scratch_crack", a, {"seed": cfg["seed"], "features": FEATURES})
    OUT.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(PROC / "neu" / "manifest.csv")[["image_id", "path"]]
    df = load_folds().merge(man, on="image_id")
    df = df[df.label.isin(LABELS)].reset_index(drop=True)
    df["y"] = df.label.map(LABELS)
    F = features(df, n_jobs=a.n_jobs)
    Fi = features(df, invert=True, n_jobs=a.n_jobs)          # every image bright <-> dark
    df = pd.concat([df, F], axis=1)
    X, Xi, y = F[FEATURES].values, Fi[FEATURES].values, df.y.values
    res_csv = RESULTS / "m1" / "scratch_crack_results.csv"

    preds, imps = [], []
    for name, make in models(cfg["seed"]).items():
        cm = np.zeros((2, 2), int)
        for k in sorted(df.fold.unique()):
            tr, te = (df.fold != k).values, (df.fold == k).values
            m = make().fit(X[tr], y[tr])
            p, pr = m.predict(X[te]), m.predict_proba(X[te])[:, 1]
            pi = m.predict(Xi[te])                            # same model, inverted test images
            cm += confusion(y[te], p, 2)
            append_result(res_csv, {"model": name, "fold": k, "n_test": int(te.sum()),
                                    "accuracy": accuracy_score(y[te], p), "macro_f1": f1_score(y[te], p, average="macro"),
                                    "auc": roc_auc_score(y[te], pr), "accuracy_inverted_test": accuracy_score(y[te], pi)})
            preds += [{"model": name, "image_id": i, "fold": k, "true": NAMES[t], "pred": NAMES[q], "p_crack": round(c, 4)}
                      for i, t, q, c in zip(df.image_id[te], y[te], p, pr)]
            if name == "random_forest":
                imps.append(m.feature_importances_)
        plot_confusion(cm, NAMES, OUT / f"confusion_{name}.png", f"Scratch vs crack, shape features + {name} (5 folds)")
    preds = pd.DataFrame(preds)
    preds.to_csv(RESULTS / "m1" / "scratch_crack_predictions.csv", index=False)

    table = []
    for j, f in enumerate(FEATURES):
        auc = roc_auc_score(y, df[f])
        table.append({"feature": f, "auc_alone": max(auc, 1 - auc), "scratch_median": df.loc[df.y == 0, f].median(),
                      "crack_median": df.loc[df.y == 1, f].median(), "rf_importance": float(np.mean(imps, axis=0)[j])})
    table = pd.DataFrame(table).sort_values("auc_alone", ascending=False)
    table.to_csv(OUT / "feature_table.csv", index=False)
    save_json({"n_images": len(df), "per_class": df.label.value_counts().to_dict(),
               "polarity_table": pd.crosstab(df.label, df.polarity).to_dict(),
               "polarity_only_accuracy": accuracy_score(y, (df.polarity == "dark").astype(int))},
              OUT / "summary.json")
    _boxplots(df, table.feature.tolist())
    _examples(df, preds, cfg["seed"])
    r = pd.read_csv(res_csv).sort_values("timestamp").groupby(["model", "fold"]).tail(1)
    print(r.groupby("model")[["accuracy", "macro_f1", "auc", "accuracy_inverted_test"]].agg(["mean", "std"]).round(4))


def _boxplots(df, order):
    fig, axes = plt.subplots(1, len(order), figsize=(2.3 * len(order), 3))
    for ax, f in zip(axes, order):
        ax.boxplot([df.loc[df.y == 0, f], df.loc[df.y == 1, f]], tick_labels=["scratch", "crack"], showfliers=False)
        ax.set_title(f, fontsize=9)
    fig.suptitle(f"Shape features: scratch vs crack (NEU, {len(df)} images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "feature_boxplots.png", dpi=150)
    plt.close(fig)


def _overlay(path):
    img = _read(path)
    _, m, _ = extract(img)
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    ov[skeletonize(m)] = (255, 0, 0)
    return ov


def _examples(df, preds, seed):
    rng = np.random.default_rng(seed)
    path = df.set_index("image_id").path
    pick = [rng.choice(df.image_id[df.y == c].values, 4, replace=False) for c in (0, 1)]
    wrong = preds[(preds.model == "random_forest") & (preds.true != preds.pred)].head(4)
    rows = 3 if len(wrong) else 2
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.7 * rows))
    for r, ids in enumerate(pick):
        for c, i in enumerate(ids):
            axes[r, c].imshow(_overlay(path[i]))
            axes[r, c].set_title(i, fontsize=8)
    for c, (_, w) in enumerate(wrong.iterrows()):
        axes[2, c].imshow(_overlay(path[w.image_id]))
        axes[2, c].set_title(f"{w.image_id}\nWRONG: called {w.pred}", fontsize=8, color="red")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Red = detected line skeleton. Row 1 scratches, row 2 cracks" + (", row 3 mistakes" if len(wrong) else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
