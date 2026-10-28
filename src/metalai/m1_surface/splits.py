"""Fixed stratified 5-fold split shared by EVERY Module-1 model (decisions.md D-005).

Run:  python -m metalai.m1_surface.splits
Out:  data/splits/neu_folds.csv  (image_id, label, label_idx, fold)   <- commit this file

For fold k:  test = fold k;  the other folds are split into train / val (val_frac, stratified,
seed + k). Use get_fold() everywhere so classical, CNN and YOLO see identical data.
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from metalai.common.utils import PROC, SPLITS, load_config

FOLDS_CSV = SPLITS / "neu_folds.csv"


def make_folds(n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    man = pd.read_csv(PROC / "neu" / "manifest.csv")
    man = man.sort_values("image_id").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    man["fold"] = -1
    for k, (_, test_idx) in enumerate(skf.split(man, man["label_idx"])):
        man.loc[test_idx, "fold"] = k
    out = man[["image_id", "label", "label_idx", "fold"]]
    SPLITS.mkdir(parents=True, exist_ok=True)
    out.to_csv(FOLDS_CSV, index=False)
    return out


def load_folds() -> pd.DataFrame:
    if not FOLDS_CSV.exists():
        raise SystemExit(f"{FOLDS_CSV} missing - run: python -m metalai.m1_surface.splits")
    return pd.read_csv(FOLDS_CSV)


def get_fold(fold: int, val_frac: float = 0.10, seed: int = 42):
    """Return (train_df, val_df, test_df) for a given test fold. fold=-1 -> 'full' (no test)."""
    df = load_folds()
    if fold == -1:
        rest, test = df, df.iloc[0:0]
    else:
        test = df[df.fold == fold]
        rest = df[df.fold != fold]
    tr, va = train_test_split(rest, test_size=val_frac, stratify=rest["label_idx"], random_state=seed + max(fold, 0))
    return tr.reset_index(drop=True), va.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    cfg = load_config("m1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=cfg["n_folds"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    a = ap.parse_args()
    df = make_folds(a.n_folds, a.seed)
    print(pd.crosstab(df["label"], df["fold"], margins=True))
    assert df["image_id"].is_unique
    print(f"\nSaved {FOLDS_CSV}")


if __name__ == "__main__":
    main()
