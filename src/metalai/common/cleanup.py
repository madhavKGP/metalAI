"""Remove every artefact of ONE experiment id (only for throw-away runs like SMOKE).

Run:  python -m metalai.common.cleanup --exp SMOKE            (dry run: lists what would go)
      python -m metalai.common.cleanup --exp SMOKE --yes      (actually deletes)
Real experiments (E0xx) are never deleted - mark them 'discarded' in experiments/EXPERIMENTS.md instead.
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from metalai.common.utils import EXPERIMENTS, MODELS, REPORTS, RESULTS, RUNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.exp.upper().startswith("E") and a.exp[1:].isdigit():
        raise SystemExit("Refusing to delete a registered experiment - mark it 'discarded' instead.")

    paths = [EXPERIMENTS / "runs" / a.exp]
    paths += list(MODELS.glob(f"*/*/{a.exp}"))
    paths += list(REPORTS.glob(f"*/*/{a.exp}_*")) + list(REPORTS.glob(f"*/gradcam/{a.exp}_*"))
    paths += list(RUNS.glob(f"*/{a.exp}_*")) + list(RESULTS.glob(f"*/*/{a.exp}_*"))
    paths += [REPORTS / "m1" / "app_smoke", REPORTS / "gpu_logs" / f"{a.exp}.csv"] if a.exp == "SMOKE" else []
    paths = [p for p in dict.fromkeys(paths) if p.exists()]
    csvs = [f for f in RESULTS.glob("*/*.csv") if "exp_id" in pd.read_csv(f, nrows=0).columns]

    for p in paths:
        print(("DELETE " if a.yes else "would delete ") + str(p))
        if a.yes:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for f in csvs:
        df = pd.read_csv(f)
        n = int((df.exp_id.astype(str) == a.exp).sum())
        if n:
            print(("DROP " if a.yes else "would drop ") + f"{n} rows from {f}")
            if a.yes:
                df[df.exp_id.astype(str) != a.exp].to_csv(f, index=False)
    if not a.yes:
        print("\nDry run only. Add --yes to delete.")


if __name__ == "__main__":
    main()
