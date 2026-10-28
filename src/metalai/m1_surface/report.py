"""Aggregate all Module-1 results into one markdown report (numbers come ONLY from results/*.csv).

Run:  python -m metalai.m1_surface.report
Out:  reports/m1/REPORT.md
"""
from __future__ import annotations

import json

import pandas as pd

from metalai.common.utils import REPORTS, RESULTS, ROOT, git_commit, now
from metalai.m1_surface import CLASSES

R = RESULTS / "m1"
OUT = REPORTS / "m1" / "REPORT.md"


def ms(s: pd.Series, pct=True) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    f = 100 if pct else 1
    return f"{s.mean() * f:.2f} ± {s.std(ddof=1) * f if len(s) > 1 else 0:.2f}"


def latest(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Keep only the most recent row per (keys) - reruns overwrite older results."""
    return df.sort_values("timestamp").groupby(keys, as_index=False).tail(1)


def section_table(df, group, metrics, title, pct=True):
    lines = [f"### {title}", "", "| " + " | ".join([group, "exp", "folds"] + metrics) + " |",
             "|" + "---|" * (len(metrics) + 3)]
    for g, sub in df.groupby(group):
        exps = ",".join(sorted(sub["exp_id"].astype(str).unique())) if "exp_id" in sub else "-"
        lines.append("| " + " | ".join([str(g), exps, str(sorted(sub.fold.tolist()))]
                                       + [ms(sub[m], pct) for m in metrics]) + " |")
    return lines + [""]


def imgs(rel_dir: str) -> list[str]:
    d = ROOT / rel_dir
    if not d.exists():
        return []
    return [f"![{p.name}]({p.relative_to(REPORTS / 'm1').as_posix()})" for p in sorted(d.glob("*.png"))]


def main() -> None:
    L = [f"# Module 1 - Surface Defect Inspector: results", "",
         f"Generated {now()} | git {git_commit()} | numbers = mean ± std over test folds (%, unless stated)", ""]

    ds = REPORTS / "m1" / "data_summary.json"
    if ds.exists():
        d = json.loads(ds.read_text())
        L += ["## Data", "", f"- images kept: **{d.get('n_images_kept')}**, with boxes: {d.get('images_with_boxes')}",
              f"- per class: {d.get('images_per_class')}", f"- boxes per class: {d.get('boxes_per_class')}",
              f"- duplicates dropped: {d.get('duplicates_dropped')}, unparsed: {d.get('n_unparsed_images')}, "
              f"xml label mismatches: {d.get('n_xml_label_mismatch')}", ""]
        L += imgs("reports/m1/eda") + [""]

    L += ["## Main comparison", ""]
    summary_rows = []
    f1cols = [f"f1_{c}" for c in CLASSES]
    if (R / "classical_results.csv").exists():
        c = latest(pd.read_csv(R / "classical_results.csv"), ["featset", "fold"])
        L += section_table(c, "featset", ["accuracy", "macro_f1"], "Classical: LBP/GLCM + SVM")
        L += section_table(c, "featset", f1cols, "Classical: per-class F1")
        for fs, sub in c.groupby("featset"):
            summary_rows.append(("SVM " + fs, ms(sub.accuracy), ms(sub.macro_f1), "-", "-"))
    if (R / "cnn_results.csv").exists():
        c = latest(pd.read_csv(R / "cnn_results.csv"), ["model", "fold"])
        L += section_table(c, "model", ["accuracy", "macro_f1"], "CNN classifiers")
        L += section_table(c, "model", f1cols, "CNN: per-class F1")
        L += section_table(c, "model", ["train_minutes", "best_epoch", "ms_per_img_bs1", "peak_vram_gb"],
                           "CNN: cost", pct=False)
        for mdl, sub in c.groupby("model"):
            summary_rows.append((mdl, ms(sub.accuracy), ms(sub.macro_f1), "-", "-"))
    if (R / "yolo_results.csv").exists():
        y = latest(pd.read_csv(R / "yolo_results.csv"), ["model", "fold"])
        L += section_table(y, "model", ["mAP50", "mAP50_95", "precision", "recall"], "YOLO detection (test folds)")
        L += section_table(y, "model", [f"AP50_{c}" for c in CLASSES], "YOLO: per-class AP50")
        L += section_table(y, "model", ["train_minutes", "ms_inference"], "YOLO: cost", pct=False)
        for mdl, sub in y.groupby("model"):
            summary_rows.append((mdl, "-", "-", ms(sub.mAP50), ms(sub.mAP50_95)))

    if summary_rows:
        L += ["### Summary", "", "| Method | Accuracy | Macro-F1 | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
        L += [f"| {' | '.join(r)} |" for r in summary_rows] + [""]
        L += ["Literature reference: NEU-CLS CNNs ~98-100% acc; NEU-DET YOLOv8n ~73% mAP@0.5 (see decisions.md D-012).", ""]

    L += ["## Figures", "", "### Classical"] + imgs("reports/m1/classical")
    cnn_dir = REPORTS / "m1" / "cnn"
    if cnn_dir.exists():
        for d in sorted(p for p in cnn_dir.iterdir() if p.is_dir()):
            L += ["", f"### CNN {d.name}"] + imgs(f"reports/m1/cnn/{d.name}")
    L += ["", "### Grad-CAM"] + imgs("reports/m1/gradcam")
    ydir = REPORTS / "m1" / "yolo"
    if ydir.exists():
        for d in sorted(p for p in ydir.iterdir() if p.is_dir()):
            L += ["", f"### YOLO {d.name}"] + imgs(f"reports/m1/yolo/{d.name}")

    L += ["", "## Observations (OpenCode fills in, Claude reviews)", "", "- ...", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
