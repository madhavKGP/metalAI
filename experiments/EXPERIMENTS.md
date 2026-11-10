# Experiment registry

One row per experiment. An **experiment** = one model/config trained on one or more folds.
IDs are assigned by **Claude** in `OPENCODE.md` tasks (`E001`, `E002`, ...). `SMOKE` = throw-away tests.

How it works:
- Before a run, OpenCode sets the id: `$env:METALAI_EXP = "E002"`.
- Every script calls `start_experiment()` → saves exact command, args, config, git commit to
  `experiments/runs/<exp_id>/<script>.json`.
- Every result row in `results/m1/*.csv` carries the `exp_id` column; `reports/m1/REPORT.md` shows it.
- After the run, OpenCode fills in the row below (status, headline metric, notes). Claude writes the
  "Conclusion" column after reviewing figures. Never delete rows — mark them `discarded` with a reason.

| ID | Date | Module | What (model / featureset) | Key config | Folds | Status | Headline result | Conclusion (Claude) |
|---|---|---|---|---|---|---|---|---|
| E001 | 2026-10-28 | M1 | LBP / GLCM / both + RBF-SVM | D-008 | 0-4 | done | both: acc 0.9944 ± 0.0020 (lbp 0.9811, glcm 0.9817) | Strong texture baseline; LBP+GLCM complementary (+1.3 pt over either alone). Classification on NEU is near-saturated even classically. |
| E002 | 2026-10-28 | M1 | ResNet-50 (ImageNet, full FT) | D-009, bs 32, 224 px | 0-4 | done | acc 99.94 (100/100/100/99.72/100), ~5 min/fold, peak VRAM 3.5 GB | Converges in ~3 epochs, no overfit. Grad-CAM localises inclusion/scratches/patches well; diffuse on crazing/RS (texture classes). Only error: scratches_149 → inclusion at 0.50 conf (dark blob beside scratch) — borderline case (D-025). Comparison model. |
| E003 | 2026-10-28 | M1 | EfficientNet-B0 (ImageNet, full FT) | D-009, bs 32, 224 px | 0-4 | done | acc 100.00 on all 5 folds, ~3-7 min/fold, peak VRAM 1.5-3.2 GB | Best classifier: 100 % at lower cost → **serves the app** (D-025). |
| E004 | 2026-10-28 | M1 | YOLOv8n | D-011, 640 px, bs 16 | 0 | **discarded** | test mAP50 74.56, mAP50-95 42.54 (22.8 min) | Discarded: 188 secondary-defect boxes had wrong class (D-020). Useful signal: errors = background FPs on crazing/RS, not class confusion; not converged at 150 ep. |
| E005 | 2026-10-29 | M1 | YOLOv8n — corrected labels | D-011, 640 px, bs 16, 150 ep, patience 30, cache=False, workers 0 (D-024) | 0 | **done — SELECTED DETECTOR** | **val mAP50-95 0.4477** (val mAP50 0.7788); test mAP50 0.7537 / mAP50-95 0.4257; best ep 118, early stop 148/150; ~51 min wall (workers=0) | Selected per D-022: highest fold-0 val mAP50-95 among E005/E006/E007 (E007 cancelled at ~ep 91 before completion). Real wall time ~51 min (CSV train_minutes=4.3 shows only resumed portion; Claude's fix reads cumulative time going forward). Crash history: (1) cache='ram' OOM'd mosaic worker @ ep 46; (2) cache=false, workers=2 killed @ ep 25 (free RAM 1.4 GB); (3) workers=0 trained fine but eval crashed WinError 1455 → fixed by passing workers to best.val() (3cce2bd). |
| E006 | 2026-10-29 | M1 | YOLOv8n — longer | 640 px, bs 16, **300 ep, patience 50**, cache=False, workers 0 | 0 | done | **val mAP50-95 0.4454** (val mAP50 0.7605); test mAP50 0.7229 / mAP50-95 0.4126; best ep 191, early stop 241/300; 436.2 min (7.26 h wall) | Longer training didn't help: best ep 191 vs E005's 118, val within 0.005 of E005 but ~8.5× slower. GPU util avg only 12.7% (disk/CPU contention). Per D-022 tie-break, E005 dominates on time. workers=0 used (D-024 fallback). |
| E007 | 2026-10-29 | M1 | YOLOv8s — bigger model | 640 px, bs 16, 150 ep, patience 30, cache=False, workers 0 | 0 | **cancelled** | — | Cancelled per D-027 (close Module 1): stopped at ~ep 91/150 (~128 min) before completion. yolov8s weights (106.6 MB) deleted from runs/. |
| E008 | 2026-11-10 | M1 | Scratch vs crack: line-shape features + logreg / RF | D-030, NEU scratches vs crazing (600 imgs) | 0-4 | done | acc 99.67 ± 0.75 (both models), AUC 1.00; brightness-inverted test 99.67 (RF) / 99.83 (logreg) | Top features: endpoint density, fragments, longest line, orientation coherence. Explainable; NEU crazing = crack network, not single micro-cracks. |
| E009 | 2026-11-10 | M1 ext | Corrosion condition, colour + LBP + GLCM + RBF-SVM, PUBLIC Jackson bridge photos | D-031, 3,180 imgs, grouped 5-fold | 0-4 | done | grouped: binary 72.48 ± 3.71 (majority 52.0), 4-class 45.94 ± 2.74 (majority 39.2); random split 2-6 pts higher | Weak honest baseline; errors mostly between neighbouring grades; bridge scale, not microstructure. CNN on GPU next. |

Selection rule (D-022): among E005–E007 pick the highest **fold-0 val_mAP50_95**; if within 0.005, pick the faster one.

Folder layout produced by runs:
```
experiments/runs/<exp_id>/<script>.json     command + args + config + commit (committed)
results/m1/*.csv                            metrics, one row per (exp, model, fold) (committed)
reports/m1/...                              figures (committed)
models/m1/...                               weights (NOT committed)
runs/m1_yolo/...                            raw Ultralytics runs (NOT committed)
```
