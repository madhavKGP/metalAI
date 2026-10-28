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
| E001 | 2026-10-28 | M1 | LBP / GLCM / both + RBF-SVM | D-008 | 0-4 | done | both: acc 0.9944 ± 0.0020 (lbp 0.9811, glcm 0.9817) | |
| E002 | 2026-10-28 | M1 | ResNet-50 (ImageNet, full FT) | D-009, bs 32, 224 px | 0 | done | acc 100.00, macroF1 100.00 (early stop ep 11, 4.4 min, peak VRAM 2.69 GB) | |
| E003 | 2026-10-28 | M1 | EfficientNet-B0 (ImageNet, full FT) | D-009, bs 32, 224 px | 0 | done | acc 100.00, macroF1 100.00 (early stop ep 24, 7.1 min, peak VRAM 1.49 GB) | |
| E004 | 2026-10-28 | M1 | YOLOv8n | D-011, 640 px, bs 16 | 0 | done | test mAP50 74.56, mAP50-95 42.54 (22.8 min) | |

Folder layout produced by runs:
```
experiments/runs/<exp_id>/<script>.json     command + args + config + commit (committed)
results/m1/*.csv                            metrics, one row per (exp, model, fold) (committed)
reports/m1/...                              figures (committed)
models/m1/...                               weights (NOT committed)
runs/m1_yolo/...                            raw Ultralytics runs (NOT committed)
```
