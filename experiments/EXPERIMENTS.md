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
| E001 | | M1 | LBP / GLCM / both + RBF-SVM | D-008 | 0-4 | planned | | |
| E002 | | M1 | ResNet-50 (ImageNet, full FT) | D-009, bs 32, 224 px | 0 | planned | | |
| E003 | | M1 | EfficientNet-B0 (ImageNet, full FT) | D-009, bs 32, 224 px | 0 | planned | | |
| E004 | | M1 | YOLOv8n | D-011, 640 px, bs 16 | 0 | planned | | |

Folder layout produced by runs:
```
experiments/runs/<exp_id>/<script>.json     command + args + config + commit (committed)
results/m1/*.csv                            metrics, one row per (exp, model, fold) (committed)
reports/m1/...                              figures (committed)
models/m1/...                               weights (NOT committed)
runs/m1_yolo/...                            raw Ultralytics runs (NOT committed)
```
