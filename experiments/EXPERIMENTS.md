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
| E005 | | M1 | YOLOv8n — corrected labels | D-011, 640 px, bs 16, 150 ep, patience 30 | 0 (→0-4 if selected) | crashed (system-RAM OOM in mosaic worker @ ep 46/150, 2026-10-30; no test metrics) → **re-run from scratch with cache=False (D-024)** | | Crash cause: cache='ram' copied into each Windows worker. Not a model issue. |
| E006 | | M1 | YOLOv8n — longer | 640 px, bs 16, **300 ep, patience 50** | 0 (→0-4 if selected) | planned | | |
| E007 | | M1 | YOLOv8s — bigger model | 640 px, bs 16, 150 ep, patience 30 | 0 (→0-4 if selected) | planned | | |

Selection rule (D-022): among E005–E007 pick the highest **fold-0 val_mAP50_95**; if within 0.005, pick the faster one.

Folder layout produced by runs:
```
experiments/runs/<exp_id>/<script>.json     command + args + config + commit (committed)
results/m1/*.csv                            metrics, one row per (exp, model, fold) (committed)
reports/m1/...                              figures (committed)
models/m1/...                               weights (NOT committed)
runs/m1_yolo/...                            raw Ultralytics runs (NOT committed)
```
