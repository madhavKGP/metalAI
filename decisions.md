# MetalAI — Decision Log

Every non-trivial decision goes here: **what**, **why**, **alternatives**, **status**.
Newest at the bottom. Never delete an entry — mark it `Superseded by D-xxx` instead.
Format: `D-NNN · date · owner · status (Accepted / Proposed / Superseded)`

---

## D-001 · 2026-10-28 · Claude+Madhav · Accepted
**Scrap the previous project's implementation; keep only the problem idea.**
- Why: audit of `Akshaay1/metal-defect-detector` found no data, no training/eval code, an "ensemble" that reads the class from the uploaded filename (label leakage), segmentation without masks, faked training UI, and thesis claims not backed by code.
- Consequence: nothing is reused except ideas (Flask/Gradio UI layout, split-then-augment, plain Grad-CAM).

## D-002 · 2026-10-28 · Madhav · Accepted
**Project = MetalAI with 3 modules: (1) Surface Defect Inspector, (2) Steel Property Predictor & Designer, (3) Alloy Design Assistant (HEA + LLM agent).** Reimplementation of published work is acceptable for the MTP; value = correctness, reproducibility, metallurgical interpretation, working app.
- Details: `MetalAI_Project_Guide.md`.
- We start with Module 1 (Part 1).

## D-003 · 2026-10-28 · Claude · Accepted
**Module 1 data = NEU surface defect database (Kaggle mirror `kaustubhdikshit/neu-surface-defect-database`, NEU-DET layout with VOC XML boxes).** One dataset serves both classification (image label) and detection (boxes).
- Labels come from the dataset's own file-name convention (`crazing_12.jpg`, `Cr_12.bmp`) with folder name + XML `<name>` as cross-checks. This is used **only to build ground truth**, never at inference.
- `prepare_data.py` is layout-agnostic and de-duplicates by (class, number) and MD5 so NEU-CLS + NEU-DET copies don't double count.
- Expected: 1,800 images, 6 classes × 300, 200×200 grayscale. Any deviation → OpenCode reports it.
- Alternatives considered: GC10-DET (harder, used later for robustness), Severstal (needed later for anomaly detection).

## D-004 · 2026-10-28 · Claude · Accepted
**Stack: PyTorch + torchvision (CNN), Ultralytics (YOLO), scikit-image + scikit-learn (classical), pytorch-grad-cam, Gradio (app).**
- Why: standard, well-documented, small dependency surface. torchvision models avoid an extra `timm` dependency.
- Install torch with CUDA wheels first; `pip install -e .` makes `metalai` importable.

## D-005 · 2026-10-28 · Claude · Accepted
**Evaluation = stratified 5-fold CV (seed 42), fold file committed at `data/splits/neu_folds.csv`; inner 10 % stratified validation carved from the 4 training folds (seed 42 + k).**
- Every model (SVM, CNNs, YOLO) uses `splits.get_fold(k)` → identical train/val/test images.
- Model selection / early stopping uses **val only**; test fold is touched once.
- SVM uses inner 3-fold CV on train+val for hyper-parameters (no early stopping needed).
- Report mean ± std across folds + per-class metrics.

## D-006 · 2026-10-28 · Madhav/Claude · Accepted
**Hardware: training runs on the GPU only (ASUS F15 laptop). Keep runs light.**
- Scripts call `require_cuda()` and refuse to train on CPU (except `--allow-cpu` for tiny smoke tests).
- Light settings: AMP mixed precision, channels-last, images cached in RAM, `workers=0` for CNN on Windows, batch 32 (CNN) / 16 (YOLO) with documented fallback to 16 / 8 on out-of-memory.
- Run fold-by-fold (`--folds 0` first, then the rest with `--skip-existing`) so the laptop never runs for hours unattended.
- Classical SVM is CPU-only (light).
- If the laptop has no CUDA GPU or VRAM < 4 GB: fall back to Kaggle Notebooks (free GPU) — Claude will provide a notebook. **To confirm in T1.**
- Update 2026-10-28: Madhav's laptop has an **RTX 4050** (stated 8 GB; laptop 4050s usually have 6 GB dedicated — T1 `env.json` will show the real number). Either is enough for all Module-1 training at default settings (ResNet-50 bs 32 AMP ≈ 3–4 GB, YOLOv8n 640 bs 16 ≈ 3 GB). Kaggle fallback not needed unless T4 shows problems.

## D-007 · 2026-10-28 · Claude · Accepted
**One preprocessing path for the CNN** (`m1_surface/preprocess.py`): grayscale → 3 channels → resize 224 → ImageNet mean/std. Used by training, evaluation, Grad-CAM and the app.
- Train-only augmentation: H/V flips, ±10° rotation, brightness/contrast ±0.2 (defect geometry preserved; flips/rotations are physically valid for strip surfaces).
- Why: the old project normalised differently at inference than ImageNet backbones expect.

## D-008 · 2026-10-28 · Claude · Accepted
**Classical baseline = multi-scale uniform LBP (P,R) ∈ {(8,1),(16,2),(24,3)} (54-d) + GLCM (6 props × 3 distances × {angle-mean, angle-range}, 64 grey levels; 36-d) → StandardScaler → RBF-SVM.**
- Ablation: LBP only / GLCM only / both.
- Why: texture descriptors are the metallurgically interpretable baseline (defects = local texture change); cheap on CPU.

## D-009 · 2026-10-28 · Claude · Accepted
**CNNs: ResNet-50 (main) and EfficientNet-B0 (light), ImageNet-pretrained, fully fine-tuned.** AdamW lr 3e-4, wd 1e-4, 1 warm-up epoch + cosine, label smoothing 0.1, ≤ 25 epochs, early stop patience 6 on val loss.
- Grad-CAM: `pytorch-grad-cam`, target = last conv block, **raw heatmaps only** (no equalisation/saturation tricks).
- Final app model: `--full` retrain on all 1,800 images (10 % val for early stopping).

## D-010 · 2026-10-28 · Claude · Accepted
**App rule: models see only pixels.** No filename, folder or form-field information reaches prediction code. Low confidence (< 0.5) → "uncertain, manual review".

## D-011 · 2026-10-28 · Claude · Accepted
**Detector = Ultralytics YOLOv8n at imgsz 640 (literature-comparable baseline), ≤150 epochs, patience 30, batch 16, AMP, `cache=ram`.** Model selection on the inner val split; final metrics via `val(split="test")` on the held-out fold. Report mAP@0.5 and mAP@0.5:0.95 + per-class AP50.
- Optional second run: YOLOv8s or YOLO11n if time/VRAM allow.

## D-012 · 2026-10-28 · Claude · Accepted
**Reference targets (from literature, not promises):** NEU-CLS CNN accuracy ≈ 98–100 %; NEU-DET YOLOv8n ≈ 73 % mAP@0.5 (improved YOLOs 77–86 %). Classical texture + SVM: no fixed target; reported honestly.

## D-013 · 2026-10-28 · Claude · Accepted
**Workflow: Claude writes code + task specs; OpenCode runs commands on the laptop and reports in `OPENCODE.md`; Claude reads reports/figures (`reports/m1/...`) and decides next steps here.**
- All numbers come from `results/m1/*.csv`; `report.py` builds `reports/m1/REPORT.md`. Nothing typed by hand.
- `results/`, `reports/`, `data/splits/` are committed; `data/raw`, `data/processed`, `models/`, `runs/` are not.

## D-015 · 2026-10-28 · Madhav/Claude · Accepted
**Fully automated workflow — Madhav does nothing manually.** OpenCode performs setup, Kaggle auth, downloads, training, checks (headless app test `app/smoke_test.py` instead of clicking in a browser). Claude writes code/tasks and reviews.

## D-016 · 2026-10-28 · Claude · Accepted
**Secrets:** Kaggle token stored at `.secrets/kaggle_access_token` (git-ignored, along with `access_token`, `kaggle.json`); OpenCode copies it to `%USERPROFILE%\.kaggle\access_token`. Never printed, logged or committed. Recommendation to Madhav: regenerate the token after setup since it was shared in chat.

## D-017 · 2026-10-28 · Claude · Accepted
**Experiment tracking.** Every run has an id (`E001`…; `SMOKE` for throw-away) set via `METALAI_EXP`.
- `start_experiment()` saves command, args, config, git commit → `experiments/runs/<id>/<script>.json`.
- Every results row has `exp_id`; weights → `models/m1/<kind>/<id>/`, figures → `reports/m1/<kind>/<id>_<model>/`, YOLO runs → `runs/m1_yolo/<id>_...` — so experiments never overwrite each other.
- Registry with status + conclusions: `experiments/EXPERIMENTS.md`. Invalid runs are marked `discarded`, not deleted; only non-`E` ids can be removed with `python -m metalai.common.cleanup`.
- The app serves the weights named in `configs/app.yaml` (Claude sets it after review), else the newest weights.

## D-018 · 2026-10-28 · Claude · Accepted
**Code organisation:** all code in `src/metalai/<module>/`, one file per pipeline step, run via `python -m`; hyper-parameters only in `configs/*.yaml` or CLI flags; no notebooks for reported numbers; app code in `app/`. Layout documented in `OPENCODE.md` §0.2.

## D-019 · 2026-10-28 · Madhav/Claude · Accepted
**Parallelism: across resources, never on the GPU.** One GPU job at a time (6 GB laptop GPU); CPU work (classical SVM with `--n-jobs 4`, EDA, report drafting), downloads and GPU monitoring run in parallel via OpenCode subagents. Orchestrator agent owns git commits and the status board; subagents return text only. Details: `OPENCODE.md` §0.4.
- Rejected: two concurrent GPU trainings (OOM risk, no net speed-up on one GPU); multi-GPU/DDP (only one GPU).

## D-020 · 2026-10-29 · Claude · Accepted (fixes a bug found in T3/T6 review)
**Detection box labels come from the XML `<name>`, not from the image's class.**
- Finding: the 188 "XML label mismatches" are **genuine secondary defects** (e.g. `patches_168.xml` = 3 patches + 4 inclusion boxes; `crazing_104.xml` = 3 crazing + 1 patches). The original `prepare_data` relabelled them to the image class → ~4.5 % of YOLO boxes had the wrong class in E004.
- Fix: box class = XML name; image (classification) label = file-name class (dataset convention, unchanged). `data_summary.json` now reports `secondary_defect_boxes` and `images_with_secondary_defects`.
- Consequence: **E004 discarded** (trained on wrong labels); YOLO is re-run as E005. CNN/SVM results (E001–E003) are unaffected (image labels unchanged, splits unchanged).
- Dataset size: **1,799 images** is canonical — `patches_105` is a byte-identical duplicate of another image and was dropped by MD5 de-duplication (fold 4 has 359 test images).

## D-021 · 2026-10-29 · Claude · Accepted
**Hardware correction: the laptop GPU is an RTX 4060 Laptop, 8 GB** (from T1 `nvidia-smi`), not a 4050. Observed: peak VRAM 1.5–4.1 GB, max 73 °C, no OOM at default batch sizes. Defaults stay; no need for fallbacks.

## D-022 · 2026-10-29 · Claude · Accepted
**Model/config selection uses inner-validation metrics only; test folds are for reporting.**
- `yolo_train.py` now also evaluates on the inner val split (`val_mAP50`, `val_mAP50_95`) and records `best_epoch`.
- YOLO variants (E005–E007) are compared on **fold-0 val mAP50-95**; the winner is then run on folds 1–4. Picking by fold-0 *test* mAP would leak the test fold into model selection.

## D-023 · 2026-10-29 · Claude · Accepted — review of T6 (fold 0)
- **Classification is saturated on NEU**: SVM LBP+GLCM 99.44 ± 0.20 % (5 folds); ResNet-50 and EfficientNet-B0 100 % on fold 0. Consistent with literature (98–100 %). Run all 5 folds anyway (cheap) for honest mean ± std; the interesting questions (robustness, localisation) move to Part 2.
- **CNN curves**: converge in 2–4 epochs; val loss < train loss is expected (train uses augmentation + dropout + label smoothing 0.1, which also floors train loss ≈ 0.43). No overfitting.
- **Grad-CAM (ResNet-50)**: tight on localised defects (inclusion, scratches, patches); diffuse/texture-wide on crazing and rolled-in scale — physically sensible since those are distributed textures. Some pitted-surface heat lies outside GT boxes → to quantify in Part 3 (pointing game).
- **YOLO E004 (discarded, but informative)**: class confusion is near-zero (diagonal 0.97–0.99); errors are mostly **background false positives** (crazing 0.35, rolled-in scale 0.20) — i.e. *box granularity* ambiguity of diffuse defects, not misclassification. Train losses still falling at epoch 150 and val mAP still creeping up → not converged; val DFL loss rising slightly late → mild localisation overfit. Hence variants: longer training (E006) and a bigger model (E007). Crazing will likely stay the weakest class (known in literature); explain rather than over-tune.
- Tech-debt notes from OpenCode: sklearn `SVC(probability=...)` deprecation warning and "lr_scheduler.step() before optimizer.step()" warning (first AMP step skipped by GradScaler) are harmless; leave.

## D-024 · 2026-10-30 · Claude · Accepted (fix for E005 crash)
**YOLO runs with `cache=False` (configurable `--cache`, default from `configs/m1.yaml`) and keep `workers=2`.**
- Cause of the E005 crash at epoch 46: system-RAM OOM (15.6 GB laptop) inside mosaic augmentation. On Windows, dataloader workers are *spawned* and each gets its own copy of the dataset object, including the `cache='ram'` image cache (images are cached at 640 px, ~1.2 MB each) → RAM multiplies by (workers + 1) and fragments over time.
- NEU images are 200×200 PNGs (~40 KB) — decoding from disk is negligible next to GPU time, so caching buys nothing. `cache='disk'` would write 640-px `.npy` files (~1.5 GB per fold) for no gain.
- Kept `workers=2` (mosaic on the main process alone would slow training). If RAM still runs out → `--workers 0`.
- Added `--resume` (continues from `runs/.../weights/last.pt`) so a future crash doesn't lose the run.
- Rejected: bigger pagefile (treats the symptom, slow), lower imgsz (changes the experiment, breaks comparability with literature at 640).
- E005 is re-run from scratch (the partial run used a different cache setting; cheap to redo, ~25 min).

## D-025 · 2026-10-30 · Claude · Accepted — review of T8
**CNN 5-fold results final:** ResNet-50 (E002) 99.94 % mean (one error, fold 3), EfficientNet-B0 (E003) 100 % on all 5 folds.
- The single ResNet error, `scratches_149` → predicted *inclusion* at 0.50 confidence: Grad-CAM sits on a dark blob next to the scratch, i.e. a locally inclusion-like feature; the classifier was uncertain (0.50 = exactly the app's "uncertain" threshold). Plausible borderline case, not a systematic failure. Useful thesis example.
- **App classifier = EfficientNet-B0 (E003)**: same/better accuracy, ~half the VRAM, faster → supersedes the earlier plan to serve ResNet-50. ResNet-50 stays as a comparison model.

## D-014 · 2026-10-28 · Claude · Accepted
**Sandbox verification done before hand-off:** `prepare_data`, `splits`, `eda`, `classical`, `yolo_data`, `report` were executed end-to-end on a synthetic NEU-like dataset (180 fake images) — all ran and produced figures. `cnn`, `gradcam`, `yolo_train`, `app` were syntax-checked only (no torch/GPU in sandbox) → OpenCode runs a 1-epoch smoke test first (T4).
