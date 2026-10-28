# MetalAI

AI toolkit for metallurgy — M.Tech project.

| Module | Status |
|---|---|
| 1 · Surface Defect Inspector (NEU: LBP/GLCM+SVM, CNN + Grad-CAM, YOLO) | **in progress** |
| 2 · Steel Property Predictor & Designer (NIMS fatigue, SHAP, Bayesian optimisation) | planned |
| 3 · Alloy Design Assistant (HEA phase prediction + LLM agent) | planned |

Key docs: [`decisions.md`](decisions.md) (why) · [`OPENCODE.md`](OPENCODE.md) (tasks + run log) · `reports/m1/REPORT.md` (auto-generated results)

## Setup (Windows, NVIDIA GPU)

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -m metalai.common.env_check
```

## Module 1 pipeline

```powershell
# data (NEU from Kaggle into data/raw/neu)
python -m metalai.m1_surface.prepare_data
python -m metalai.m1_surface.splits
python -m metalai.m1_surface.eda
python -m metalai.m1_surface.yolo_data
# models
$env:METALAI_EXP = "E001"                                     # every run gets an experiment id
python -m metalai.m1_surface.classical                       # CPU
python -m metalai.m1_surface.cnn --model resnet50 --folds 0 1 2 3 4
python -m metalai.m1_surface.gradcam --model resnet50 --fold 0
python -m metalai.m1_surface.yolo_train --folds 0 1 2 3 4
# results + app
python -m metalai.m1_surface.report                           # -> reports/m1/REPORT.md
python app\app.py                                             # -> http://127.0.0.1:7860
```

## Layout

```
configs/        hyper-parameters (m1.yaml) + which weights the app serves (app.yaml)
experiments/    EXPERIMENTS.md registry + runs/<exp_id>/ (command, config, commit per run)
data/raw/       downloads (not committed)      data/processed/  generated (not committed)
data/splits/    fold files (committed)
src/metalai/    common/ (utils, metrics, env_check) · m1_surface/ (all Module-1 code)
models/ runs/   weights + Ultralytics runs (not committed)
results/        metric CSVs (committed)        reports/  figures + REPORT.md (committed)
app/            Gradio app
```

## Rules

Split first, augment second · one preprocessing path · GPU-only training · every number from a CSV · models see only pixels.
