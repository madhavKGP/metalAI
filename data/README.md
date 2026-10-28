# Data sources (single source of truth)

## NEU surface defect database (Module 1)
- Source: Kaggle mirror https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- Original: Northeastern University (China), Song & Yan — 6 classes × 300 grayscale 200×200 images of hot-rolled steel strip, with bounding boxes (NEU-DET).
- Downloaded on: 2026-10-28
- Downloaded by: OpenCode (`kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database --unzip`)
- Local path: `data/raw/neu/` (not committed)
- Layout as downloaded: `data/raw/neu/NEU-DET/{train,validation}/{images,annotations}/` — 1,800 `.jpg` (300/class, 6 classes) + 1,800 Pascal VOC `.xml`, images 200×200 grayscale.
- Licence / terms: Kaggle metadata reports license `unknown`; original source is Northeastern University (China), Song & Yan — cite the original NEU-DET paper in any publication. Mirror: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- Processed by: `python -m metalai.m1_surface.prepare_data` → `data/processed/neu/`
- Splits: `data/splits/neu_folds.csv` (committed; stratified 5-fold, seed 42)

## Later
- GC10-DET (Module 1 robustness) — TBD
- Severstal (Module 1 anomaly detection) — TBD
- NIMS steel fatigue (Module 2) — TBD
- HEA phase dataset (Module 3) — TBD
