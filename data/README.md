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

## Steel corrosion condition photos (Module 1 extension, E009)
- Source: Jackson (2025), https://github.com/bennett-jackson/steel-corrosion-dataset (commit d9e5336), **MIT licence**. Public data, not collected by us; cite it.
- Content: 3,417 photos (512x512 JPG) of steel bridge members from inspections, in 5 folders: good 287, fair 1,262, poor 735, severe 946, weathering 187.
- Local path: `data/raw/corrosion_jackson/` (not committed). Download: `git clone --depth 1 https://github.com/bennett-jackson/steel-corrosion-dataset data/raw/corrosion_jackson`
- Used: good/fair/poor/severe (3,230 files). Weathering left out (a steel type, not a condition grade). 43 exact duplicates dropped, 3 photos that appear under two different labels dropped entirely (7 files) -> 3,180 images.
- Scale: whole structural members, metres. NOT microstructure. See docs/CORROSION_DATA.md.

## NEU mirror check (2026-11-10)
- NEU-DET was also re-prepared from the GitHub mirror https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection (IMAGES/ + Validation_Images/, 1,800 jpg + xml). `prepare_data` gives the identical manifest (1,799 images, same box counts, same fold ids) as the Kaggle copy.

## Later
- GC10-DET (Module 1 robustness) — TBD
- Severstal (Module 1 anomaly detection) — TBD
- NIMS steel fatigue (Module 2) — TBD
- HEA phase dataset (Module 3) — TBD
