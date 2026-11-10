# MTP progress update

**Topic:** Use of ML to identify surface defects in materials and prediction of microstructure of metals
**Date:** 10 Nov 2026. All numbers come from `results/m1/*.csv` (see `reports/m1/REPORT.md`).

## 1. Done: surface defect inspector (NEU steel strip, public data)

1,799 images, 6 defect types, 5-fold cross-validation (test images never seen in training).

| Method | Result |
|---|---|
| Texture features (LBP + GLCM) + SVM | 99.44 ± 0.20 % accuracy |
| EfficientNet-B0 (deep learning) | 100.00 ± 0.00 % |
| ResNet-50 | 99.94 ± 0.12 % |
| YOLOv8n detector (draws boxes) | 75.4 % mAP@0.5 (fold 0 only) |

There's a live web app: upload an image to get the defect type, boxes and a Grad-CAM heatmap of where the model looked.
Classification on NEU is near-solved in the literature, so it's our baseline, not our contribution.

## 2. New: scratch vs crack, from the shape of the line

- Find the thin lines, thin them to a 1-pixel skeleton, then measure: number of line ends, number of pieces, longest line, whether the lines are parallel, branching.
- **99.67 ± 0.75 %** accuracy (600 images, same 5 folds).
- A scratch is one or a few long parallel lines. A crack is many short, branching lines in random directions.
- Check: in NEU, scratches look bright and cracks dark. We flipped every image (bright to dark) and the result held (99.67 %), so the model uses shape, not brightness.
- Limit: NEU crazing is a crack network, not single micro-cracks. True micro-cracks need microscope images.

Figures: `reports/m1/scratch_crack/examples.png`, `feature_boxplots.png`.

## 3. New: first corrosion model, on public data

- No lab access yet, so we used a public dataset: Jackson (2025), steel bridge inspection photos, MIT licence, condition graded good / fair / poor / severe.
- Cleaned it: removed 43 duplicate photos and 3 photos that appear under two different grades.
- Kept photos of the same structure together in one fold, so the test is fair.
- Results (colour + texture features + SVM):
  - corroded vs not corroded: **72.5 %** (chance about 52 %)
  - 4 grades: **45.9 %** (chance about 39 %); most errors are between neighbouring grades.
- Honest reading: a weak first baseline. These are whole bridge members photographed from metres away, not microstructure.

## 4. Gap we found

We found no public, labelled set of **corrosion micrographs** (pitting vs intergranular vs uniform, or corroded vs mechanical pits). That's the part that needs our own samples, and it would be our real contribution.

## 5. Proposed next step (needs lab access)

1. Mild steel and 304 stainless coupons: polish, etch, photograph.
2. Pitting: 3.5 % NaCl for a few days.
3. Intergranular attack: sensitize 304 at about 650 °C, then oxalic acid etch (ASTM A262 Practice A).
4. Mechanical pits for comparison: punch or indent clean coupons.
5. Same microscope and settings for all; a few hundred images.
6. Reuse the existing pipeline (texture SVM, CNN, Grad-CAM, line-shape features for cracks).

**Ask:** can we get lab and microscope access for a few days?

## Likely questions

**Why is 100 % not impressive?** NEU is small and clean, and many papers already get 98-100 %. The harder test is new factories, new lighting and microscope images.

**How do you know the model isn't cheating?** It only sees pixels, never file names. We test on images it never trained on, removed duplicate images, and for scratch vs crack we flipped brightness to rule out a shortcut.

**Why is the corrosion accuracy low?** The photos show whole bridge members, the grades are inspectors' judgement calls, and neighbouring grades overlap. It's a baseline, and it's the reason we want controlled lab samples.

**Why use simple features as well as deep learning?** They're explainable. For scratch vs crack we can say exactly why: number of line ends, branching, direction.

**Where does microstructure come in?** Corrosion type (pitting, intergranular, galvanic between phases) is visible in etched micrographs and depends on microstructure, for example chromium carbides at grain boundaries in sensitized stainless steel. That's the lab experiment above.
