# Module 1 - Surface Defect Inspector: results

Generated 2026-10-28T23:07:35 | git 54d2a14 | numbers = mean ± std over test folds (%, unless stated)

## Data

- images kept: **1799**, with boxes: 1799
- per class: {'crazing': 300, 'inclusion': 300, 'patches': 299, 'pitted_surface': 300, 'rolled-in_scale': 300, 'scratches': 300}
- boxes per class: {'crazing': 694, 'inclusion': 884, 'patches': 870, 'pitted_surface': 489, 'rolled-in_scale': 628, 'scratches': 621}
- duplicates dropped: 1, unparsed: 0, xml label mismatches: 188

![box_stats.png](eda/box_stats.png)
![class_counts.png](eda/class_counts.png)
![intensity_hist.png](eda/intensity_hist.png)
![samples_grid.png](eda/samples_grid.png)

## Main comparison

### Classical: LBP/GLCM + SVM

| featset | exp | folds | accuracy | macro_f1 |
|---|---|---|---|---|
| both | E001 | [0, 1, 2, 3, 4] | 99.44 ± 0.20 | 99.44 ± 0.20 |
| glcm | E001 | [0, 1, 2, 3, 4] | 98.17 ± 0.58 | 98.17 ± 0.57 |
| lbp | E001 | [0, 1, 2, 3, 4] | 98.11 ± 0.60 | 98.11 ± 0.60 |

### Classical: per-class F1

| featset | exp | folds | f1_crazing | f1_inclusion | f1_patches | f1_pitted_surface | f1_rolled-in_scale | f1_scratches |
|---|---|---|---|---|---|---|---|---|
| both | E001 | [0, 1, 2, 3, 4] | 100.00 ± 0.00 | 98.51 ± 0.69 | 100.00 ± 0.00 | 99.83 ± 0.38 | 100.00 ± 0.00 | 98.32 ± 0.59 |
| glcm | E001 | [0, 1, 2, 3, 4] | 98.18 ± 1.35 | 96.59 ± 1.60 | 98.14 ± 1.41 | 99.33 ± 0.71 | 100.00 ± 0.00 | 96.77 ± 1.32 |
| lbp | E001 | [0, 1, 2, 3, 4] | 100.00 ± 0.00 | 94.93 ± 1.96 | 99.83 ± 0.38 | 99.33 ± 1.10 | 100.00 ± 0.00 | 94.58 ± 2.28 |

### CNN classifiers

| model | exp | folds | accuracy | macro_f1 |
|---|---|---|---|---|
| resnet50 | SMOKE | [0] | 96.94 ± 0.00 | 96.93 ± 0.00 |

### CNN: per-class F1

| model | exp | folds | f1_crazing | f1_inclusion | f1_patches | f1_pitted_surface | f1_rolled-in_scale | f1_scratches |
|---|---|---|---|---|---|---|---|---|
| resnet50 | SMOKE | [0] | 90.91 ± 0.00 | 99.16 ± 0.00 | 92.31 ± 0.00 | 99.17 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

### CNN: cost

| model | exp | folds | train_minutes | best_epoch | ms_per_img_bs1 | peak_vram_gb |
|---|---|---|---|---|---|---|
| resnet50 | SMOKE | [0] | 0.45 ± 0.00 | 1.00 ± 0.00 | 6.48 ± 0.00 | 2.69 ± 0.00 |

### YOLO detection (test folds)

| model | exp | folds | mAP50 | mAP50_95 | precision | recall |
|---|---|---|---|---|---|---|
| yolov8n | SMOKE | [0] | 1.67 ± 0.00 | 0.36 ± 0.00 | 51.85 ± 0.00 | 11.08 ± 0.00 |

### YOLO: per-class AP50

| model | exp | folds | AP50_crazing | AP50_inclusion | AP50_patches | AP50_pitted_surface | AP50_rolled-in_scale | AP50_scratches |
|---|---|---|---|---|---|---|---|---|
| yolov8n | SMOKE | [0] | 0.01 ± 0.00 | 1.00 ± 0.00 | 2.85 ± 0.00 | 1.22 ± 0.00 | 0.01 ± 0.00 | 4.92 ± 0.00 |

### YOLO: cost

| model | exp | folds | train_minutes | ms_inference |
|---|---|---|---|---|
| yolov8n | SMOKE | [0] | 0.80 ± 0.00 | 3.19 ± 0.00 |

### Summary

| Method | Accuracy | Macro-F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| SVM both | 99.44 ± 0.20 | 99.44 ± 0.20 | - | - |
| SVM glcm | 98.17 ± 0.58 | 98.17 ± 0.57 | - | - |
| SVM lbp | 98.11 ± 0.60 | 98.11 ± 0.60 | - | - |
| resnet50 | 96.94 ± 0.00 | 96.93 ± 0.00 | - | - |
| yolov8n | - | - | 1.67 ± 0.00 | 0.36 ± 0.00 |

Literature reference: NEU-CLS CNNs ~98-100% acc; NEU-DET YOLOv8n ~73% mAP@0.5 (see decisions.md D-012).

## Figures

### Classical
![confusion_both.png](classical/confusion_both.png)
![confusion_glcm.png](classical/confusion_glcm.png)
![confusion_lbp.png](classical/confusion_lbp.png)

### Grad-CAM

## Observations (OpenCode fills in, Claude reviews)

- ...
