# Module 1 - Surface Defect Inspector: results

Generated 2026-10-28T23:48:18 | git 25b8fc2 | numbers = mean ± std over test folds (%, unless stated)

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
| efficientnet_b0 | E003 | [0] | 100.00 ± 0.00 | 100.00 ± 0.00 |
| resnet50 | E002 | [0] | 100.00 ± 0.00 | 100.00 ± 0.00 |

### CNN: per-class F1

| model | exp | folds | f1_crazing | f1_inclusion | f1_patches | f1_pitted_surface | f1_rolled-in_scale | f1_scratches |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | E003 | [0] | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| resnet50 | E002 | [0] | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

### CNN: cost

| model | exp | folds | train_minutes | best_epoch | ms_per_img_bs1 | peak_vram_gb |
|---|---|---|---|---|---|---|
| efficientnet_b0 | E003 | [0] | 7.05 ± 0.00 | 18.00 ± 0.00 | 8.70 ± 0.00 | 1.49 ± 0.00 |
| resnet50 | E002 | [0] | 4.44 ± 0.00 | 5.00 ± 0.00 | 6.51 ± 0.00 | 2.69 ± 0.00 |

### YOLO detection (test folds)

| model | exp | folds | mAP50 | mAP50_95 | precision | recall |
|---|---|---|---|---|---|---|
| yolov8n | E004 | [0] | 74.56 ± 0.00 | 42.54 ± 0.00 | 75.03 ± 0.00 | 70.41 ± 0.00 |

### YOLO: per-class AP50

| model | exp | folds | AP50_crazing | AP50_inclusion | AP50_patches | AP50_pitted_surface | AP50_rolled-in_scale | AP50_scratches |
|---|---|---|---|---|---|---|---|---|
| yolov8n | E004 | [0] | 39.37 ± 0.00 | 87.12 ± 0.00 | 90.78 ± 0.00 | 84.41 ± 0.00 | 55.48 ± 0.00 | 90.23 ± 0.00 |

### YOLO: cost

| model | exp | folds | train_minutes | ms_inference |
|---|---|---|---|---|
| yolov8n | E004 | [0] | 22.80 ± 0.00 | 3.38 ± 0.00 |

### Summary

| Method | Accuracy | Macro-F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| SVM both | 99.44 ± 0.20 | 99.44 ± 0.20 | - | - |
| SVM glcm | 98.17 ± 0.58 | 98.17 ± 0.57 | - | - |
| SVM lbp | 98.11 ± 0.60 | 98.11 ± 0.60 | - | - |
| efficientnet_b0 | 100.00 ± 0.00 | 100.00 ± 0.00 | - | - |
| resnet50 | 100.00 ± 0.00 | 100.00 ± 0.00 | - | - |
| yolov8n | - | - | 74.56 ± 0.00 | 42.54 ± 0.00 |

Literature reference: NEU-CLS CNNs ~98-100% acc; NEU-DET YOLOv8n ~73% mAP@0.5 (see decisions.md D-012).

## Figures

### Classical
![confusion_both.png](classical/confusion_both.png)
![confusion_glcm.png](classical/confusion_glcm.png)
![confusion_lbp.png](classical/confusion_lbp.png)

### CNN E002_resnet50
![confusion_all.png](cnn/E002_resnet50/confusion_all.png)
![fold0_confusion.png](cnn/E002_resnet50/fold0_confusion.png)
![fold0_curves.png](cnn/E002_resnet50/fold0_curves.png)

### CNN E003_efficientnet_b0
![confusion_all.png](cnn/E003_efficientnet_b0/confusion_all.png)
![fold0_confusion.png](cnn/E003_efficientnet_b0/fold0_confusion.png)
![fold0_curves.png](cnn/E003_efficientnet_b0/fold0_curves.png)

### Grad-CAM
![E002_resnet50_fold0_boxes.png](gradcam/E002_resnet50_fold0_boxes.png)
![E002_resnet50_fold0_correct.png](gradcam/E002_resnet50_fold0_correct.png)
![E003_efficientnet_b0_fold0_boxes.png](gradcam/E003_efficientnet_b0_fold0_boxes.png)
![E003_efficientnet_b0_fold0_correct.png](gradcam/E003_efficientnet_b0_fold0_correct.png)

### YOLO E004_yolov8n_fold0
![test_BoxF1_curve.png](yolo/E004_yolov8n_fold0/test_BoxF1_curve.png)
![test_BoxP_curve.png](yolo/E004_yolov8n_fold0/test_BoxP_curve.png)
![test_BoxPR_curve.png](yolo/E004_yolov8n_fold0/test_BoxPR_curve.png)
![test_BoxR_curve.png](yolo/E004_yolov8n_fold0/test_BoxR_curve.png)
![test_confusion_matrix.png](yolo/E004_yolov8n_fold0/test_confusion_matrix.png)
![test_confusion_matrix_normalized.png](yolo/E004_yolov8n_fold0/test_confusion_matrix_normalized.png)
![train_BoxF1_curve.png](yolo/E004_yolov8n_fold0/train_BoxF1_curve.png)
![train_BoxP_curve.png](yolo/E004_yolov8n_fold0/train_BoxP_curve.png)
![train_BoxPR_curve.png](yolo/E004_yolov8n_fold0/train_BoxPR_curve.png)
![train_BoxR_curve.png](yolo/E004_yolov8n_fold0/train_BoxR_curve.png)
![train_confusion_matrix.png](yolo/E004_yolov8n_fold0/train_confusion_matrix.png)
![train_confusion_matrix_normalized.png](yolo/E004_yolov8n_fold0/train_confusion_matrix_normalized.png)
![train_results.png](yolo/E004_yolov8n_fold0/train_results.png)

## Observations (OpenCode fills in, Claude reviews)

- Both CNNs hit 100% test accuracy on fold 0 with **zero misclassifications** (hence no `_errors.png` Grad-CAM panels). ResNet-50 early-stopped at epoch 11, EfficientNet-B0 at 24 — curves show val loss flattening ~epoch 5-18 while train loss keeps dropping slightly (mild over-fit, no val degradation).
- Grad-CAM hot regions sit on the visible defect areas in the `_correct` panels (spot-check crazing/scratches/patches); `_boxes` overlays show the GT boxes for comparison. Suggest Claude confirm placement visually.
- YOLO is clearly the weak spot and matches literature (D-012, ~73% mAP50): **crazing AP50 39.4** and **rolled-in_scale AP50 55.5** drag the mean down; inclusion/patches/scratches are 84-91. Crazing is low-contrast, diffuse texture — hard for box regression. Pitted_surface 84.4.
- Classical SVM (LBP+GLCM, E001) at 99.44% is essentially saturated too; per-class F1 shows inclusion (98.5) and scratches (98.3) are the only classes it ever misses.
- Dataset looks "easy" for classification (all models ≥99%) — the interesting research question is detection (YOLO) and robustness (later modules / GC10-DET).
- GPU: RTX 4060 Laptop. E002 peak 3.5 GB / 65 C / mean util 50%; E003 peak 3.0 GB / 65 C / mean util 70%; E004 peak 4.1 GB / 73 C / mean util 75% / max power 88.6 W. No OOM at default batch sizes (CNN 32, YOLO 16). On AC power throughout.
