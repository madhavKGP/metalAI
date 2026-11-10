# Module 1 - Surface Defect Inspector: results

Generated 2026-11-10T05:58:22 | git 6add1fe | numbers = mean ± std over test folds (%, unless stated)

## Data

- images kept: **1799**, with boxes: 1799
- per class: {'crazing': 300, 'inclusion': 300, 'patches': 299, 'pitted_surface': 300, 'rolled-in_scale': 300, 'scratches': 300}
- boxes per class: {'crazing': 689, 'inclusion': 1011, 'patches': 878, 'pitted_surface': 432, 'rolled-in_scale': 628, 'scratches': 548}
- duplicates dropped: 1, unparsed: 0, xml label mismatches: 0

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
| efficientnet_b0 | E003 | [0, 1, 2, 3, 4] | 100.00 ± 0.00 | 100.00 ± 0.00 |
| resnet50 | E002 | [0, 1, 2, 3, 4] | 99.94 ± 0.12 | 99.94 ± 0.12 |

### CNN: per-class F1

| model | exp | folds | f1_crazing | f1_inclusion | f1_patches | f1_pitted_surface | f1_rolled-in_scale | f1_scratches |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | E003 | [0, 1, 2, 3, 4] | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| resnet50 | E002 | [0, 1, 2, 3, 4] | 100.00 ± 0.00 | 99.83 ± 0.37 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.83 ± 0.38 |

### CNN: cost

| model | exp | folds | train_minutes | best_epoch | ms_per_img_bs1 | peak_vram_gb |
|---|---|---|---|---|---|---|
| efficientnet_b0 | E003 | [0, 1, 2, 3, 4] | 4.49 ± 1.74 | 9.40 ± 5.94 | 9.29 ± 0.99 | 1.45 ± 0.03 |
| resnet50 | E002 | [0, 1, 2, 3, 4] | 5.46 ± 1.04 | 7.40 ± 2.51 | 7.31 ± 0.60 | 2.08 ± 0.55 |

### YOLO inner-validation (all runs, used for selection, D-022)

| model | exp | folds | val_mAP50 | val_mAP50_95 |
|---|---|---|---|---|
| E005 yolov8n | E005 | [0] | 77.88 ± 0.00 | 44.77 ± 0.00 |
| E006 yolov8n | E006 | [0] | 76.05 ± 0.00 | 44.54 ± 0.00 |

Selected detector: **E005** (configs/app.yaml). Test-fold numbers below are for the selected run only.

### YOLO detection (test folds)

| model | exp | folds | mAP50 | mAP50_95 | precision | recall |
|---|---|---|---|---|---|---|
| E005 yolov8n | E005 | [0] | 75.37 ± 0.00 | 42.57 ± 0.00 | 71.05 ± 0.00 | 70.77 ± 0.00 |

### YOLO: per-class AP50

| model | exp | folds | AP50_crazing | AP50_inclusion | AP50_patches | AP50_pitted_surface | AP50_rolled-in_scale | AP50_scratches |
|---|---|---|---|---|---|---|---|---|
| E005 yolov8n | E005 | [0] | 44.66 ± 0.00 | 83.37 ± 0.00 | 92.44 ± 0.00 | 81.35 ± 0.00 | 59.59 ± 0.00 | 90.81 ± 0.00 |

### YOLO: cost

| model | exp | folds | train_minutes | ms_inference |
|---|---|---|---|---|
| E005 yolov8n | E005 | [0] | 4.30 ± 0.00 | 6.91 ± 0.00 |

### Scratch vs crack: line-shape features (E008, D-030)

| model | exp | folds | accuracy | macro_f1 | auc | accuracy_inverted_test |
|---|---|---|---|---|---|---|
| logreg | E008 | [0, 1, 2, 3, 4] | 99.67 ± 0.75 | 99.67 ± 0.75 | 100.00 ± 0.00 | 99.83 ± 0.37 |
| random_forest | E008 | [0, 1, 2, 3, 4] | 99.67 ± 0.75 | 99.67 ± 0.75 | 99.99 ± 0.01 | 99.67 ± 0.75 |

`accuracy_inverted_test` = same trained model tested on brightness-inverted images: the result holds, so the model uses line shape, not line brightness.

| feature | AUC alone | scratch median | crack median | RF importance |
|---|---|---|---|---|
| endpoint_density | 1.000 | 2.16 | 11.19 | 0.325 |
| fragments_per_100 | 1.000 | 0.63 | 3.99 | 0.268 |
| longest_frac | 0.996 | 0.73 | 0.17 | 0.164 |
| orient_coherence | 0.987 | 1.00 | 0.38 | 0.178 |
| branch_density | 0.909 | 3.39 | 11.42 | 0.045 |
| straightness | 0.686 | 0.89 | 0.83 | 0.007 |
| width_cv | 0.669 | 0.30 | 0.33 | 0.013 |

### Corrosion condition, PUBLIC bridge photos (Jackson 2025, MIT) - colour+texture SVM (E009, D-031)

| setting | exp | folds | accuracy | macro_f1 |
|---|---|---|---|---|
| 4class / grouped | E009 | [0, 1, 2, 3, 4] | 45.94 ± 2.74 | 43.42 ± 2.51 |
| 4class / random | E009 | [0, 1, 2, 3, 4] | 51.48 ± 3.07 | 48.61 ± 2.24 |
| binary / grouped | E009 | [0, 1, 2, 3, 4] | 72.48 ± 3.71 | 72.46 ± 3.70 |
| binary / random | E009 | [0, 1, 2, 3, 4] | 74.28 ± 2.26 | 74.25 ± 2.25 |

- images used: 3180 (dropped 43 exact duplicates and 7 files of 3 photos labelled with conflicting classes); majority-class baseline: 4class 39.2 %, binary 52.0 %.
- `grouped` keeps photos of the same structure (near-consecutive photo numbers) in one fold; `random` does not and is shown only to measure leakage. Quote the grouped number.

### Summary

| Method | Accuracy | Macro-F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| SVM both | 99.44 ± 0.20 | 99.44 ± 0.20 | - | - |
| SVM glcm | 98.17 ± 0.58 | 98.17 ± 0.57 | - | - |
| SVM lbp | 98.11 ± 0.60 | 98.11 ± 0.60 | - | - |
| efficientnet_b0 | 100.00 ± 0.00 | 100.00 ± 0.00 | - | - |
| resnet50 | 99.94 ± 0.12 | 99.94 ± 0.12 | - | - |
| E005 yolov8n | - | - | 75.37 ± 0.00 | 42.57 ± 0.00 |

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
![fold1_confusion.png](cnn/E002_resnet50/fold1_confusion.png)
![fold1_curves.png](cnn/E002_resnet50/fold1_curves.png)
![fold2_confusion.png](cnn/E002_resnet50/fold2_confusion.png)
![fold2_curves.png](cnn/E002_resnet50/fold2_curves.png)
![fold3_confusion.png](cnn/E002_resnet50/fold3_confusion.png)
![fold3_curves.png](cnn/E002_resnet50/fold3_curves.png)
![fold4_confusion.png](cnn/E002_resnet50/fold4_confusion.png)
![fold4_curves.png](cnn/E002_resnet50/fold4_curves.png)

### CNN E003_efficientnet_b0
![confusion_all.png](cnn/E003_efficientnet_b0/confusion_all.png)
![fold0_confusion.png](cnn/E003_efficientnet_b0/fold0_confusion.png)
![fold0_curves.png](cnn/E003_efficientnet_b0/fold0_curves.png)
![fold1_confusion.png](cnn/E003_efficientnet_b0/fold1_confusion.png)
![fold1_curves.png](cnn/E003_efficientnet_b0/fold1_curves.png)
![fold2_confusion.png](cnn/E003_efficientnet_b0/fold2_confusion.png)
![fold2_curves.png](cnn/E003_efficientnet_b0/fold2_curves.png)
![fold3_confusion.png](cnn/E003_efficientnet_b0/fold3_confusion.png)
![fold3_curves.png](cnn/E003_efficientnet_b0/fold3_curves.png)
![fold4_confusion.png](cnn/E003_efficientnet_b0/fold4_confusion.png)
![fold4_curves.png](cnn/E003_efficientnet_b0/fold4_curves.png)

### Grad-CAM
![E002_resnet50_fold0_boxes.png](gradcam/E002_resnet50_fold0_boxes.png)
![E002_resnet50_fold0_correct.png](gradcam/E002_resnet50_fold0_correct.png)
![E002_resnet50_fold3_boxes.png](gradcam/E002_resnet50_fold3_boxes.png)
![E002_resnet50_fold3_correct.png](gradcam/E002_resnet50_fold3_correct.png)
![E002_resnet50_fold3_errors.png](gradcam/E002_resnet50_fold3_errors.png)
![E003_efficientnet_b0_fold0_boxes.png](gradcam/E003_efficientnet_b0_fold0_boxes.png)
![E003_efficientnet_b0_fold0_correct.png](gradcam/E003_efficientnet_b0_fold0_correct.png)

### Scratch vs crack
![confusion_logreg.png](scratch_crack/confusion_logreg.png)
![confusion_random_forest.png](scratch_crack/confusion_random_forest.png)
![examples.png](scratch_crack/examples.png)
![feature_boxplots.png](scratch_crack/feature_boxplots.png)

### Corrosion condition (public data)
![confusion_4class.png](corrosion/confusion_4class.png)
![confusion_binary.png](corrosion/confusion_binary.png)

### YOLO E004_yolov8n_fold0
![test_BoxF1_curve.png](yolo/E004_yolov8n_fold0/test_BoxF1_curve.png)
![test_BoxPR_curve.png](yolo/E004_yolov8n_fold0/test_BoxPR_curve.png)
![test_BoxP_curve.png](yolo/E004_yolov8n_fold0/test_BoxP_curve.png)
![test_BoxR_curve.png](yolo/E004_yolov8n_fold0/test_BoxR_curve.png)
![test_confusion_matrix.png](yolo/E004_yolov8n_fold0/test_confusion_matrix.png)
![test_confusion_matrix_normalized.png](yolo/E004_yolov8n_fold0/test_confusion_matrix_normalized.png)
![train_BoxF1_curve.png](yolo/E004_yolov8n_fold0/train_BoxF1_curve.png)
![train_BoxPR_curve.png](yolo/E004_yolov8n_fold0/train_BoxPR_curve.png)
![train_BoxP_curve.png](yolo/E004_yolov8n_fold0/train_BoxP_curve.png)
![train_BoxR_curve.png](yolo/E004_yolov8n_fold0/train_BoxR_curve.png)
![train_confusion_matrix.png](yolo/E004_yolov8n_fold0/train_confusion_matrix.png)
![train_confusion_matrix_normalized.png](yolo/E004_yolov8n_fold0/train_confusion_matrix_normalized.png)
![train_results.png](yolo/E004_yolov8n_fold0/train_results.png)

### YOLO E005_yolov8n_fold0
![test_BoxF1_curve.png](yolo/E005_yolov8n_fold0/test_BoxF1_curve.png)
![test_BoxPR_curve.png](yolo/E005_yolov8n_fold0/test_BoxPR_curve.png)
![test_BoxP_curve.png](yolo/E005_yolov8n_fold0/test_BoxP_curve.png)
![test_BoxR_curve.png](yolo/E005_yolov8n_fold0/test_BoxR_curve.png)
![test_confusion_matrix.png](yolo/E005_yolov8n_fold0/test_confusion_matrix.png)
![test_confusion_matrix_normalized.png](yolo/E005_yolov8n_fold0/test_confusion_matrix_normalized.png)
![train_BoxF1_curve.png](yolo/E005_yolov8n_fold0/train_BoxF1_curve.png)
![train_BoxPR_curve.png](yolo/E005_yolov8n_fold0/train_BoxPR_curve.png)
![train_BoxP_curve.png](yolo/E005_yolov8n_fold0/train_BoxP_curve.png)
![train_BoxR_curve.png](yolo/E005_yolov8n_fold0/train_BoxR_curve.png)
![train_confusion_matrix.png](yolo/E005_yolov8n_fold0/train_confusion_matrix.png)
![train_confusion_matrix_normalized.png](yolo/E005_yolov8n_fold0/train_confusion_matrix_normalized.png)
![train_results.png](yolo/E005_yolov8n_fold0/train_results.png)

### YOLO E006_yolov8n_fold0
![test_BoxF1_curve.png](yolo/E006_yolov8n_fold0/test_BoxF1_curve.png)
![test_BoxPR_curve.png](yolo/E006_yolov8n_fold0/test_BoxPR_curve.png)
![test_BoxP_curve.png](yolo/E006_yolov8n_fold0/test_BoxP_curve.png)
![test_BoxR_curve.png](yolo/E006_yolov8n_fold0/test_BoxR_curve.png)
![test_confusion_matrix.png](yolo/E006_yolov8n_fold0/test_confusion_matrix.png)
![test_confusion_matrix_normalized.png](yolo/E006_yolov8n_fold0/test_confusion_matrix_normalized.png)
![train_BoxF1_curve.png](yolo/E006_yolov8n_fold0/train_BoxF1_curve.png)
![train_BoxPR_curve.png](yolo/E006_yolov8n_fold0/train_BoxPR_curve.png)
![train_BoxP_curve.png](yolo/E006_yolov8n_fold0/train_BoxP_curve.png)
![train_BoxR_curve.png](yolo/E006_yolov8n_fold0/train_BoxR_curve.png)
![train_confusion_matrix.png](yolo/E006_yolov8n_fold0/train_confusion_matrix.png)
![train_confusion_matrix_normalized.png](yolo/E006_yolov8n_fold0/train_confusion_matrix_normalized.png)
![train_results.png](yolo/E006_yolov8n_fold0/train_results.png)

## Observations (OpenCode fills in, Claude reviews)

- ...
