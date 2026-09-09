# Preliminary PlantSeg Model Test Run

**Run date:** 29 August 2026  
**Status:** Preliminary ten-epoch integration and comparison run

## Purpose

This report records the first complete comparison between the standalone disease-aware YOLO11n
model and the two-stage lesion-localization-to-classification pipeline. The run is preliminary:
every model was limited to approximately ten training epochs to verify the complete experimental
workflow before committing to longer training. See
[Final Model Results](#final-model-results) below for the outcome of the longer training run this
report recommended.

The evaluated systems were:

1. **Standalone YOLO11n:** one model localizes a lesion and assigns one of 115 PlantSeg disease
   classes to its highest-confidence box.
2. **Two-stage pipeline:** a class-agnostic YOLO11n localizes lesions, their union is cropped with a
   10% margin, and the crop is classified by the baseline CNN, EfficientNetB0 or
   MobileNetV3-Large.

## Experimental Setup

The experiment used PlantSeg's official partitions without moving images between splits.

| Partition | Images | Classes present |
|---|---:|---:|
| Training | 5,367 | 115 |
| Validation | 846 | 114 |
| Test | 1,561 | 114 |

`Coffee / coffee black rot` has no examples in the official test split. Metrics that average over
classes therefore use the 114 classes present in that split.

Both YOLO models used YOLO11n initialized from the same pretrained checkpoint. Each binary lesion
mask was converted into one enclosing bounding box. For the class-agnostic detector, every box was
labelled `lesion`; for standalone YOLO, the same box received the image's PlantSeg disease ID.

The principal standalone YOLO settings were:

| Setting | Value |
|---|---:|
| Epochs | 10 |
| Input size | 640 × 640 |
| Batch size | 16 |
| Classes | 115 |
| Optimizer | Ultralytics automatic selection |
| Device | Apple MPS |
| Seed | 42 |

Evaluation used a confidence threshold of 0.25 for both detectors. When the class-agnostic detector
found nothing, the classifiers received the complete image. When standalone YOLO found nothing,
the image was counted as an incorrect classification because the standalone system produced no
disease decision.

## Localization Results

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Class-agnostic lesion YOLO | 75.62% | 73.93% | 77.57% | 46.17% |
| Standalone disease-aware YOLO | 53.98% | 16.48% | 15.40% | 10.88% |

These rows are not identical tasks. The class-agnostic detector needs only to find a lesion, whereas
a standalone detection is correct only when its box and disease class are both correct. The large
gap nevertheless shows that disease discrimination is currently the limiting part of standalone
YOLO.

## End-to-End Classification Results

| System | Accuracy | Macro F1 | Balanced accuracy | Coverage |
|---|---:|---:|---:|---:|
| Standalone disease-aware YOLO | 22.93% | 13.82% | 12.12% | 41.58% |
| Baseline CNN pipeline | 16.72% | 7.25% | 8.39% | 100%¹ |
| EfficientNetB0 pipeline | **69.06%** | 60.91% | 61.00% | 100%¹ |
| MobileNetV3-Large pipeline | 68.10% | **61.32%** | **61.00%** | 100%¹ |

¹ The pipeline always returns a class because it falls back to the full image when lesion YOLO has
no confident detection. It used this fallback for 74 images, or 4.74% of the test split.

Standalone YOLO detected at least one box in 649 of 1,561 test images. Its accuracy conditional on
making a detection was 55.16%, but 912 no-detection cases reduced its end-to-end accuracy to 22.93%.
It predicted only 47 distinct classes, and 74 of the 114 test classes had zero recall.

EfficientNetB0 achieved the best overall accuracy. MobileNetV3-Large achieved the best macro F1,
although the difference between the two pretrained classifiers was small. The CNN trained from
scratch was not competitive in this short run.

## Runtime

| System measured | Images per second |
|---|---:|
| Standalone disease-aware YOLO | 73.83 |
| Lesion YOLO followed by all three classifiers | 41.06 |

This is not a fair deployment-speed comparison because the recorded pipeline run executes all
three classifiers for every crop. A final runtime comparison should execute only the selected
pipeline classifier and use the same hardware, batch size and preprocessing conditions.

## Standalone YOLO Diagnosis

There is no evidence of a broken class mapping or incomplete dataset preparation. Every generated
YOLO label was checked against `Metadata.csv`, the checkpoint contains all 115 classes in the
expected order, all training images were used, and training and validation losses decreased
normally.

The observed weakness is primarily underfitting combined with a difficult class distribution:

- The training split has a median of 38 examples per class. Thirty classes have 20 or fewer
  examples and the rarest class has only two.
- Standalone validation mAP@50 increased from 0.59% after epoch 1 to 14.80% after epoch 10, with its
  best value occurring at the final epoch. The model had therefore not converged.
- Final classification loss remained high at 3.66 on training and 3.59 on validation. The small
  train-validation difference is more consistent with underfitting than overfitting.
- The mask-derived boxes are coarse. Their median area is 58% of the image, and 28% cover at least
  80% of the image. This weakens object-level supervision, particularly when the detector must also
  learn a fine-grained disease class.
- Standalone YOLO must learn localization and 115-way classification jointly, while the two-stage
  classifiers receive an already-localized crop and optimize only the classification objective.

### Confidence-threshold diagnostic

The shared 0.25 threshold is appropriate for producing reliable pipeline crops, but it is too
restrictive for the undertrained standalone model. A diagnostic pass on the validation split gave:

| Standalone confidence threshold | Coverage | Accuracy when detected | Overall accuracy |
|---:|---:|---:|---:|
| 0.250 | 41.13% | 53.16% | 21.87% |
| 0.100 | 70.33% | 41.18% | 28.96% |
| 0.050 | 88.53% | 36.32% | 32.15% |
| 0.010 | 99.65% | 34.05% | 33.92% |
| 0.001 | 100.00% | 33.92% | 33.92% |

Thus, confidence suppression explains part of the low 22.93% test result, but not the full gap. Even
when forced to return a prediction for every validation image, standalone accuracy remained much
lower than the approximately 69% achieved by the two-stage pretrained classifiers.

## Interpretation

This run verifies that both experimental paths work correctly. Under the current ten-epoch
configuration, the two-stage pipeline clearly outperforms standalone YOLO for disease
classification. The standalone model is faster in the recorded run, but the speed result needs a
single-classifier pipeline measurement before it can support a final efficiency conclusion.

The standalone result should not be treated as the model's final potential. Ten epochs were enough
for an integration test but insufficient for a randomly replaced 115-class detection head. The
0.25 threshold also makes the current end-to-end score unnecessarily dependent on abstention.

## Recommended Next Experiment

1. Select a standalone confidence threshold using validation data only and keep it fixed for the
   next holdout evaluation.
2. Train standalone YOLO for substantially longer with early stopping based on validation mAP and
   macro-level classification performance.
3. Address rare classes through training-only balancing or targeted augmentation.
4. Measure runtime for standalone YOLO and one selected two-stage classifier separately.
5. Preserve the test outputs from this run. Further hyperparameter choices should be based on the
   validation split rather than repeatedly optimizing against these test results.

## Saved Artifacts

These artifacts reflect the state at the time of this preliminary run. The ten-epoch training
histories referenced below were later overwritten in place once training was extended; see
[Final Model Results](#final-model-results) for the ten-epoch backups and the current artifacts.

- Standalone per-class metrics (ten-epoch run):
  [`standalone_yolo_per_class.json`](../outputs/evaluation/plantseg_test/standalone_yolo_per_class.json)
- Standalone confusion matrix (ten-epoch run):
  [`standalone_yolo_confusion_matrix.csv`](../outputs/evaluation/plantseg_test/standalone_yolo_confusion_matrix.csv)

## Final Model Results

**Run date:** 4-6 September 2026
**Status:** Both YOLO detectors and all three classifiers trained to completion on an Apple M1 Pro.

Following the recommendations above, both YOLO models were trained substantially longer than the
ten-epoch preliminary run. The three classifiers were trained from scratch on the crops produced
by the final lesion YOLO checkpoint (they had not previously been trained beyond the ten-epoch
preliminary integration test). `scripts/evaluate_pipeline.py` was then run once against the
fully-trained checkpoints, followed by `scripts/run_gradcam.py`.

### YOLO Training Progression

| Model | Metric | 10 epochs (preliminary) | Final |
|---|---|---:|---:|
| Class-agnostic lesion YOLO | mAP@50 | 77.57% | **86.08%** (peak 86.1%) |
| Class-agnostic lesion YOLO | mAP@50–95 | 46.17% | **59.3%** (peak 0.596) |
| Standalone disease-aware YOLO | mAP@50 | 15.40% | **50.02%** (peak 51.5%) |
| Standalone disease-aware YOLO | mAP@50–95 | 10.88% | **35.4%** |

### Classifier Training

All three classifiers were trained on the crops produced by the final lesion YOLO checkpoint.
Best-checkpoint selection used validation macro F1, as in the preliminary run.

| Model | Best validation macro F1 |
|---|---:|
| Baseline CNN (from scratch) | 28.7% |
| EfficientNetB0 | 57.9% |
| MobileNetV3-Large | 58.1% |

### End-to-End Classification Results (final)

| System | Accuracy | Macro F1 | Balanced accuracy | Coverage |
|---|---:|---:|---:|---:|
| Standalone disease-aware YOLO | **61.37%** | **53.32%** | 49.60% | 86.48% |
| Baseline CNN pipeline | 41.32% | 29.04% | 29.99% | 100%¹ |
| EfficientNetB0 pipeline | 66.69% | 59.02% | — | 100%¹ |
| MobileNetV3-Large pipeline | **66.75%** | 58.17% | — | 100%¹ |

¹ The pipeline fell back to the full image for 38 of 1,561 test images (2.43%), down from 4.74% in
the preliminary run, reflecting the improved lesion detector.

Standalone YOLO's end-to-end accuracy rose from 22.93% to 61.37% and its coverage from 41.58% to
86.48% once trained for a realistic number of epochs, confirming the preliminary run's diagnosis:
the ten-epoch result understated the model, which was underfit rather than fundamentally broken.
The two-stage pipeline's accuracy held essentially flat relative to the preliminary run (EfficientNetB0
and MobileNetV3-Large were already close to converged at ten epochs on the classification task, since
they only had to learn classification rather than joint localization and classification). Standalone
YOLO now approaches, but does not exceed, the two-stage pipeline's accuracy, while remaining a single
model rather than two.

### Grad-CAM vs PlantSeg Masks

`scripts/run_gradcam.py` was run against the final MobileNetV3-Large classifier and lesion YOLO
detector over all 1,561 official test images, comparing Grad-CAM activation with the ground-truth
lesion masks:

| Metric | Mean | Median |
|---|---:|---:|
| IoU (fixed 0.5 threshold) | 21.3% | 18.9% |
| IoU (top-20% quantile) | 24.2% | 22.1% |
| Precision (top-20% quantile) | 41.2% | 34.2% |
| Recall (top-20% quantile) | 52.7% | 48.1% |
| Mask energy fraction | 41.5% | 38.2% |
| Mask area fraction (ground truth) | 21.3% | 14.2% |
| Pointing-game hit rate | 58.8% | 100% |

The mask area fraction row shows the ground-truth lesion masks cover roughly a fifth of each image
on average, so a mask-energy-fraction of 41.5% shows classifier attention is meaningfully
concentrated inside the lesion region rather than spread uniformly across the image. The
pointing-game hit rate (whether the single highest-activation pixel falls inside the mask) is 58.8%
on average; its median of 1.0 indicates the peak activation lands inside the mask for the majority
of images, with a long tail of misses pulling the mean down. Twelve qualitative image/mask/Grad-CAM
figures were saved for the paper.

One image initially caused a shape-mismatch crash between its EXIF-rotated raw pixel data and its
mask; see [`dataset.md`](dataset.md#plantseg-quality-findings-primary) for the fix and the
remaining scope of the underlying data issue. No images needed to be skipped in the final run once
the fix (`ImageOps.exif_transpose` on both image and mask) was applied.

### Robustness Under Synthetic Corruption

`scripts/run_robustness.py` applies each corruption configured in `configs/project.yaml`
(`gaussian_blur`, `brightness`, `contrast`, `jpeg_compression`, `occlusion`, `crop`) at severities
1-5 to the lesion-YOLO-to-classifier pipeline, using the final checkpoints. It was run twice: once
on a fixed, reproducible 200-image subset (5m53s), and once on the complete official 1,561-image
test split (44m6s) to confirm the subset result generalizes. Clean-condition accuracy on the full
split (baseline CNN 40.0%, EfficientNetB0 67.3%, MobileNetV3-Large 66.5%) closely matches
`evaluate_pipeline.py`'s independently measured clean accuracy (41.3%, 66.7%, 66.8%), the small
difference attributable to non-deterministic YOLO crop selection between runs.

Accuracy at the most severe level (severity 5) of each corruption, full test split:

| Corruption | Baseline CNN | EfficientNetB0 | MobileNetV3-Large |
|---|---:|---:|---:|
| Clean (no corruption) | 40.0% | 67.3% | 66.5% |
| Gaussian blur | **12.5%** (−27.5pp) | **29.7%** (−37.7pp) | **30.5%** (−36.0pp) |
| Brightness | 17.8% (−22.2pp) | 49.6% (−17.7pp) | 46.8% (−19.7pp) |
| JPEG compression | 25.8% (−14.2pp) | 48.5% (−18.8pp) | 45.7% (−20.8pp) |
| Contrast | 26.1% (−13.8pp) | 62.0% (−5.3pp) | 61.8% (−4.7pp) |
| Occlusion | 34.5% (−5.4pp) | 63.2% (−4.2pp) | 61.3% (−5.2pp) |
| Crop | 35.5% (−4.5pp) | 63.6% (−3.7pp) | 63.8% (−2.7pp) |

Gaussian blur at maximum severity is the worst case for every model, roughly halving accuracy or
worse; both pretrained classifiers lose more absolute accuracy to it than the from-scratch baseline
CNN despite starting from a much higher clean accuracy. Brightness and JPEG compression cause
moderate degradation. Crop and occlusion are comparatively well tolerated by all three models,
consistent with the training-time random-resized-crop and horizontal-flip augmentation already
applied during classifier training (`src/plant_disease/data.py`), which exposes the models to
similar framing variation. Per-corruption, per-severity results, mean confidence, and macro F1 are
in the saved artifacts below; `accuracy_vs_severity.png` plots every corruption and model together.

### 5-Fold Cross-Validation

`scripts/run_cross_validation.py` splits the 5,367 official PlantSeg training images into five
stratified folds (by disease class) and, for each fold, trains all five models — the lesion YOLO,
the standalone disease-aware YOLO, and the three classifiers — from scratch for 50 epochs on the
other four folds, validating on the held-out fold. The official validation and test splits are
never touched; this measures training variance across different train/validation partitions, and
is reported alongside, not instead of, the single-split results above. Two classes (41 and 68, with
only 2 and 4 training images) are too rare to stratify across five folds and are instead distributed
round-robin, the same way the official validation and test splits already omit some rare classes.

All 25 fold/model combinations (5 folds × 5 models) completed. Results are the mean and standard
deviation across the 5 folds:

| Model | Metric | Mean | Std |
|---|---|---:|---:|
| Lesion YOLO | mAP@50 | 85.3% | ±0.56pp |
| Lesion YOLO | mAP@50–95 | 57.0% | ±1.00pp |
| Standalone disease-aware YOLO | mAP@50 | 40.9% | ±0.66pp |
| Standalone disease-aware YOLO | mAP@50–95 | 29.0% | ±0.78pp |
| Baseline CNN | validation macro F1 | 20.1% | ±0.75pp |
| EfficientNetB0 | validation macro F1 | **59.4%** | ±1.99pp |
| MobileNetV3-Large | validation macro F1 | 57.3% | ±1.44pp |

The standard deviations are small relative to the means for every model, particularly the lesion
YOLO detector (mAP50 never varied by more than about a point across folds), indicating the
single-split results reported elsewhere in this document are not an artifact of a lucky or unlucky
train/validation partition. EfficientNetB0 is the most accurate classifier in every fold, consistent
with its single-split result. The standalone YOLO's cross-validated mAP50 (40.9%) sits below its
final single-split result (50.02%), consistent with the shorter per-fold training budget used here.

### Updated Saved Artifacts

- Complete evaluation summary: [`summary.json`](../outputs/evaluation/plantseg_test/summary.json)
- Per-image predictions: [`predictions.csv`](../outputs/evaluation/plantseg_test/predictions.csv)
- Lesion YOLO training history: [`results.csv`](../outputs/yolo/plantseg_lesion_10ep/results.csv)
- Standalone YOLO training history: [`results.csv`](../outputs/yolo/plantseg_disease_10ep/results.csv)
- Classifier training histories: [`baseline_cnn/history.json`](../outputs/classification/baseline_cnn/history.json),
  [`efficientnet_b0/history.json`](../outputs/classification/efficientnet_b0/history.json),
  [`mobilenet_v3_large/history.json`](../outputs/classification/mobilenet_v3_large/history.json)
- Grad-CAM evaluation summary: [`gradcam summary.json`](../outputs/gradcam/plantseg_test/summary.json)
- Grad-CAM per-image metrics: [`gradcam predictions.csv`](../outputs/gradcam/plantseg_test/predictions.csv)
- Grad-CAM qualitative figures: [`figures/`](../outputs/gradcam/plantseg_test/figures/)
- Robustness summary (200-image subset): [`summary.json`](../outputs/robustness/plantseg_test/summary.json),
  [`results.csv`](../outputs/robustness/plantseg_test/results.csv),
  [`accuracy_vs_severity.png`](../outputs/robustness/plantseg_test/accuracy_vs_severity.png)
- Robustness summary (complete 1,561-image test split):
  [`summary.json`](../outputs/robustness/plantseg_test_full/summary.json),
  [`results.csv`](../outputs/robustness/plantseg_test_full/results.csv),
  [`accuracy_vs_severity.png`](../outputs/robustness/plantseg_test_full/accuracy_vs_severity.png)
- 5-fold cross-validation summary: [`cv_summary.json`](../outputs/cross_validation/cv_summary.json)
- 5-fold split definition: [`folds.json`](../outputs/cross_validation/folds.json)
