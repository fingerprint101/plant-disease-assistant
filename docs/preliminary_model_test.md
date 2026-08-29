# Preliminary PlantSeg Model Test Run

**Run date:** 29 August 2026  
**Status:** Preliminary ten-epoch integration and comparison run

## Purpose

This report records the first complete comparison between the standalone disease-aware YOLO11n
model and the two-stage lesion-localization-to-classification pipeline. The run is preliminary:
every model was limited to approximately ten training epochs to verify the complete experimental
workflow before committing to longer training.

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

- Complete evaluation summary: [`summary.json`](../outputs/evaluation/plantseg_test/summary.json)
- Per-image predictions: [`predictions.csv`](../outputs/evaluation/plantseg_test/predictions.csv)
- Standalone training history: [`results.csv`](../outputs/yolo/plantseg_disease_10ep/results.csv)
- Standalone per-class metrics:
  [`standalone_yolo_per_class.json`](../outputs/evaluation/plantseg_test/standalone_yolo_per_class.json)
- Standalone confusion matrix:
  [`standalone_yolo_confusion_matrix.csv`](../outputs/evaluation/plantseg_test/standalone_yolo_confusion_matrix.csv)
