# Plant Disease Assistant

This document is the central technical overview of the project. Dataset measurements, integrity
findings and preparation rules are documented separately in [`dataset.md`](dataset.md).

## Project Overview

The project studies plant disease recognition and localization in field photographs. PlantSeg is
the primary dataset because it provides 7,774 image-mask pairs across 115 plant-disease classes.
PlantVillage is a secondary controlled benchmark, and PlantDoc is retained as backup external or
supplementary data if its annotations can be cleaned and mapped safely.

The main objective is to compare classification architectures, measure robustness under synthetic
and cross-domain shifts, and evaluate whether model attention overlaps known lesion regions.

## Proposed System

The prototype accepts a leaf image and produces:

1. A plant-disease prediction.
2. The confidence score and most likely alternative classes.
3. A Grad-CAM activation map from the classifier.
4. Optional YOLO detections for lesion or symptom regions.
5. A warning when confidence is low or the image differs substantially from training conditions.

A small Streamlit or Gradio interface can support the final demonstration. The main technical work
remains model comparison, localization, robustness evaluation, and explanation analysis.

## Dataset Roles

### Primary: PlantSeg

PlantSeg contains 7,774 field image-mask pairs, 34 plants and 115 plant-disease classes. Its
predefined train, validation and test splits are the main experimental partitions. `Metadata.csv`
provides class labels, and the binary PNG masks are the authoritative localization annotations.

Preparation must apply EXIF orientation, interpret masks as binary lesion masks, and derive fresh
bounding boxes from connected mask regions. The supplied COCO annotations are not authoritative
because some boxes disagree with masks or extend outside image bounds.

### Secondary: PlantVillage

PlantVillage contains 54,305 controlled color images across 38 classes. Its centered leaves and
simple backgrounds make it useful for secondary controlled experiments, but not as the main
evidence for field performance. Comparisons with PlantSeg must use an explicit taxonomy mapping.

### Backup: PlantDoc

PlantDoc contains 2,578 field images, 29 labels and 8,910 supplied boxes. It is smaller, severely
imbalanced and affected by duplicate leakage, invalid boxes and mixed annotation granularity. It is
therefore reserved for backup external validation or supplementary detection data if time permits.

## Machine Learning Tasks

### Disease Classification

The primary classifier predicts PlantSeg plant-disease classes. Three configurations are compared:

- A small convolutional neural network trained from scratch.
- EfficientNetB0 initialized with ImageNet weights and fine-tuned on PlantSeg.
- MobileNetV2 initialized with ImageNet weights and fine-tuned on PlantSeg.

All models use the same cleaned split, preprocessing policy and evaluation protocol. Checkpoints
are selected using validation macro F1.

### Localization and Explanations

- **Grad-CAM** explains classifier predictions. Its activation maps are compared quantitatively
  with PlantSeg lesion masks, in addition to representative qualitative examples.
- **YOLO11n** is trained on bounding boxes derived from PlantSeg masks. This provides a consistent
  lesion-region target without relying on the problematic supplied COCO boxes.

PlantDoc detection boxes may be considered only as supplementary backup data after cleaning.

### Robustness Evaluation

Each trained classifier is evaluated on:

- The clean PlantSeg test split.
- Controlled corruptions of a fixed PlantSeg test subset, including blur, brightness, contrast,
  JPEG compression, partial occlusion and crop changes.
- A taxonomy-matched PlantVillage subset as a controlled cross-domain benchmark.
- A cleaned taxonomy-matched PlantDoc subset only as optional backup external validation.

Each corruption is stored with its type and severity so performance degradation can be measured as
severity increases.

## Experimental Plan

### Classification Experiments

1. Preprocess PlantSeg using metadata and masks as authoritative sources.
2. Train the baseline CNN, EfficientNetB0 and MobileNetV2 on the same split.
3. Use class weighting or balanced sampling to address rare classes.
4. Compare macro F1, balanced accuracy, per-class recall, calibration, model size and inference time.
5. Evaluate a manually mapped PlantVillage subset as the secondary domain.

### Localization Experiments

1. Derive connected-component boxes from PlantSeg binary masks.
2. Train YOLO11n on the derived boxes.
3. Report mAP50, mAP50-95, precision, recall and per-class AP.
4. Measure Grad-CAM overlap with masks using localization metrics and representative examples.

### Robustness Experiments

1. Apply each corruption at several severity levels to a fixed PlantSeg test subset.
2. Evaluate all classifiers on exactly the same transformed images.
3. Measure absolute and relative changes in macro F1, recall, confidence and calibration.
4. Compare performance on the mapped PlantVillage subset.
5. Add PlantDoc only if a reliable cleaned backup subset can be produced.

## Expected Deliverables

- PlantSeg exploration and preprocessing notebook.
- Classification training and comparison notebook.
- Mask-to-box conversion and YOLO evaluation.
- Robustness test-set generator with corruption metadata.
- Quantitative and qualitative Grad-CAM analysis against PlantSeg masks.
- Controlled PlantVillage cross-domain evaluation.
- Optional cleaned PlantDoc backup evaluation.
- A simple image-upload prototype and final IEEE-format report.

## Main Limitations

- PlantSeg has rare classes, and validation/test do not contain every class.
- Twelve masks use inconsistent encoded values, so masks must be treated as binary.
- The supplied PlantSeg COCO boxes contain disagreements and out-of-bounds cases.
- PlantVillage uses a different taxonomy and highly controlled imagery.
- PlantDoc is small, imbalanced and affected by annotation and split issues.
- A closed-set classifier may assign high confidence to unknown diseases.
- Grad-CAM attention is an explanation, not a verified lesion boundary.

## Scope Decision

The minimum project uses PlantSeg for classification, Grad-CAM evaluation, YOLO localization and
robustness analysis. PlantVillage supplies a bounded controlled-domain comparison. PlantDoc is a
backup dataset rather than a required experimental dependency. Pixel-level segmentation training
remains an optional extension because the main localization tasks already use masks for supervision
and explanation evaluation.
