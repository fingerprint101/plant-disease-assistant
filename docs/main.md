# Plant Disease Assistant

This document is the central technical overview of the project. Dataset measurements, integrity
findings and preparation rules are documented separately in [`dataset.md`](dataset.md).

## Project Overview

The project studies plant disease recognition and localization in field photographs. PlantSeg is
the primary dataset because it provides 7,774 image-mask pairs across 115 plant-disease classes.
PlantVillage is a secondary controlled benchmark, and PlantDoc is retained as backup external or
supplementary data if its annotations can be cleaned and mapped safely.

The main objective is to compare a two-stage localization-to-classification pipeline with a
standalone disease-aware YOLO detector, measure robustness under synthetic and cross-domain shifts,
and determine whether classifier attention overlaps known lesion regions.

## Proposed System

The prototype accepts a leaf image and processes it in this order:

1. A class-agnostic YOLO detector localizes visible lesion regions.
2. The detected region is cropped and passed to a plant-disease classifier.
3. The classifier returns a prediction, confidence score and likely alternatives.
4. Grad-CAM explains the classifier decision within the localized crop.
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

The primary classifiers predict PlantSeg plant-disease classes from YOLO-generated lesion crops.
Three configurations are compared:

- A small convolutional neural network trained from scratch.
- EfficientNetB0 initialized with ImageNet weights and fine-tuned on PlantSeg.
- MobileNetV3-Large initialized with ImageNet weights and fine-tuned on PlantSeg.

YOLO is trained first. Its predictions are then used to create classifier crops for both the
training and validation splits. All classifiers use exactly the same crops, preprocessing policy
and evaluation protocol. Checkpoints are selected using validation macro F1.

### Localization and Explanations

- **Grad-CAM** explains classifier predictions. Its activation maps are compared quantitatively
  with PlantSeg lesion masks, in addition to representative qualitative examples.
- **YOLO11n** is trained as a one-class lesion detector on bounding boxes derived from PlantSeg
  masks. Disease identity is deliberately left to the second-stage classifiers. This provides a
  consistent lesion target without relying on the problematic supplied COCO boxes.
- A second **YOLO11n** uses the same boxes labelled with PlantSeg's 115 disease IDs. It performs
  localization and disease classification in one model and is the standalone comparison baseline.

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
2. Train class-agnostic YOLO and generate predicted train/validation lesion crops.
3. Train disease-aware YOLO on the same mask-derived boxes as the standalone model.
4. Train the baseline CNN, EfficientNetB0 and MobileNetV3-Large on the same crops.
5. Use class weighting or balanced sampling to address rare classes.
6. Compare macro F1, balanced accuracy, per-class recall, calibration, model size and inference time.
7. Evaluate the complete YOLO-to-classifier pipeline on a mapped PlantVillage subset.

### Localization Experiments

1. Derive one enclosing lesion box from each PlantSeg binary mask.
2. Train YOLO11n as a class-agnostic lesion detector on the derived boxes.
3. Train a second YOLO11n with each box labelled by its image-level PlantSeg disease class.
4. Report mAP50, mAP50-95, precision, recall and per-class AP for both detectors.
5. Measure Grad-CAM overlap with masks using localization metrics and representative examples.

The primary holdout evaluation is implemented by `scripts/evaluate_pipeline.py`. It compares the
standalone disease-aware YOLO detector with end-to-end classification after class-agnostic YOLO
crops, using only the official PlantSeg test split. It reports detection metrics for both YOLO
models and image-level accuracy, macro F1, coverage and runtime for the standalone model. Run it
once the training choices are fixed with `make evaluate`.

Results and analysis from the first complete ten-epoch run are recorded in
[`preliminary_model_test.md`](preliminary_model_test.md).

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

The minimum project uses a YOLO-to-classifier cascade trained on PlantSeg, Grad-CAM evaluation on
the resulting crops, and end-to-end robustness analysis. PlantVillage supplies a bounded
controlled-domain comparison. PlantDoc is a backup dataset rather than a required dependency.
Full-image classification and pixel-level segmentation training are outside the core experiment.
