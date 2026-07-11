# Plant Disease Identification and Robustness Analysis

## Project Summary

The project studies plant disease recognition from leaf photographs. A classifier is trained on PlantVillage, where leaves are centered and photographed against simple backgrounds, and is then evaluated under image corruptions and on field photographs from PlantDoc.

The main objective is to determine how much classification performance changes when test images differ from the controlled training data. The project also includes localization methods so that the prototype can show which leaf or image region contributed to a prediction.

## Proposed System

The prototype accepts a leaf image and produces:

1. A crop and disease prediction.
2. The confidence score and the most likely alternative classes.
3. A Grad-CAM activation map from the classifier.
4. Optional YOLO detections for disease-labelled leaves or visible symptomatic regions.
5. A warning when confidence is low or the image differs substantially from the training conditions.

A small web interface can be implemented with Streamlit or Gradio. The interface is part of the final demonstration, while the main technical work concerns model comparison and evaluation.

## Machine Learning Tasks

### Disease Classification

The classifier predicts one of the crop-disease classes represented in PlantVillage. Three model configurations are proposed:

- A small convolutional neural network trained from scratch, used as the baseline.
- EfficientNetB0 initialized with ImageNet weights and fine-tuned on PlantVillage.
- MobileNetV2 initialized with ImageNet weights and fine-tuned on PlantVillage.

EfficientNetB0 and MobileNetV2 allow a comparison between classification performance, model size and inference time. The baseline indicates how much transfer learning contributes to the result.

### Localization

Two methods are used for different purposes:

- **Grad-CAM** is computed from a trained classifier. It shows the image regions that contributed most strongly to the predicted class. It does not require additional region annotations.
- **YOLO** is trained on cleaned PlantDoc bounding boxes. It detects disease-labelled leaves or annotated symptomatic regions in field images.

PlantDoc boxes have mixed granularity: some cover visible symptoms, while others cover an entire diseased leaf. YOLO should therefore be described as leaf or symptom-region detection, not pixel-level lesion segmentation.

### Robustness Evaluation

Robustness is evaluated without introducing a separate network. Each trained classifier is tested on:

- The clean PlantVillage test set.
- Controlled transformations of the same images, including blur, brightness and contrast changes, JPEG compression, partial occlusion and crop changes.
- PlantDoc field images that can be mapped to the PlantVillage taxonomy.

Each transformation is stored with its type and severity. This allows performance to be plotted as corruption severity increases and provides more information than a single standard/non-standard label.

## Datasets

### PlantVillage

PlantVillage contains 54,305 color images in 38 crop-disease classes. All downloaded images are 256 by 256 pixels and decode correctly. The images are suitable for classifier training but do not reproduce field conditions.

Important preparation steps are:

- Use the official leaf-aware split.
- Remove exact images shared by the train and test lists.
- Account for the 36.23:1 class imbalance with class weighting or balanced sampling.
- Report macro F1 and per-class recall in addition to accuracy.

### PlantDoc

The downloaded PlantDoc representation contains 2,578 field images, 29 labels and 8,910 bounding boxes. It provides cluttered backgrounds, multiple leaves, occlusion and variable image dimensions.

Before training YOLO, the data requires:

- Deduplication and a new grouped train/test split.
- Removal or repair of invalid bounding boxes.
- Selection of classes with enough annotations.
- A manual mapping between PlantDoc and PlantVillage class names.

The complete measurements and identified issues are documented in `dataset_audit.md`.

## Experimental Plan

### Classification Experiments

1. Train the baseline CNN, EfficientNetB0 and MobileNetV2 on the same cleaned split.
2. Use the same input size, augmentation policy and training budget where possible.
3. Select checkpoints using validation macro F1.
4. Compare macro F1, balanced accuracy, per-class recall, calibration, model size and inference time.

### Robustness Experiments

1. Apply each corruption at several severity levels to a fixed test set.
2. Evaluate all classifiers on exactly the same transformed images.
3. Measure the absolute and relative change in macro F1, recall, confidence and calibration.
4. Evaluate the taxonomy-matched PlantDoc subset as an external field-domain test.

### Localization Experiments

1. Train a YOLO nano or small model on the cleaned PlantDoc subset.
2. Report mAP50, mAP50-95, precision, recall and per-class AP.
3. Generate Grad-CAM examples for correct predictions, incorrect predictions and field images.
4. Compare Grad-CAM regions with PlantDoc boxes qualitatively where the class mapping permits it.

## Expected Deliverables

- Dataset exploration and cleaning notebook.
- Classification training and comparison notebook.
- YOLO training configuration and evaluation results.
- Robustness test-set generator with recorded corruption metadata.
- Grad-CAM visualizations and error analysis.
- A simple image-upload prototype.
- Final report in IEEE conference format.

## Main Limitations

- PlantVillage performance can be unrealistically high because of its standardized images.
- PlantDoc is smaller, imbalanced and contains annotation errors.
- The two datasets use different class taxonomies.
- Similar symptoms can be caused by diseases, pests or nutrient deficiencies not represented in the training data.
- A closed-set classifier may assign high confidence to an unknown disease.
- Grad-CAM is an explanation of model activation, not a verified lesion boundary.
- Training YOLO on all 29 PlantDoc labels is unlikely to be useful because several classes have very few boxes.

## Scope Decision

The minimum project consists of classification, Grad-CAM and robustness analysis. YOLO is included as a bounded localization experiment using a cleaned subset of PlantDoc. Pixel-level segmentation remains an optional extension and is not required for the main result.

This scope provides several connected machine learning tasks while remaining feasible for a course project: dataset analysis, model comparison, localization, domain-shift evaluation and an end-to-end prototype.
