# Presentation Content — Plant Disease Assistant

Structured to mirror the sample slide deck format (title → problem → tasks → dataset → models →
model selection → cross-validation → per-task results tables → confusion matrix / curves →
end-to-end results → future work → thank you). Hand this whole file to Claude to generate slides.

---

## Slide 1 — Title

**Plant Disease Detection and Classification**

Intelligent Systems Course Project

[Your names]

A.Y. 2025 / 2026

---

## Slide 2 — The Problem

**Our objective is to obtain a computer-friendly diagnosis of a plant disease from a photograph of
a diseased leaf**

This could be useful for different applications, such as:
- To give farmers or gardeners an instant diagnosis from a phone photo
- To flag disease outbreaks early, before they spread across a field
- As a component of a larger crop-monitoring or advisory assistant

(Input) Leaf photograph → (Output) Disease name + confidence + highlighted lesion region

---

## Slide 3 — Defining the Tasks

Two complementary approaches, both built on the same dataset:

**(1) Standalone approach** — one YOLO model localizes the lesion and directly assigns one of 115
disease classes to its highest-confidence detection, in a single pass.

**(2) Two-stage pipeline** — a class-agnostic YOLO model first finds the lesion (draws a box around
it, no disease name), the lesion is cropped out, and a separate image classifier (one of three
architectures) names the disease from the crop.

Diagram: (Input) Leaf Image → [Path A: Standalone disease-aware YOLO → disease name] and
[Path B: Class-agnostic lesion YOLO → crop → Classifier → disease name]

---

## Slide 4 — The Dataset

**PlantSeg**: 7,774 field photographs of diseased plant leaves, each with:
1. A pixel-level binary lesion mask (used to derive bounding boxes)
2. A disease label (115 plant-disease classes across 34 plant species)
3. An official train / validation / test split

| Partition | Images | Classes present |
|---|---:|---:|
| Training | 5,367 | 115 |
| Validation | 846 | 114 |
| Test | 1,561 | 114 |

- Median 55 images per class; smallest class has only 3 images, largest has 323
- Images are real field photographs (not synthetic/studio), so lighting, angle, and background
  vary a lot
- Masks and metadata are treated as the authoritative source of truth; boxes are derived from the
  masks rather than trusting any pre-supplied annotation

---

## Slide 5 — Computer Vision Models

**Lesion localization**

YOLO11n (nano), used two ways:
- Class-agnostic: single "lesion" class, just finds where the disease is
- Disease-aware (standalone): 115 classes, finds and names the disease in one shot

**Disease classification** (on the cropped lesion region)

- A small CNN trained from scratch (baseline, no pretraining)
- EfficientNetB0, initialized with ImageNet weights
- MobileNetV3-Large, initialized with ImageNet weights

**Explanation**

Grad-CAM, applied post-hoc to the trained classifier (not trained separately) to visualize which
pixels drove its prediction, compared quantitatively against the ground-truth lesion masks.

---

## Slide 6 — Training Setup

| Setting | Value |
|---|---:|
| YOLO epochs | 50–100 (varies by experiment) |
| Input size | 640 × 640 (detection), 224 × 224 (classification) |
| Batch size | 16 |
| Classes | 115 |
| Optimizer | Ultralytics automatic selection (YOLO) / AdamW (classifiers) |
| Device | Apple M1 Pro (MPS) |
| Seed | 42 |

Model selection used validation macro F1 (not raw accuracy), because class sizes are unbalanced
and macro F1 weights every disease class equally regardless of how many examples it has.

---

## Slide 7 — 5-Fold Cross-Validation

**Dataset Partition:**

The official 5,367-image PlantSeg training split, divided into 5 stratified folds by disease
class. Two extremely rare classes (2 and 4 total training images) are too small to stratify
across 5 folds and are instead distributed round-robin.

For each fold: train on the other 4 folds, validate on the held-out one. All 5 models (both YOLO
variants + all 3 classifiers) are trained from scratch per fold, for 50 epochs each — 25
independent training runs in total.

The official validation and test splits are never touched by this process; cross-validation
measures training stability, not final test performance.

---

## Slide 8 — Cross-Validation Results

| Model | Metric | Mean | Std |
|---|---|---:|---:|
| Class-agnostic lesion YOLO | mAP@50 | 85.3% | ±0.56pp |
| Class-agnostic lesion YOLO | mAP@50–95 | 57.0% | ±1.00pp |
| Disease-aware standalone YOLO | mAP@50 | 40.9% | ±0.66pp |
| Disease-aware standalone YOLO | mAP@50–95 | 29.0% | ±0.78pp |
| Baseline CNN (from scratch) | validation macro F1 | 20.1% | ±0.75pp |
| EfficientNetB0 | validation macro F1 | **59.4%** | ±1.99pp |
| MobileNetV3-Large | validation macro F1 | 57.3% | ±1.44pp |

The small standard deviations (all under 2 percentage points) show these results are stable across
different train/validation partitions, not an artifact of one lucky split. EfficientNetB0 is the
most accurate classifier in every fold.

---

## Slide 9 — Lesion Detection Results (final models)

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Class-agnostic lesion YOLO | 85.27% | 81.94% | **88.22%** | 61.52% |
| Disease-aware standalone YOLO | 54.93% | 49.90% | 50.15% | 34.47% |

These numbers are from `evaluate_pipeline.py` run once against the final checkpoints on the
official held-out test split (1,561 images) — the authoritative final result. A separate
training-time figure appears in `docs/preliminary_model_test.md` (86.08% / 50.02%), measured on
the internal validation split during training itself rather than the held-out test set; the two
are close but not identical because they measure different data. Use the test-split numbers above
for the presentation.

The class-agnostic detector only needs to find a lesion; the standalone detector must find *and*
correctly name it in the same step, which is a much harder joint task — explaining the large gap.

[Optional image: outputs/yolo/plantseg_lesion_10ep/val_batch0_pred.jpg — shows real validation
images with predicted boxes]

---

## Slide 10 — End-to-End Classification Results

| System | Accuracy | Macro F1 | Balanced accuracy | Coverage |
|---|---:|---:|---:|---:|
| Standalone disease-aware YOLO | 61.37% | 53.32% | 49.60% | 86.48% |
| Lesion YOLO + baseline CNN | 41.32% | 29.04% | 29.99% | 100%¹ |
| Lesion YOLO + EfficientNetB0 | 66.69% | 59.02% | 58.69% | 100%¹ |
| **Lesion YOLO + MobileNetV3-Large** | **66.75%** | 58.17% | 58.69% | 100%¹ |

¹ The two-stage pipeline always returns an answer: it falls back to classifying the full image
when the lesion detector finds nothing. This happened for only 38 of 1,561 test images (2.4%).

EfficientNetB0 and MobileNetV3-Large achieve essentially the same accuracy; the two-stage pipeline
outperforms the single-shot standalone YOLO by about 5 percentage points, at the cost of running
two models instead of one.

---

## Slide 11 — Grad-CAM vs Ground-Truth Lesion Masks

Comparing what the classifier "looks at" (Grad-CAM activation) against the actual lesion mask, on
all 1,561 test images:

| Metric | Value |
|---|---:|
| IoU (top-20% activation quantile) | 24.2% |
| Mask energy fraction | 41.5% |
| Pointing-game hit rate | 58.8% |

The ground-truth lesion masks cover only about 21% of each image on average, so a mask-energy
fraction of 41.5% means the model's attention is genuinely concentrated on the diseased region,
not spread randomly across the leaf. The pointing-game hit rate (does the single most-activated
pixel fall inside the lesion?) succeeds for the majority of images.

[Include 2–4 qualitative figures from outputs/gradcam/plantseg_test/figures/ — side-by-side
photo / ground-truth mask / Grad-CAM heatmap]

---

## Slide 12 — Robustness Under Synthetic Corruption

Testing how much accuracy degrades when images are corrupted (blur, brightness/contrast shift,
JPEG compression, occlusion, crop/reframing) at 5 increasing severity levels, on the full
1,561-image test split:

| Corruption (severity 5) | Baseline CNN | EfficientNetB0 | MobileNetV3-Large |
|---|---:|---:|---:|
| Clean (no corruption) | 40.0% | 67.3% | 66.5% |
| **Gaussian blur (worst case)** | 12.5% | 29.7% | 30.5% |
| Brightness | 17.8% | 49.6% | 46.8% |
| Crop | 35.5% | 63.6% | 63.8% |

Gaussian blur is by far the worst-case corruption for every model — accuracy roughly halves.
Crop and occlusion are comparatively well tolerated, consistent with the random-crop/flip
augmentation already used during training.

[Optional chart: outputs/robustness/plantseg_test_full/accuracy_vs_severity.png]

---

## Slide 13 — Future Work

**Ways to explore for possible improvements:**

- Address the standalone YOLO's joint localization + 115-way classification difficulty — possibly
  via curriculum training or a two-stage loss
- Improve robustness to blur specifically, since it is the dominant failure mode for every model
- Extend evaluation to a second, independently-collected dataset (cross-domain generalization) to
  test whether performance holds outside PlantSeg's specific photography conditions
- Explore class rebalancing or targeted augmentation for the rarest disease classes (some have
  only 2–3 training images)

---

## Slide 14 — Thank You

Thank you for listening.

---

## Notes for whoever builds the slides

- All numbers above are pulled directly from this project's own evaluation output files
  (`outputs/evaluation/plantseg_test/summary.json`, `outputs/cross_validation/cv_summary.json`,
  `outputs/gradcam/plantseg_test/summary.json`, `outputs/robustness/plantseg_test_full/summary.json`)
  — verified current as of this document's creation, not estimated.
- Real images to embed if desired:
  - `outputs/gradcam/plantseg_test/figures/*.png` — photo/mask/heatmap triptychs (12 available)
  - `outputs/yolo/plantseg_lesion_10ep/val_batch0_pred.jpg` — real leaf photos with predicted boxes
  - `outputs/robustness/plantseg_test_full/accuracy_vs_severity.png` — corruption robustness chart
- Full narrative writeup with additional detail and interpretation is in
  `docs/preliminary_model_test.md` if more context is needed for any slide.
