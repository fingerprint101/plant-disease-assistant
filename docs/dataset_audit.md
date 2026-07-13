# Plant Disease Dataset Audit

Audit date: 2026-07-13

## Verdict

The project has enough good data to proceed. The strongest feasible design is:

1. Train the disease classifier on PlantVillage.
2. Use PlantSeg as the main field dataset for classification and supervised localization.
3. Derive detection boxes from PlantSeg masks rather than trusting every supplied COCO box.
4. Compare Grad-CAM attention with PlantSeg masks quantitatively.
5. Add cleaned PlantDoc data as external validation or supplementary detection data if time permits.

PlantVillage is a good controlled classification baseline. PlantSeg supports the main field-image,
detection, segmentation and explanation experiments, but requires a preprocessing pass. PlantDoc
remains useful for external validation, but its supplied split and annotations also need cleaning.

## Downloaded Data

### PlantVillage

- Source: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
- Local archive: `datasets/PlantVillage-data.zip`
- Extracted color images: `datasets/PlantVillage/raw/color/`
- SHA-256: `fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84`
- License stated by the dataset repository: CC BY-SA 3.0.

### PlantDoc Object Detection

- Original source: [PlantDoc Object Detection repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
- Downloaded representation: [Hugging Face PlantDoc mirror](https://huggingface.co/datasets/agyaatcoder/PlantDoc), which states that it was produced from the original repository.
- Local shards: `datasets/PlantDoc-parquet/`
- Combined downloaded size: approximately 977 MB.
- Test shard SHA-256: `9462c686c9596435a7917ebc0f8a238d9eed921307ab1113d978882d68eda406`.
- Train shard 0 SHA-256: `7df66cea85687fb66027fda64b478de80650fd42f4b17b7106d75a419f07e2d2`.
- Train shard 1 SHA-256: `b94d5480c7c615810fef367dcb84915c1ef9031f8e07ae56925e37373f2b78a6`.

### PlantSeg

- Source: [latest published Zenodo release](https://zenodo.org/records/17719108)
- DOI: `10.5281/zenodo.17719108`.
- Local archive: `data/downloads/plantseg/plantseg.zip`.
- Extracted data: `data/raw/PlantSeg/plantseg/`.
- Archive size: 1,057,281,724 bytes.
- MD5: `9358a66dff88cdd15c4fe009763c40a3` (verified).
- Release license: CC BY-NC 4.0; per-image metadata lists 7,741 CC-BY-NC and 33 CC0 images.

## PlantVillage Findings

| Check | Result |
|---|---:|
| Images | 54,305 |
| Classes | 38 |
| Resolution | 54,305 at 256 x 256 |
| Corrupt images | 0 |
| Smallest class | 152 images |
| Largest class | 5,507 images |
| Max/min imbalance | 36.23:1 |
| Exact duplicate extras | 21 |
| Filenames containing `copy` | 1,367 |
| Official train/test sizes | 43,596 / 10,709 |
| Exact images shared by official train/test | 5 |
| Known leaf IDs shared by official train/test | 0 |

The images are technically clean and immediately trainable. The entire color set decoded successfully. The visual samples confirm that it is a controlled dataset: leaves are centered, backgrounds are simple, framing is consistent, and all images are square. This is useful for learning disease appearance but inadequate as evidence of field deployment.

The class distribution is strongly imbalanced. `Potato___healthy` has only 152 images while the largest class has 5,507. Report macro F1 and per-class recall, and use class-weighted loss or a balanced sampler. Accuracy alone will hide failures on small classes.

The official split is preferable to a new random split because its known leaf groups do not overlap. However, 13,194 images could not be associated with a leaf ID by the repository's own lookup logic, and five exact images occur in both sides. Remove cross-split duplicates before final evaluation and document the resulting counts.

## PlantDoc Findings

| Check | Result |
|---|---:|
| Images | 2,578 |
| Supplied train/test sizes | 2,342 / 236 |
| Classes | 29 |
| Bounding boxes | 8,910 |
| Mean boxes per image | 3.46 |
| Median box area / image area | 6.71% |
| Unique image dimensions | 1,506 |
| Image decode failures | 0 |
| Exact duplicate extras | 12 |
| Exact images shared by train/test | 11 |
| Invalid/out-of-bounds boxes | 11 |
| Image-size metadata mismatches | 12 |
| Classes absent from test | 2 |
| Annotation max/min imbalance | 424.5:1 |

PlantDoc provides the variability missing from PlantVillage: cluttered backgrounds, multiple leaves, occlusion, different cameras, and a wide range of resolutions. It is therefore useful for the YOLO and robustness branches.

It should not be trained as downloaded. Eleven exact images leak across the supplied train/test split. Two classes are absent from the test set, and the rarest class has only two boxes. Build a new grouped split after deduplication, and either remove extremely rare classes or merge only when the taxonomy genuinely permits it.

Eleven boxes extend outside the decoded image. Most are associated with twelve records whose stored width/height metadata does not match the embedded image, suggesting stale annotations after resizing. Reject these records or rescale their boxes from the original coordinate system; do not silently clip them without first checking the corresponding image.

The boxes also have mixed meaning. Visual inspection shows that some boxes tightly surround symptomatic areas, while many surround whole disease-labelled leaves. Accordingly, YOLO can honestly be presented as detecting disease-labelled leaves or visible symptomatic regions. It is not a lesion-segmentation model.

PlantDoc and PlantVillage do not share an identical label taxonomy. Create an explicit mapping table for the overlapping crop-disease pairs and report results only on that shared subset. Do not treat differently named classes as equivalent without manual review.

## PlantSeg Findings

| Check | Result |
|---|---:|
| Image/mask pairs | 7,774 |
| Plants | 34 |
| Plant-disease classes | 115 |
| Train / validation / test | 5,367 / 846 / 1,561 |
| Missing or corrupt files | 0 |
| Median images per class | 55 |
| Smallest / largest class | 3 / 323 |
| EXIF orientation cases | 8 |
| Masks with inconsistent encoded class value | 12 |
| COCO images with mask/box disagreement | 220 |
| COCO images with out-of-bounds boxes | 51 |
| COCO images without annotations | 1 |

PlantSeg is suitable for the project, but the PNG masks and `Metadata.csv` should be treated as
the authoritative sources. Convert images with EXIF orientation applied, interpret each mask as a
binary lesion mask, obtain the class from the metadata, and derive fresh boxes from connected mask
regions. This avoids the empty COCO `categories` arrays, twelve inconsistent encoded class values,
and the supplied box disagreements.

The predefined training split contains all 115 classes. Validation is missing class ID 68 and test
is missing class ID 41, while several classes contain very few images overall. Full 115-class
results therefore require macro metrics and an explicit note that not every class is represented in
every evaluation split. A reduced well-supported taxonomy is a reasonable course-project option.

## Recommended Experiments

### Classification

- Use PlantVillage color images with the official leaf-aware split after exact deduplication.
- Compare a simple CNN with EfficientNetB0 and optionally MobileNetV2.
- Use macro F1, balanced accuracy, per-class recall, calibration error, and confusion matrices.
- Fit preprocessing and class weights on training data only.

### YOLO

- Deduplicate PlantDoc before splitting.
- Remove or repair invalid annotations.
- Start with disease classes that have enough boxes; do not attempt all 29 classes merely because they exist.
- Use YOLO nano or small for a course-scale experiment.
- Report mAP50, mAP50-95, per-class AP, precision, recall, and representative failures.

### Robustness

- Controlled shift: apply blur, brightness, contrast, JPEG compression, partial occlusion, and crop/framing transformations to a fixed clean PlantVillage test set at several severity levels.
- Real shift: evaluate only taxonomy-matched PlantDoc crops or images.
- Report absolute and relative drops in macro F1, recall, confidence, and calibration.
- Keep corruption type and severity as metadata. A single binary `standard/non-standard` label would discard useful information.

## Assets and Reproduction

- Raw measurements: `datasets/audit/audit_results.json`
- PlantVillage contact sheet: `datasets/audit/plantvillage_samples.jpg`
- PlantDoc box contact sheet: `datasets/audit/plantdoc_box_samples.jpg`
- PlantSeg machine-readable audit: `outputs/plantseg_audit.json`
- PlantSeg example and overlay: `outputs/dataset_examples/plantseg_example_annotated.jpg`
- PlantSeg audit script: `scripts/audit_plantseg.py`
- Audit script: `../Working Files/audit_plant_datasets.py`

Run the audit from the vault root with a Python environment containing Pillow, NumPy, and PyArrow.
