# Plant Disease Assistant

Course project on plant disease classification, localization and robustness under domain shift.

## Documentation

- [`docs/main.md`](docs/main.md): project objectives, architecture, experiments and deliverables.
- [`docs/dataset.md`](docs/dataset.md): dataset roles, measurements, limitations and preparation rules.
- [`docs/project_presentation.pptx`](docs/project_presentation.pptx): project presentation.

## Objective

PlantSeg is the primary dataset because it combines field photographs, 115 plant-disease classes,
and lesion masks. PlantVillage is the secondary controlled dataset, while PlantDoc is retained as
a backup external dataset. The project uses them as follows:

1. Train classification models on the cleaned PlantSeg split.
2. Derive YOLO boxes from PlantSeg lesion masks and train localization models.
3. Compare Grad-CAM activation maps quantitatively with PlantSeg masks.
4. Use PlantVillage as a controlled secondary benchmark on a mapped taxonomy subset.
5. Use cleaned PlantDoc only as backup external validation or supplementary detection data.

The central experiment is how well classifiers and localization methods perform on PlantSeg field
images, and how their behavior changes on the more controlled PlantVillage domain.

## Planned Models

- Small CNN trained from scratch as the classification baseline.
- EfficientNetB0 and MobileNetV2 initialized with ImageNet weights.
- YOLO nano or small for disease-labelled leaf or symptom-region detection.
- Grad-CAM for classifier activation visualization; Grad-CAM is not trained separately.

## Environment Setup

The project uses Python 3.13 and `uv`. From this directory:

```bash
make setup
source .venv/bin/activate
make check
```

The project-local cache avoids depending on a global `uv` cache. Activate the environment with `source .venv/bin/activate`, or run commands directly through `.venv/bin/python`.

## One-command Initialization

Prepare and verify the complete core project with:

```bash
make init
```

This creates the Python environment when it is missing, downloads and extracts PlantSeg,
PlantVillage, and PlantDoc when they are missing, caches the EfficientNetB0, MobileNetV2, and YOLO11n checkpoints,
and runs environment, dataset, classification, and detection smoke tests. The baseline CNN is
initialized from scratch as designed, so it has no checkpoint to download. Existing prepared
datasets and cached model checkpoints are reused without downloading them again. Results are
printed directly in the terminal.

## Jupyter

Start JupyterLab with:

```bash
make notebook
```

Select the kernel named **Python (plant-disease-assistant)**. Begin with `notebooks/00_environment_and_data.ipynb`.

## Downloading Data

Download and prepare all three datasets:

```bash
make data
```

The explicit per-dataset targets remain available when only one source is needed.

Or download them separately:

```bash
make data-plantseg
make data-plantvillage
make data-plantdoc
```

The downloader uses the PlantSeg Zenodo release and public Hugging Face dataset repositories:

- PlantSeg Zenodo release `10.5281/zenodo.17719108`
- `mohanty/PlantVillage`
- `agyaatcoder/PlantDoc`

PlantSeg is extracted under `data/raw/PlantSeg/plantseg` and is the primary source for
classification, mask-supervised localization, and explanation evaluation. PlantVillage archives
are extracted under `data/raw/PlantVillage` for controlled secondary experiments. PlantDoc
Parquet shards are downloaded to `data/downloads/PlantDoc` and converted into:

```text
data/raw/PlantDoc/
├── images/train/ and images/test/
├── labels/train/ and labels/test/   # YOLO format
├── annotations.json
├── classes.txt
└── extraction_report.json
```

Rows with invalid or out-of-bounds boxes are skipped by default and recorded in the extraction report. Extraction does not remove duplicates or create the final grouped split; those steps belong in the data-preparation notebook so that the decisions remain visible and reproducible.

Useful options:

```bash
.venv/bin/python scripts/download_datasets.py --help
.venv/bin/python scripts/extract_plantdoc.py --help
.venv/bin/python scripts/audit_plantseg.py --help
```

Run the primary PlantSeg structural audit and export one complete example with:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_plantseg.py
PYTHONPATH=src .venv/bin/python scripts/show_dataset_examples.py
```

## Recommended Workflow

1. Run `make check` and inspect the bundled sample notebook.
2. Download and verify all datasets with `make init`.
3. Preprocess PlantSeg using `Metadata.csv` and binary lesion masks as authoritative sources.
4. Train all classifiers on the cleaned PlantSeg split.
5. Derive YOLO boxes from PlantSeg masks and train the detector.
6. Compare Grad-CAM attention with PlantSeg masks.
7. Map a supported PlantVillage subset for controlled secondary evaluation.
8. Compare clean, corrupted and cross-domain performance.
9. Clean PlantDoc only if backup external validation is needed.
10. Add Grad-CAM and the upload prototype after the evaluation pipeline is stable.

## Model Smoke Test

Download the official torchvision ImageNet weights for EfficientNetB0 and MobileNetV2 and
validate all classifier architectures and YOLO11n against real project images with:

```bash
make check-models
```

The checkpoints are cached under `models/hub/checkpoints/`. The script replaces each pretrained
ImageNet output layer with a 115-class PlantSeg head and checks output shape, finite values,
softmax probabilities, and gradient flow. Results are printed directly in the terminal. The new
disease-classification heads are randomly initialized and must still be trained on PlantSeg
before their predictions are meaningful.

YOLO11n is cached at `models/yolo11n.pt` and tested with a real PlantSeg image. Its downloaded
weights are pretrained on COCO; the detector must still be trained on boxes derived from PlantSeg masks
before its disease-region detections are meaningful.

## Evaluation

Classification metrics:

- Macro F1 and balanced accuracy.
- Per-class precision, recall and F1.
- Confusion matrix and calibration error.
- Model size and inference time.

Detection metrics:

- mAP50 and mAP50-95.
- Per-class AP, precision and recall.

Robustness metrics:

- Absolute and relative macro-F1 drop.
- Per-class recall drop.
- Confidence and calibration changes by corruption type and severity.
- Performance on PlantSeg test images, controlled PlantVillage subsets, and optional PlantDoc backup data.

## Important Data Rules

- Do not merge the datasets without an explicit label-mapping table.
- Treat PlantSeg `Metadata.csv` and binary PNG masks as authoritative.
- Preserve the predefined PlantSeg split and document classes absent from validation or test.
- Deduplicate before creating final splits.
- Use PlantVillage only after explicit taxonomy mapping to PlantSeg.
- Treat PlantDoc as backup data; its boxes are mixed leaf/symptom annotations and require cleaning.
- Keep corruption type and severity as metadata rather than using one standard/non-standard label.

## Device Support

PyTorch will use CUDA when available, Apple Metal (`mps`) on compatible Macs, and otherwise the CPU. `scripts/check_environment.py` reports the selected device.
