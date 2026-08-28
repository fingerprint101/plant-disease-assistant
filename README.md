# Plant Disease Assistant

Course project on plant disease classification, localization and robustness under domain shift.

## Documentation

- [`docs/main.md`](docs/main.md): project objectives, architecture, experiments and deliverables.
- [`docs/dataset.md`](docs/dataset.md): dataset roles, measurements, limitations and preparation rules.

## Objective

PlantSeg is the primary dataset because it combines field photographs, 115 plant-disease classes,
and lesion masks. PlantVillage is the secondary controlled dataset, while PlantDoc is retained as
a backup external dataset. The project uses them as follows:

1. Derive class-agnostic YOLO boxes from PlantSeg lesion masks and train the detector.
2. Generate YOLO lesion crops and train classification models on those crops.
3. Compare classifier Grad-CAM activation maps quantitatively with PlantSeg masks.
4. Use PlantVillage as a controlled secondary benchmark on a mapped taxonomy subset.
5. Use cleaned PlantDoc only as backup external validation or supplementary detection data.

The central experiment is how well classifiers and localization methods perform on PlantSeg field
images, and how their behavior changes on the more controlled PlantVillage domain.

## Planned Models

- YOLO11n trained first as a class-agnostic lesion detector.
- Small CNN trained from scratch on YOLO lesion crops.
- EfficientNetB0 and MobileNetV3-Large initialized with ImageNet weights and trained on the same crops.
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
PlantVillage, and PlantDoc when they are missing, caches the EfficientNetB0, MobileNetV3-Large, and YOLO11n checkpoints,
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
make prepare-data
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
classification, mask-supervised localization, and explanation evaluation. `make prepare-data`
creates an idempotent training-only view under `data/training/PlantSeg` containing the official
5,367-image training split, its masks, filtered metadata, and COCO annotations. It also creates
`data/training/PlantSegDetection`, with one generic lesion box per mask for YOLO train/validation.
Images, masks, and annotations are linked to the raw extraction rather than duplicated. It prepares the test
views under `data/tests`: the complete PlantSeg test split, the mapped PlantSeg/PlantVillage
21-class subsets, and the 30 configured PlantSeg robustness conditions. Robustness corruptions are
generated on demand from the clean test view instead of storing duplicate images. PlantVillage archives
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
4. Train class-agnostic YOLO on boxes derived from PlantSeg masks.
5. Generate predicted YOLO crops and train all classifiers on those crops.
6. Compare crop-level Grad-CAM attention with PlantSeg masks.
7. Map a supported PlantVillage subset for controlled secondary evaluation.
8. Compare clean, corrupted and cross-domain performance.
9. Clean PlantDoc only if backup external validation is needed.
10. Add Grad-CAM and the upload prototype after the evaluation pipeline is stable.

## Model Smoke Test

Download the official torchvision ImageNet weights for EfficientNetB0 and MobileNetV3-Large and
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

## Classification Training

Train the complete localization-to-classification pipeline in the required order:

```bash
make train-pipeline
```

This trains the one-class YOLO lesion detector, creates predicted lesion crops for the official
PlantSeg training and validation splits, and then trains the baseline CNN, EfficientNetB0, and
MobileNetV3-Large sequentially on exactly those crops. Individual stages are also available as
`make train-yolo`, `make prepare-crops`, and `make train-classifiers`.
An existing trained YOLO checkpoint is reused; use `scripts/train_yolo.py --resume` to continue an
interrupted run or `scripts/train_yolo.py --force` to deliberately replace it.

Each model writes `last.pt`, the best macro-F1 checkpoint as `best.pt`, and `history.json` under
`outputs/classification/<model>/`. Resume an interrupted run or override training settings with:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_classifiers.py --resume
PYTHONPATH=src .venv/bin/python scripts/train_classifiers.py --models efficientnet_b0 --epochs 10
```

Run `scripts/train_classifiers.py --help` for batch-size, device, worker, output-directory, and
smoke-test batch-limit options. Classifier training uses only
`data/training/PlantSegCrops`; model selection uses crops from PlantSeg's official validation
split. No dataset under `data/tests` is read during training.

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
