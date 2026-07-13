# Plant Disease Assistant

Course project on plant disease classification, localization and robustness under domain shift.

## Objective

PlantVillage provides many labelled images, but its leaves are centered and photographed against simple backgrounds. PlantDoc is smaller and noisier, but contains realistic field photographs and bounding boxes. The project uses both datasets for different stages:

1. Train classification models on PlantVillage.
2. Adapt selected classifiers with taxonomy-matched PlantDoc crops.
3. Train YOLO on cleaned PlantDoc bounding boxes.
4. Generate Grad-CAM activation maps from the classifiers.
5. Compare clean performance with synthetic corruptions and an untouched PlantDoc field test set.

The central experiment is whether PlantDoc adaptation improves field performance without causing an unacceptable loss on controlled PlantVillage images.

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

Equivalent commands without `make`:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.13
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name plant-disease-assistant \
  --display-name "Python (plant-disease-assistant)"
```

The project-local cache avoids depending on a global `uv` cache. Activate the environment with `source .venv/bin/activate`, or run commands directly through `.venv/bin/python`.

## Jupyter

Start JupyterLab with:

```bash
make notebook
```

Select the kernel named **Python (plant-disease-assistant)**. Begin with `notebooks/00_environment_and_data.ipynb`.

## Downloading Data

Download and prepare both datasets:

```bash
make data
```

Or download them separately:

```bash
make data-plantvillage
make data-plantdoc
make data-plantseg
```

The downloader uses the public Hugging Face dataset repositories:

- `mohanty/PlantVillage`
- `agyaatcoder/PlantDoc`
- PlantSeg Zenodo release `10.5281/zenodo.17719108`

PlantVillage archives are extracted under `data/raw/PlantVillage`. PlantDoc Parquet shards are downloaded to `data/downloads/PlantDoc` and converted into:

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

PlantSeg is extracted under `data/raw/PlantSeg/plantseg`. Run its structural audit and
export one complete example with:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_plantseg.py
PYTHONPATH=src .venv/bin/python scripts/show_dataset_examples.py
```

## Recommended Workflow

1. Run `make check` and inspect the bundled sample notebook.
2. Download the full datasets with `make data`.
3. Audit duplicates, class counts and PlantDoc annotations.
4. Define a shared PlantVillage/PlantDoc taxonomy before creating crops.
5. Reserve a deduplicated PlantDoc field test set before fine-tuning.
6. Train the PlantVillage-only baseline models.
7. Fine-tune selected models on the PlantDoc training crops.
8. Compare clean, corrupted and real-field performance.
9. Train YOLO on the cleaned PlantDoc detection split.
10. Add Grad-CAM and the upload prototype after the evaluation pipeline is stable.

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
- Performance on the untouched taxonomy-matched PlantDoc field test set.

## Important Data Rules

- Do not merge the datasets without an explicit label-mapping table.
- Do not adapt on the PlantDoc images reserved for external testing.
- Deduplicate before creating final splits.
- Treat PlantDoc boxes as disease-labelled leaves or visible symptom regions, not guaranteed lesion masks.
- Keep corruption type and severity as metadata rather than using one standard/non-standard label.

## Device Support

PyTorch will use CUDA when available, Apple Metal (`mps`) on compatible Macs, and otherwise the CPU. `scripts/check_environment.py` reports the selected device.
