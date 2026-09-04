# Plant Disease Assistant

Course project on plant disease classification, localization and robustness under domain shift.

## Documentation

- [`docs/main.md`](docs/main.md): project objectives, architecture, experiments and deliverables.
- [`docs/dataset.md`](docs/dataset.md): dataset roles, measurements, limitations and preparation rules.

## Objective

PlantSeg is the primary dataset because it combines field photographs, 115 plant-disease classes,
and lesion masks. PlantVillage is the secondary controlled dataset, while PlantDoc is retained as
a backup external dataset. The project uses them as follows:

1. Derive boxes from PlantSeg lesion masks and train both class-agnostic and disease-aware YOLO detectors.
2. Generate class-agnostic YOLO lesion crops and train classification models on those crops.
3. Compare standalone disease-aware YOLO with the YOLO-to-classifier pipeline.
4. Compare classifier Grad-CAM activation maps quantitatively with PlantSeg masks.
5. Use PlantVillage as a controlled secondary benchmark on a mapped taxonomy subset.
6. Use cleaned PlantDoc only as backup external validation or supplementary detection data.

The central experiment is how well classifiers and localization methods perform on PlantSeg field
images, and how their behavior changes on the more controlled PlantVillage domain.

## Planned Models

- YOLO11n trained both as a class-agnostic crop detector and a standalone 115-class disease detector.
- Small CNN trained from scratch on YOLO lesion crops.
- EfficientNetB0 and MobileNetV3-Large initialized with ImageNet weights and trained on the same crops.
- Grad-CAM for classifier activation visualization; Grad-CAM is not trained separately.

## Environment Setup

The project uses Python 3.13 and `uv`. From this directory:

```bash
make setup
source .venv/bin/activate
```

The project-local cache avoids depending on a global `uv` cache. Activate the environment with `source .venv/bin/activate`, or run commands directly through `.venv/bin/python`.

## One-command Initialization

Prepare and verify the complete core project with:

```bash
make init
```

This creates the Python environment when it is missing, downloads and extracts PlantSeg,
PlantVillage, and PlantDoc when they are missing, caches the EfficientNetB0, MobileNetV3-Large, and YOLO11n checkpoints,
and runs dataset, classification, and detection smoke tests. The baseline CNN is
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
`data/training/PlantSegDetection`, with one generic lesion box per mask, and
`data/training/PlantSegDiseaseDetection`, where the same boxes carry PlantSeg disease IDs for the
standalone YOLO comparison.
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
```

## Recommended Workflow

1. Download, prepare and verify all datasets and model dependencies with `make init`.
2. Inspect the bundled sample notebook.
3. Preprocess PlantSeg using `Metadata.csv` and binary lesion masks as authoritative sources.
4. Train both YOLO variants on boxes derived from PlantSeg masks.
5. Generate predicted class-agnostic YOLO crops and train all classifiers on those crops.
6. Compare standalone disease-aware YOLO with the crop-classification pipeline.
7. Compare crop-level Grad-CAM attention with PlantSeg masks.
8. Map a supported PlantVillage subset for controlled secondary evaluation.
9. Compare clean, corrupted and cross-domain performance.
10. Clean PlantDoc only if backup external validation is needed.
11. Add the upload prototype after the evaluation pipeline is stable.

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

This trains the one-class YOLO lesion detector and the standalone 115-class YOLO detector, creates
predicted lesion crops for the official PlantSeg training and validation splits, and then trains
the baseline CNN, EfficientNetB0, and MobileNetV3-Large sequentially on exactly those crops.
`make train-yolo` handles both detectors. Individual stages are available as
`make train-yolo-lesion`, `make train-yolo-standalone`, `make prepare-crops`, and
`make train-classifiers`. Existing completed checkpoints are reused; pass `--mode lesion` or
`--mode disease` to `scripts/train_yolo.py` when resuming or replacing one specific run.

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

## Primary Evaluation

Compare standalone disease-aware YOLO with the complete localization-to-classification pipeline on
PlantSeg's untouched official test split:

```bash
make evaluate
```

The command evaluates both detectors against test boxes derived from the authoritative lesion
masks. For standalone YOLO, the highest-confidence detected box supplies the image's disease
prediction; an image with no detection is counted as incorrect. It then streams every test image
through the class-agnostic detector and evaluates all three classifiers on the resulting crops.
Results are written under `outputs/evaluation/plantseg_test/`, including `summary.json`, per-image
predictions, per-class metrics, confusion matrices, and separate YOLO plots. Pipeline images
without a confident lesion detection use the full image and are reported through the fallback rate.

A small integration run can be launched without consuming the complete test evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_pipeline.py \
  --max-images 8 --device mps \
  --output-dir /tmp/plantseg-evaluation-smoke
```

## Grad-CAM vs PlantSeg Masks

Compare classifier attention with the authoritative lesion masks on the official PlantSeg test
split:

```bash
make gradcam
```

For each test image, the class-agnostic lesion YOLO detector supplies the same crop used by the
classification pipeline (falling back to the full image when it finds nothing). Grad-CAM is
computed on that crop for the model's predicted class and pasted back into full-image coordinates
so it can be compared pixel-for-pixel with the ground-truth mask. Results are written under
`outputs/gradcam/plantseg_test/`: `summary.json` with mean/median IoU, precision, recall, mask
energy fraction and pointing-game hit rate; `predictions.csv` with per-image metrics; and a
`figures/` folder with side-by-side image/mask/Grad-CAM qualitative examples.

By default `make gradcam` explains `mobilenet_v3_large` (best validation macro-F1 in the
preliminary run). Explain a different model or limit the run with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_gradcam.py --model efficientnet_b0 --max-images 20
```

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

PyTorch will use CUDA when available, Apple Metal (`mps`) on compatible Macs, and otherwise the CPU.
