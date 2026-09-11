# Plant Disease Assistant

Plant Disease Assistant is a computer vision project for recognizing plant diseases in field
images. It compares two approaches trained on PlantSeg:

- A class-agnostic YOLO lesion detector followed by a dedicated image classifier.
- A standalone disease-aware YOLO model that detects and classifies lesions directly.

The classifiers are a small CNN, EfficientNetB0, and MobileNetV3-Large. The experiments compare
classification quality, localization, cross-domain performance on equivalent PlantVillage
classes, robustness to image corruption, model latency, and Grad-CAM explanations.

The main results are collected in
[`notebooks/01_experiments_and_results.ipynb`](notebooks/01_experiments_and_results.ipynb).

## Setup

The project requires Python 3.13, `uv`, and `make`. Create the environment with:

```bash
make setup
source .venv/bin/activate
```

To set up the environment, download and prepare the datasets, and run the initial checks in one
step:

```bash
make init
```

Commands automatically use CUDA when available, Apple Metal on compatible Macs, and otherwise
the CPU. Most Python scripts also accept `--device cuda`, `--device mps`, or `--device cpu`.

## Main commands

Run these commands from the repository root.

| Command | What it does |
| --- | --- |
| `make data` | Downloads PlantSeg, PlantVillage, and PlantDoc. |
| `make prepare-data` | Creates the prepared training and test views, label mappings, and YOLO annotations. |
| `make check-datasets` | Checks dataset paths, annotations, masks, and class mappings. |
| `make check-models` | Downloads required pretrained weights and runs model smoke tests. |
| `make train-pipeline` | Trains both YOLO variants, creates lesion crops, and trains all three classifiers. |
| `make evaluate` | Evaluates the classifier pipelines and standalone YOLO on the PlantSeg test split. |
| `make evaluate-cross-domain` | Compares the models on equivalent PlantSeg and PlantVillage classes. |
| `make robustness` | Measures accuracy under blur, brightness, contrast, JPEG compression, occlusion, and crop changes. |
| `make benchmark-efficiency` | Measures latency, throughput, parameter count, and checkpoint size. |
| `make gradcam` | Compares classifier Grad-CAM maps with PlantSeg lesion masks. |
| `make notebook` | Starts JupyterLab with the project environment. |

Use `make data-plantseg`, `make data-plantvillage`, or `make data-plantdoc` to download only one
dataset. `make kernel` registers the Jupyter kernel again, and `make clean-cache` removes the local
`uv` download cache.

The training pipeline can also be run one stage at a time:

| Command | What it does |
| --- | --- |
| `make train-yolo-lesion` | Trains the class-agnostic lesion detector used to create classifier crops. |
| `make train-yolo-standalone` | Trains the disease-aware standalone YOLO model. |
| `make train-yolo` | Runs both YOLO training commands. |
| `make prepare-crops` | Uses the lesion detector to create classifier training and validation crops. |
| `make train-classifiers` | Trains the CNN, EfficientNetB0, and MobileNetV3-Large on those crops. |

Run five-fold cross-validation with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cross_validation.py --device mps
```

Replace `mps` with `cuda` or `cpu` as needed. Cross-validation trains each model on five
stratified partitions of the PlantSeg training pool and saves the fold metrics for the results
notebook.

## Useful command options

Every script provides a complete option list through `--help`. For example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_robustness.py --help
PYTHONPATH=src .venv/bin/python scripts/evaluate_pipeline.py --help
PYTHONPATH=src .venv/bin/python scripts/train_classifiers.py --help
```

Small evaluation runs can be useful for checking the pipeline before a full experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_pipeline.py \
  --max-images 8 --device mps --output-dir /tmp/plantseg-evaluation-smoke
```

To train only one classifier or resume interrupted classifier training:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_classifiers.py \
  --models mobilenet_v3_large --resume
```

## Project files

- `configs/project.yaml`: dataset paths, training settings, corruption levels, and random seed.
- `scripts/`: data preparation, training, evaluation, robustness, and analysis commands.
- `src/`: shared datasets, models, metrics, and pipeline code.
- `notebooks/01_experiments_and_results.ipynb`: consolidated experiment results and plots.
- `outputs/`: trained models, metrics, figures, and evaluation artifacts.
- `docs/`: detailed project and dataset documentation.
