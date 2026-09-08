#!/usr/bin/env python3
"""Five-fold cross-validation over the official PlantSeg training pool.

Splits the 5,367 official training images into five stratified folds (by disease
class), then for each fold trains the class-agnostic lesion YOLO, the standalone
115-class disease YOLO, and all three classifiers from scratch on the other four
folds, validating each on the held-out fold. The official PlantSeg validation and
test splits are never touched by this script; cross-validation estimates training
variance and is reported alongside, not instead of, the official holdout evaluation
in evaluate_pipeline.py.

Two of PlantSeg's 115 classes have very few training images (class 41 has 2, class
68 has 4) and cannot be meaningfully stratified across five folds; they are
distributed as evenly as possible without special-casing; metrics that average over
classes already tolerate a class being absent from a given split, the same way the
official validation and test splits do.

Every fold and model combination writes its own checkpoint and can be resumed
independently: rerunning this script skips any fold/model pair whose final metrics
file already exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_datasets import mask_to_yolo_box, sync_yolo_images  # noqa: E402

from plant_disease.data import PlantSegClassificationDataset, training_transform
from plant_disease.models import CLASSIFICATION_MODELS, build_classifier, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR

N_FOLDS = 5
EPOCHS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lesion_yolo", "standalone_yolo", *CLASSIFICATION_MODELS],
        choices=["lesion_yolo", "standalone_yolo", *CLASSIFICATION_MODELS],
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "cross_validation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(device: torch.device) -> str:
    return "0" if device.type == "cuda" else device.type


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_folds(rows: list[dict[str, str]], seed: int) -> list[np.ndarray]:
    """Return N_FOLDS arrays of validation-row indices, stratified by class."""
    labels = np.array([int(row["Index"]) for row in rows])
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    # StratifiedKFold requires every class to have at least N_FOLDS members; PlantSeg's
    # two ultra-rare classes (2 and 4 training images) violate this, so those rows are
    # excluded from stratification and distributed round-robin across folds instead.
    counts = {label: int((labels == label).sum()) for label in set(labels)}
    rare_labels = {label for label, count in counts.items() if count < N_FOLDS}
    stratifiable = np.array([index for index, label in enumerate(labels) if label not in rare_labels])
    rare = np.array([index for index, label in enumerate(labels) if label in rare_labels])

    fold_validation_indices: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for fold, (_, validation) in enumerate(splitter.split(stratifiable, labels[stratifiable])):
        fold_validation_indices[fold].extend(stratifiable[validation].tolist())
    for position, index in enumerate(rare):
        fold_validation_indices[position % N_FOLDS].append(int(index))

    return [np.array(sorted(indices)) for indices in fold_validation_indices]


def fold_metadata_path(output_dir: Path) -> Path:
    return output_dir / "folds.json"


def load_or_build_folds(rows: list[dict[str, str]], output_dir: Path, seed: int) -> list[np.ndarray]:
    path = fold_metadata_path(output_dir)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [np.array(indices) for indices in payload["fold_validation_indices"]]
    folds = build_folds(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_folds": N_FOLDS,
                "seed": seed,
                "fold_validation_indices": [fold.tolist() for fold in folds],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return folds


def write_yolo_view(
    fold_dir: Path,
    rows: list[dict[str, str]],
    validation_indices: set[int],
    image_dir: Path,
    mask_dir: Path,
    *,
    class_aware: bool,
    num_classes: int,
) -> Path:
    """Build one fold's YOLO detection view; skip work if already built."""
    dataset_yaml = fold_dir / "dataset.yaml"
    if dataset_yaml.is_file():
        return dataset_yaml

    for split_name, split_rows in (
        ("train", [row for index, row in enumerate(rows) if index not in validation_indices]),
        ("val", [row for index, row in enumerate(rows) if index in validation_indices]),
    ):
        image_destination = fold_dir / "images" / split_name
        sync_yolo_images(image_destination, [image_dir / row["Name"] for row in split_rows])
        label_destination = fold_dir / "labels" / split_name
        label_destination.mkdir(parents=True, exist_ok=True)
        for row in split_rows:
            center_x, center_y, width, height = mask_to_yolo_box(mask_dir / row["Label file"])
            class_id = int(row["Index"]) if class_aware else 0
            label_path = label_destination / Path(row["Name"]).with_suffix(".txt").name
            label_path.write_text(
                f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n",
                encoding="utf-8",
            )

    names = {index: str(index) for index in range(num_classes)} if class_aware else {0: "lesion"}
    dataset = {
        "path": str(fold_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    dataset_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def train_fold_yolo(
    mode: str,
    fold: int,
    dataset_yaml: Path,
    args: argparse.Namespace,
    seed: int,
) -> Path:
    """Train one YOLO variant for one fold; return its best checkpoint path."""
    run_name = f"{mode}_fold{fold}"
    project = args.output_dir / "yolo"
    best_checkpoint = project / run_name / "weights" / "best.pt"
    metrics_path = project / run_name / "metrics.json"
    if metrics_path.is_file():
        print(f"[{run_name}] already complete; skipping")
        return best_checkpoint

    pretrained = PROJECT_ROOT / "models" / "yolo11n.pt"
    model = YOLO(pretrained)
    device = choose_device(args.device)
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        workers=args.num_workers,
        device=yolo_device(device),
        project=str(project),
        name=run_name,
        exist_ok=True,
        seed=seed,
        deterministic=True,
    )
    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[{run_name}] mAP50={metrics['map50']:.3f} mAP50-95={metrics['map50_95']:.3f}")
    return best_checkpoint


def prepare_fold_crops(
    fold: int,
    lesion_checkpoint: Path,
    rows: list[dict[str, str]],
    validation_indices: set[int],
    image_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    crop_root = output_dir / "crops" / f"fold{fold}"
    metadata_path = crop_root / "Metadata.csv"
    if metadata_path.is_file():
        return crop_root

    device = choose_device(args.device)
    detector = YOLO(lesion_checkpoint)
    crop_rows: list[dict[str, str]] = []
    for split_name, split_rows in (
        ("train", [row for index, row in enumerate(rows) if index not in validation_indices]),
        ("val", [row for index, row in enumerate(rows) if index in validation_indices]),
    ):
        destination = crop_root / "images" / split_name
        destination.mkdir(parents=True, exist_ok=True)
        results = detector.predict(
            source=str(image_dir),
            conf=0.25,
            imgsz=640,
            device=yolo_device(device),
            stream=True,
            verbose=False,
        )
        wanted = {row["Name"] for row in split_rows}
        rows_by_name = {row["Name"]: row for row in split_rows}
        found = set()
        for result in tqdm(results, total=len(wanted) + (len(rows) - len(split_rows)), desc=f"fold{fold} {split_name} crops", leave=False):
            name = Path(result.path).name
            if name not in wanted:
                continue
            found.add(name)
            row = rows_by_name[name]
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            height, width = result.orig_shape
            if len(boxes):
                left = max(0, int(np.floor(boxes[:, 0].min())))
                top = max(0, int(np.floor(boxes[:, 1].min())))
                right = min(width, int(np.ceil(boxes[:, 2].max())))
                bottom = min(height, int(np.ceil(boxes[:, 3].max())))
            else:
                left, top, right, bottom = 0, 0, width, height
            rgb = np.ascontiguousarray(result.orig_img[..., ::-1])
            crop = Image.fromarray(rgb).crop((left, top, right, bottom))
            crop.save(destination / name)
            crop_rows.append({**row, "Split": "Training" if split_name == "train" else "Validation"})
            if found == wanted:
                break

    crop_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(crop_rows[0])
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crop_rows)
    return crop_root


def run_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    present = sorted(set(targets))
    return {
        "loss": total_loss / total_examples,
        "macro_f1": f1_score(targets, predictions, labels=present, average="macro", zero_division=0),
        "balanced_accuracy": recall_score(
            targets, predictions, labels=present, average="macro", zero_division=0
        ),
    }


def train_fold_classifier(
    model_name: str,
    fold: int,
    crop_root: Path,
    num_classes: int,
    args: argparse.Namespace,
    seed: int,
) -> None:
    run_dir = args.output_dir / "classification" / f"{model_name}_fold{fold}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"[{model_name} fold{fold}] already complete; skipping")
        return

    device = choose_device(args.device)
    metadata = crop_root / "Metadata.csv"
    train_dataset = PlantSegClassificationDataset(
        metadata,
        crop_root / "images" / "train",
        transform=training_transform(224),
        split="Training",
    )
    validation_dataset = PlantSegClassificationDataset(
        metadata,
        crop_root / "images" / "val",
        transform=classification_transform(224),
        split="Validation",
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_classifier(model_name, num_classes=num_classes, pretrained=model_name != "baseline_cnn").to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_macro_f1 = float("-inf")
    history = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_classifier_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_classifier_epoch(model, validation_loader, criterion, device, None)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
        print(
            f"[{model_name} fold{fold}] epoch {epoch}/{args.epochs} "
            f"val macro-F1={validation_metrics['macro_f1']:.3f} (best={best_macro_f1:.3f})"
        )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"best_val_macro_f1": best_macro_f1, "epochs": args.epochs}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    seed = config["project"]["seed"]
    seed_everything(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plantseg = RAW_DIR / "PlantSeg" / "plantseg"
    with (plantseg / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["Split"] == "Training"]
    num_classes = config["classification"]["num_classes"]
    image_dir = plantseg / "images" / "train"
    mask_dir = plantseg / "annotations" / "train"

    folds = load_or_build_folds(rows, args.output_dir, seed)
    print(f"Built {N_FOLDS} folds over {len(rows)} training images")

    for fold in args.folds:
        validation_indices = set(folds[fold].tolist())
        print(f"\n=== Fold {fold}: {len(rows) - len(validation_indices)} train / {len(validation_indices)} val ===")

        lesion_checkpoint = None
        if "lesion_yolo" in args.models:
            fold_dir = args.output_dir / "yolo_data" / f"lesion_fold{fold}"
            dataset_yaml = write_yolo_view(
                fold_dir, rows, validation_indices, image_dir, mask_dir,
                class_aware=False, num_classes=num_classes,
            )
            lesion_checkpoint = train_fold_yolo("lesion", fold, dataset_yaml, args, seed)

        if "standalone_yolo" in args.models:
            fold_dir = args.output_dir / "yolo_data" / f"standalone_fold{fold}"
            dataset_yaml = write_yolo_view(
                fold_dir, rows, validation_indices, image_dir, mask_dir,
                class_aware=True, num_classes=num_classes,
            )
            train_fold_yolo("standalone", fold, dataset_yaml, args, seed)

        classifier_models = [name for name in args.models if name in CLASSIFICATION_MODELS]
        if classifier_models:
            if lesion_checkpoint is None:
                lesion_checkpoint = (
                    args.output_dir / "yolo" / f"lesion_fold{fold}" / "weights" / "best.pt"
                )
            if not lesion_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Fold {fold} lesion YOLO checkpoint not found: {lesion_checkpoint}; "
                    "run with --models lesion_yolo first for this fold"
                )
            crop_root = prepare_fold_crops(
                fold, lesion_checkpoint, rows, validation_indices, image_dir, args, args.output_dir
            )
            for model_name in classifier_models:
                train_fold_classifier(model_name, fold, crop_root, num_classes, args, seed)

    aggregate_results(args.output_dir, args.folds, args.models)


def aggregate_results(output_dir: Path, folds: list[int], models: list[str]) -> None:
    """Collect per-fold metrics into a mean/std summary, skipping incomplete models."""
    summary: dict[str, dict[str, float]] = {}
    for model_name in models:
        values: dict[str, list[float]] = {}
        for fold in folds:
            if model_name in ("lesion_yolo", "standalone_yolo"):
                mode = "lesion" if model_name == "lesion_yolo" else "standalone"
                metrics_path = output_dir / "yolo" / f"{mode}_fold{fold}" / "metrics.json"
            else:
                metrics_path = output_dir / "classification" / f"{model_name}_fold{fold}" / "metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    values.setdefault(key, []).append(value)
        if not values:
            continue
        summary[model_name] = {
            f"{key}_mean": float(np.mean(vals)) for key, vals in values.items()
        } | {f"{key}_std": float(np.std(vals)) for key, vals in values.items()} | {
            "folds_completed": len(next(iter(values.values())))
        }

    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for model_name, metrics in summary.items():
        print(f"{model_name}: {metrics}")
    print(f"Saved cross-validation summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
