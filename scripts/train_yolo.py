#!/usr/bin/env python3
"""Train the class-agnostic PlantSeg lesion detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from plant_disease.paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, TRAINING_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=detection["epochs"])
    parser.add_argument("--batch-size", type=int, default=detection["batch_size"])
    parser.add_argument("--image-size", type=int, default=detection["image_size"])
    parser.add_argument("--workers", type=int, default=detection["num_workers"])
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA device ID")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=TRAINING_DIR / "PlantSegDetection" / "dataset.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace an existing trained run.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "yolo")
    parser.add_argument("--run-name", default=detection["run_name"])
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        help="Limit both train and validation data for a primary-settings preflight.",
    )
    return parser.parse_args()


def yolo_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def limited_dataset(source: Path, output_dir: Path, limit: int) -> Path:
    """Create a temporary YOLO YAML using a deterministic slice of each split."""
    if limit < 1:
        raise ValueError("max-images-per-split must be positive")
    dataset = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    subset_dir = output_dir / ".preflight_data"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        image_dir = root / dataset[split]
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.casefold() in IMAGE_SUFFIXES
        )[:limit]
        if not images:
            raise FileNotFoundError(f"No images available for YOLO {split} preflight")
        list_path = subset_dir / f"{split}.txt"
        list_path.write_text(
            "".join(f"{path.absolute()}\n" for path in images),
            encoding="utf-8",
        )
        dataset[split] = str(list_path.absolute())
    subset_yaml = subset_dir / "dataset.yaml"
    subset_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return subset_yaml


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if not args.data.is_file():
        raise FileNotFoundError(f"YOLO dataset not found: {args.data}; run make prepare-data first")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.image_size < 32
        or args.workers < 0
        or not 0 < args.fraction <= 1
    ):
        raise ValueError("Invalid YOLO training settings")

    detection = config["detection"]
    project = args.output_dir
    run_name = args.run_name
    training_data = args.data
    if args.max_images_per_split is not None:
        training_data = limited_dataset(
            args.data,
            project / run_name,
            args.max_images_per_split,
        )
        print(
            f"Primary-settings preflight: using at most {args.max_images_per_split} "
            "images per split"
        )
    last_checkpoint = project / run_name / "weights" / "last.pt"
    best_checkpoint = project / run_name / "weights" / "best.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_checkpoint}")
        model = YOLO(last_checkpoint)
        print(f"Resuming YOLO from {last_checkpoint}")
        model.train(resume=True)
        return
    if best_checkpoint.is_file() and not args.force:
        print(f"Trained YOLO checkpoint already exists; reusing {best_checkpoint}")
        print("Pass --force to restart training or --resume to continue the existing run")
        return

    pretrained = MODELS_DIR / detection["model"]
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"YOLO base checkpoint not found: {pretrained}; run make init first"
        )
    print("Training a one-class lesion detector; disease labels are reserved for the classifiers")
    model = YOLO(pretrained)
    model.train(
        data=str(training_data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        device=yolo_device(args.device),
        project=str(project),
        name=run_name,
        exist_ok=True,
        seed=config["project"]["seed"],
        deterministic=True,
        fraction=args.fraction,
    )
    print(f"YOLO training complete; best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
