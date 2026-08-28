#!/usr/bin/env python3
"""Train every configured classification model on PlantSeg."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from plant_disease.data import (
    PlantSegClassificationDataset,
    plantseg_class_names,
    training_transform,
)
from plant_disease.models import CLASSIFICATION_MODELS, build_classifier, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TRAINING_DIR


def parse_args(config: dict) -> argparse.Namespace:
    classification = config["classification"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIFICATION_MODELS,
        default=classification["models"],
        help="Models to train sequentially (default: every configured classifier).",
    )
    parser.add_argument("--epochs", type=int, default=classification["epochs"])
    parser.add_argument("--batch-size", type=int, default=classification["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=classification["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=classification["weight_decay"])
    parser.add_argument("--num-workers", type=int, default=classification["num_workers"])
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "classification",
        help="Root directory for checkpoints and metric histories.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=TRAINING_DIR / "PlantSegCrops",
        help="YOLO-generated PlantSeg crop dataset.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume each model from last.pt.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize EfficientNet or MobileNet with ImageNet weights.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit training batches per epoch (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Limit validation batches per epoch (useful for smoke tests).",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    crop_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, Path]:
    crop_metadata = crop_root / "Metadata.csv"
    if not crop_metadata.is_file():
        raise FileNotFoundError(
            f"YOLO classifier crops not found at {crop_root}; run make prepare-crops first"
        )
    raw_plantseg = RAW_DIR / "PlantSeg" / "plantseg"
    full_metadata = raw_plantseg / "Metadata.csv"

    train_dataset = PlantSegClassificationDataset(
        crop_metadata,
        crop_root / "images" / "train",
        transform=training_transform(image_size),
        split="Training",
    )
    validation_dataset = PlantSegClassificationDataset(
        crop_metadata,
        crop_root / "images" / "val",
        transform=classification_transform(image_size),
        split="Validation",
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    return train_loader, validation_loader, full_metadata


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int | None,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    targets: list[int] = []
    predictions: list[int] = []

    progress = tqdm(loader, desc=description, leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
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
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

    if total_examples == 0:
        raise ValueError("No batches were processed; batch limits must be greater than zero")
    present_classes = sorted(set(targets))
    return {
        "loss": total_loss / total_examples,
        "accuracy": sum(
            actual == predicted
            for actual, predicted in zip(targets, predictions, strict=True)
        )
        / total_examples,
        "macro_f1": f1_score(
            targets,
            predictions,
            labels=present_classes,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": recall_score(
            targets,
            predictions,
            labels=present_classes,
            average="macro",
            zero_division=0,
        ),
        "examples": total_examples,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_model(
    model_name: str,
    args: argparse.Namespace,
    config: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    last_path = model_dir / "last.pt"
    best_path = model_dir / "best.pt"
    history_path = model_dir / "history.json"

    model = build_classifier(
        model_name,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    start_epoch = 1
    best_macro_f1 = float("-inf")
    history: list[dict] = []

    if args.resume:
        if not last_path.is_file():
            print(f"[{model_name}] no checkpoint at {last_path}; starting from epoch 1")
        else:
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_macro_f1 = checkpoint["best_macro_f1"]
            history = checkpoint.get("history", [])
            print(f"[{model_name}] resumed after epoch {checkpoint['epoch']}")

    if start_epoch > args.epochs:
        print(f"[{model_name}] already completed {args.epochs} epochs; skipping")
        return

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n[{model_name}] training on {device} ({parameter_count:,} parameters)")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            args.max_train_batches,
            f"{model_name} train {epoch}/{args.epochs}",
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            None,
            args.max_val_batches,
            f"{model_name} val   {epoch}/{args.epochs}",
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        improved = validation_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = validation_metrics["macro_f1"]

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "num_classes": len(class_names),
            "class_names": class_names,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_f1": best_macro_f1,
            "history": history,
            "training_config": {
                "image_size": config["project"]["image_size"],
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "pretrained": not args.no_pretrained,
            },
        }
        save_checkpoint(last_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

        print(
            f"[{model_name}] epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"acc={validation_metrics['accuracy']:.3f} "
            f"macro-F1={validation_metrics['macro_f1']:.3f} "
            f"balanced-acc={validation_metrics['balanced_accuracy']:.3f} "
            f"({epoch_metrics['seconds']:.1f}s){' BEST' if improved else ''}"
        )

    print(f"[{model_name}] complete; best checkpoint: {best_path}")


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch size must be positive; num-workers cannot be negative")
    for value, name in (
        (args.max_train_batches, "max-train-batches"),
        (args.max_val_batches, "max-val-batches"),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")

    seed = config["project"]["seed"]
    seed_everything(seed)
    device = choose_device(args.device)
    train_loader, validation_loader, metadata_path = build_loaders(
        crop_root=args.data_dir,
        image_size=config["project"]["image_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed,
    )
    class_names = plantseg_class_names(metadata_path, config["classification"]["num_classes"])
    print(
        f"PlantSeg YOLO-crop classification: {len(train_loader.dataset):,} training crops, "
        f"{len(validation_loader.dataset):,} validation images, {len(class_names)} classes"
    )
    print(f"Models: {', '.join(args.models)}")
    print(f"Outputs: {args.output_dir.resolve()}")

    for model_name in args.models:
        train_model(
            model_name,
            args,
            config,
            train_loader,
            validation_loader,
            class_names,
            device,
        )


if __name__ == "__main__":
    main()
