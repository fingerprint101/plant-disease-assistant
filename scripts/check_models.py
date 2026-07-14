#!/usr/bin/env python3
"""Download missing weights and smoke-test all four project models."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights

from plant_disease.models import (
    CLASSIFICATION_MODELS,
    build_classifier,
    build_gradcam,
    classification_transform,
    load_yolo,
)
from plant_disease.paths import MODELS_DIR, PROJECT_ROOT, RAW_DIR

PRETRAINED_WEIGHTS = {
    "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
    "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--plantvillage-dir",
        type=Path,
        default=RAW_DIR / "PlantVillage" / "raw" / "color",
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=Path,
        default=RAW_DIR / "PlantDoc" / "images",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_images(directory: Path, count: int) -> list[Path]:
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < count:
        raise FileNotFoundError(f"Expected at least {count} images below {directory}")
    return images[:count]


def verify_classifier_checkpoint(name: str, model_dir: Path) -> None:
    weights = PRETRAINED_WEIGHTS.get(name)
    if weights is None:
        return
    checkpoint = model_dir / "hub" / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"torchvision did not create {checkpoint}")
    expected_prefix = checkpoint.stem.rsplit("-", maxsplit=1)[-1]
    if not sha256(checkpoint).startswith(expected_prefix):
        raise RuntimeError(f"Checksum verification failed for {checkpoint}")


def check_classifiers(dataset_dir: Path, model_dir: Path) -> None:
    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    if len(class_names) != 38:
        raise RuntimeError(f"Expected 38 PlantVillage classes, found {len(class_names)}")
    transform = classification_transform()
    batch = torch.stack(
        [transform(Image.open(path).convert("RGB")) for path in sample_images(dataset_dir, 2)]
    )

    for name in CLASSIFICATION_MODELS:
        print(f"Checking {name} ...", flush=True)
        model = build_classifier(name, num_classes=len(class_names), pretrained=True)
        model.eval()
        with torch.inference_mode():
            logits = model(batch)
            probabilities = logits.softmax(dim=1)
        expected_shape = (batch.shape[0], len(class_names))
        if tuple(logits.shape) != expected_shape or not torch.isfinite(logits).all():
            raise RuntimeError(f"{name} returned invalid logits with shape {tuple(logits.shape)}")
        if not torch.allclose(probabilities.sum(dim=1), torch.ones(batch.shape[0]), atol=1e-5):
            raise RuntimeError(f"{name} returned invalid probabilities")

        model.train()
        model.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(batch), torch.arange(batch.shape[0]))
        loss.backward()
        if not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"{name} did not produce finite gradients")

        model.eval()
        with build_gradcam(name, model) as gradcam:
            activation = gradcam(
                input_tensor=batch[:1],
                targets=[ClassifierOutputTarget(0)],
            )
        if activation.shape != (1, batch.shape[2], batch.shape[3]):
            raise RuntimeError(f"{name} Grad-CAM returned invalid shape {activation.shape}")
        if not torch.isfinite(torch.from_numpy(activation)).all():
            raise RuntimeError(f"{name} Grad-CAM returned non-finite values")
        verify_classifier_checkpoint(name, model_dir)
        print(
            f"  [ok] output={list(logits.shape)}, gradients=finite, gradcam=valid",
            flush=True,
        )


def check_detector(plantdoc_dir: Path, model_dir: Path) -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text())
    checkpoint = model_dir / str(config["detection"]["model"])
    existed = checkpoint.is_file()
    print(
        f"Using cached detection checkpoint {checkpoint}"
        if existed
        else f"Downloading detection checkpoint to {checkpoint}",
        flush=True,
    )
    model = load_yolo(checkpoint.name, model_dir=model_dir)
    results = model.predict(
        source=str(sample_images(plantdoc_dir, 1)[0]),
        imgsz=320,
        device="cpu",
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"YOLO returned {len(results)} results; expected 1")
    result = results[0]
    if (
        result.orig_shape[0] <= 0
        or result.orig_shape[1] <= 0
        or result.boxes is None
        or not torch.isfinite(result.boxes.data).all()
    ):
        raise RuntimeError("YOLO returned invalid detection data")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"YOLO checkpoint was not created correctly at {checkpoint}")
    action = "reused" if existed else "downloaded"
    print(
        f"[ok] yolo11n: checkpoint={action}, detections={len(result.boxes)}, "
        f"input_shape={list(result.orig_shape)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.model_dir.resolve())
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    check_classifiers(args.plantvillage_dir, args.model_dir)
    check_detector(args.plantdoc_dir, args.model_dir)
    print("All 4 planned models passed")


if __name__ == "__main__":
    main()
