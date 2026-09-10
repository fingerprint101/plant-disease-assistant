#!/usr/bin/env python3
"""Benchmark end-to-end batch-1 latency for the PlantSeg systems."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from plant_disease.models import CLASSIFICATION_MODELS, build_classifier, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, TESTS_DIR


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=OUTPUTS_DIR / "yolo" / detection["run_name"] / "weights" / "best.pt",
    )
    parser.add_argument(
        "--standalone-yolo-checkpoint",
        type=Path,
        default=(OUTPUTS_DIR / "yolo" / detection["standalone_run_name"] / "weights" / "best.pt"),
    )
    parser.add_argument("--classifier-dir", type=Path, default=OUTPUTS_DIR / "classification")
    parser.add_argument(
        "--models", nargs="+", choices=CLASSIFICATION_MODELS, default=CLASSIFICATION_MODELS
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=detection["crop_confidence"])
    parser.add_argument("--margin", type=float, default=detection["crop_margin"])
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR / "evaluation" / "efficiency"
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(device: torch.device) -> str:
    return "0" if device.type == "cuda" else device.type


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def expanded_union_box(
    boxes: np.ndarray, image_width: int, image_height: int, margin: float
) -> tuple[int, int, int, int]:
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = boxes[:, 2:].max(axis=0)
    padding_x = (right - left) * margin
    padding_y = (bottom - top) * margin
    return (
        max(0, int(np.floor(left - padding_x))),
        max(0, int(np.floor(top - padding_y))),
        min(image_width, int(np.ceil(right + padding_x))),
        min(image_height, int(np.ceil(bottom + padding_y))),
    )


def checkpoint_parameter_count(checkpoint: Path, kind: str, classes: int = 115) -> int:
    if kind == "yolo":
        return sum(parameter.numel() for parameter in YOLO(checkpoint).model.parameters())
    model = build_classifier(kind, classes, pretrained=False)
    return sum(parameter.numel() for parameter in model.parameters())


def load_classifier(name: str, directory: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(directory / name / "best.pt", map_location="cpu", weights_only=True)
    model = build_classifier(name, checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval().to(device)


def predict_detector(
    detector: YOLO, image_path: Path, args: argparse.Namespace, device: torch.device
):
    return detector.predict(
        source=str(image_path),
        conf=args.confidence,
        imgsz=640,
        batch=1,
        device=yolo_device(device),
        verbose=False,
    )[0]


def benchmark_standalone(
    detector: YOLO, images: list[Path], args: argparse.Namespace, device: torch.device
) -> list[float]:
    latencies = []
    for index, image_path in enumerate(images):
        synchronize(device)
        started = time.perf_counter_ns()
        predict_detector(detector, image_path, args, device)
        synchronize(device)
        if index >= args.warmup_images:
            latencies.append((time.perf_counter_ns() - started) / 1e6)
    return latencies


def benchmark_pipeline(
    detector: YOLO,
    classifier: torch.nn.Module,
    transform,
    images: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    latencies = []
    for index, image_path in enumerate(images):
        synchronize(device)
        started = time.perf_counter_ns()
        result = predict_detector(detector, image_path, args, device)
        height, width = result.orig_shape
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        if len(boxes):
            bounds = expanded_union_box(boxes, width, height, args.margin)
        else:
            bounds = (0, 0, width, height)
        rgb = np.ascontiguousarray(result.orig_img[..., ::-1])
        crop = Image.fromarray(rgb).crop(bounds)
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.inference_mode():
            classifier(tensor)
        synchronize(device)
        if index >= args.warmup_images:
            latencies.append((time.perf_counter_ns() - started) / 1e6)
    return latencies


def summarize(latencies: list[float]) -> dict[str, float | int]:
    values = np.asarray(latencies)
    return {
        "timed_images": len(values),
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "throughput_images_per_second": float(1000 / values.mean()),
    }


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/project.yaml").read_text())
    args = parse_args(config)
    if args.sample_size < 1 or args.warmup_images < 0:
        raise ValueError("sample-size must be positive and warmup-images cannot be negative")
    for checkpoint in (args.yolo_checkpoint, args.standalone_yolo_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    test_root = TESTS_DIR / "PlantSeg" / "full"
    with (test_root / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample = random.Random(config["project"]["seed"]).sample(
        rows, min(args.sample_size + args.warmup_images, len(rows))
    )
    images = [test_root / "images" / row["Name"] for row in sample]
    device = choose_device(args.device)
    transform = classification_transform(config["project"]["image_size"])

    results = {}
    standalone = YOLO(args.standalone_yolo_checkpoint)
    results["standalone_yolo"] = summarize(
        benchmark_standalone(standalone, images, args, device)
    ) | {
        "parameters": checkpoint_parameter_count(args.standalone_yolo_checkpoint, "yolo"),
        "checkpoint_size_mb": args.standalone_yolo_checkpoint.stat().st_size / 1e6,
    }
    del standalone

    lesion = YOLO(args.yolo_checkpoint)
    lesion_parameters = checkpoint_parameter_count(args.yolo_checkpoint, "yolo")
    lesion_size = args.yolo_checkpoint.stat().st_size / 1e6
    for name in args.models:
        classifier_path = args.classifier_dir / name / "best.pt"
        classifier = load_classifier(name, args.classifier_dir, device)
        results[name] = summarize(
            benchmark_pipeline(lesion, classifier, transform, images, args, device)
        ) | {
            "parameters": lesion_parameters + checkpoint_parameter_count(classifier_path, name),
            "checkpoint_size_mb": lesion_size + classifier_path.stat().st_size / 1e6,
        }
        del classifier

    payload = {
        "benchmark": "End-to-end single-image inference",
        "device": str(device),
        "hardware": platform.machine() + " / " + platform.platform(),
        "sample_size": len(images) - args.warmup_images,
        "warmup_images": args.warmup_images,
        "batch_size": 1,
        "timing_includes": (
            "Image loading, YOLO inference, crop and preprocessing, and one classifier "
            "where applicable"
        ),
        "timing_excludes": "One-time model loading",
        "seed": config["project"]["seed"],
        "systems": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
