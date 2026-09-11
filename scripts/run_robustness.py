#!/usr/bin/env python3
"""Evaluate pipeline and standalone YOLO robustness on the PlantSeg test set.

Applies each corruption in configs/project.yaml at every configured severity level to a
fixed, reproducible subset of the official PlantSeg test split, then runs the same
lesion-YOLO-to-classifier pipelines and disease-aware YOLO on each corrupted variant.
Corruptions are generated on demand rather than stored, matching the "on_demand" plan
recorded in data/tests/robustness/PlantSeg/variants.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from ultralytics import YOLO

from plant_disease.data import plantseg_class_names
from plant_disease.models import CLASSIFICATION_MODELS, build_classifier, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TESTS_DIR

SUBSET_SIZE = 200


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    classification = config["classification"]
    robustness = config["robustness"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=(OUTPUTS_DIR / "yolo" / detection["run_name"] / "weights" / "best.pt"),
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=OUTPUTS_DIR / "classification",
    )
    parser.add_argument(
        "--standalone-yolo-checkpoint",
        type=Path,
        default=(OUTPUTS_DIR / "yolo" / detection["standalone_run_name"] / "weights" / "best.pt"),
        help="Disease-aware YOLO checkpoint used for the standalone comparison.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIFICATION_MODELS,
        default=classification["models"],
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--confidence", type=float, default=detection["crop_confidence"])
    parser.add_argument("--margin", type=float, default=detection["crop_margin"])
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=robustness["corruptions"],
        choices=robustness["corruptions"],
    )
    parser.add_argument(
        "--severity-levels",
        nargs="+",
        type=int,
        default=robustness["severity_levels"],
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=SUBSET_SIZE,
        help="Fixed reproducible PlantSeg test subset size (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "robustness" / "plantseg_test",
    )
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


def expanded_union_box(
    boxes: np.ndarray, image_width: int, image_height: int, margin: float
) -> tuple[int, int, int, int]:
    left = float(boxes[:, 0].min())
    top = float(boxes[:, 1].min())
    right = float(boxes[:, 2].max())
    bottom = float(boxes[:, 3].max())
    padding_x = (right - left) * margin
    padding_y = (bottom - top) * margin
    return (
        max(0, int(np.floor(left - padding_x))),
        max(0, int(np.floor(top - padding_y))),
        min(image_width, int(np.ceil(right + padding_x))),
        min(image_height, int(np.ceil(bottom + padding_y))),
    )


def select_subset(rows: list[dict], subset_size: int, seed: int) -> list[dict]:
    """Deterministically sample a fixed PlantSeg test subset, reused across runs."""
    ordered = sorted(rows, key=lambda row: row["Name"])
    if subset_size >= len(ordered):
        return ordered
    generator = random.Random(seed)
    return generator.sample(ordered, subset_size)


def apply_corruption(image: Image.Image, corruption: str, severity: int) -> Image.Image:
    """Apply one corruption at one of five increasing severity levels."""
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in 1..5")

    if corruption == "gaussian_blur":
        radius = severity * 1.2
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        # Alternate dimmer/brighter so severity 5 is not simply "very bright".
        factor = [0.7, 0.55, 1.6, 1.9, 2.3][severity - 1]
        return ImageEnhance.Brightness(image).enhance(factor)

    if corruption == "contrast":
        factor = [0.7, 0.5, 1.6, 1.9, 2.3][severity - 1]
        return ImageEnhance.Contrast(image).enhance(factor)

    if corruption == "jpeg_compression":
        quality = [50, 35, 22, 12, 5][severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if corruption == "occlusion":
        fraction = [0.10, 0.16, 0.22, 0.30, 0.40][severity - 1]
        occluded = image.copy()
        width, height = occluded.size
        box_width = int(width * fraction)
        box_height = int(height * fraction)
        generator = random.Random(hash((corruption, severity, width, height)) & 0xFFFFFFFF)
        left = generator.randint(0, max(0, width - box_width))
        top = generator.randint(0, max(0, height - box_height))
        gray = Image.new("RGB", (box_width, box_height), (128, 128, 128))
        occluded.paste(gray, (left, top))
        return occluded

    if corruption == "crop":
        # Increasing severity keeps a smaller centered fraction of the frame, then
        # resizes back to the original size, simulating a tighter or looser photo.
        keep_fraction = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
        width, height = image.size
        crop_width = int(width * keep_fraction)
        crop_height = int(height * keep_fraction)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.BILINEAR)

    raise ValueError(f"Unknown corruption: {corruption!r}")


def load_classifiers(
    names: list[str],
    classifier_dir: Path,
    class_names: list[str],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models = {}
    for name in names:
        checkpoint_path = classifier_dir / name / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != name or checkpoint["num_classes"] != len(class_names):
            raise RuntimeError(f"Classifier checkpoint metadata is invalid: {checkpoint_path}")
        if checkpoint["class_names"] != class_names:
            raise RuntimeError(f"Classifier taxonomy differs from PlantSeg: {checkpoint_path}")
        model = build_classifier(name, len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(device)
        models[name] = model
        print(f"Loaded {name}")
    return models


def standalone_prediction(result) -> tuple[int, float]:
    """Return the highest-confidence disease prediction, or -1 when nothing is detected."""
    if not len(result.boxes):
        return -1, 0.0
    best = int(result.boxes.conf.argmax().item())
    return int(result.boxes.cls[best].item()), float(result.boxes.conf[best].item())


def evaluate_variant(
    detector: YOLO,
    standalone_detector: YOLO,
    models: dict[str, torch.nn.Module],
    rows: list[dict],
    test_root: Path,
    corruption: str | None,
    severity: int | None,
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    transform,
) -> dict[str, dict[str, float]]:
    """Run the lesion-YOLO-to-classifier pipeline on one corruption/severity variant."""
    targets: list[int] = []
    system_names = [*models, "standalone_yolo"]
    predictions: dict[str, list[int]] = {name: [] for name in system_names}
    confidences: dict[str, list[float]] = {name: [] for name in system_names}
    fallback_count = 0

    for row in rows:
        image_path = test_root / "images" / row["Name"]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if corruption is not None:
            image = apply_corruption(image, corruption, severity)
        width, height = image.size

        results = detector.predict(
            source=image,
            conf=args.confidence,
            imgsz=config["detection"]["image_size"],
            device=yolo_device(device),
            verbose=False,
        )
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        if len(boxes):
            left, top, right, bottom = expanded_union_box(boxes, width, height, args.margin)
        else:
            left, top, right, bottom = 0, 0, width, height
            fallback_count += 1
        crop = image.crop((left, top, right, bottom))
        tensor = transform(crop).unsqueeze(0).to(device)

        targets.append(int(row["Index"]))
        with torch.inference_mode():
            for name, model in models.items():
                probabilities = model(tensor).softmax(dim=1)
                confidence, prediction = probabilities.max(dim=1)
                predictions[name].append(int(prediction.item()))
                confidences[name].append(float(confidence.item()))

        standalone_result = standalone_detector.predict(
            source=image,
            conf=args.confidence,
            imgsz=config["detection"]["image_size"],
            device=yolo_device(device),
            verbose=False,
        )[0]
        prediction, confidence = standalone_prediction(standalone_result)
        predictions["standalone_yolo"].append(prediction)
        confidences["standalone_yolo"].append(confidence)

    results_by_model: dict[str, dict[str, float]] = {}
    for name in system_names:
        present = sorted(set(targets))
        prediction_array = np.asarray(predictions[name])
        results_by_model[name] = {
            "accuracy": float(accuracy_score(targets, predictions[name])),
            "macro_f1": float(
                f1_score(
                    targets, predictions[name], labels=present, average="macro", zero_division=0
                )
            ),
            "recall_macro": float(
                recall_score(
                    targets, predictions[name], labels=present, average="macro", zero_division=0
                )
            ),
            "mean_confidence": float(np.mean(confidences[name])),
            "coverage": float(np.mean(prediction_array >= 0)),
            "no_detections": int(np.sum(prediction_array < 0)),
        }
    results_by_model["_meta"] = {
        "images": len(rows),
        "full_image_fallback_rate": fallback_count / len(rows),
    }
    return results_by_model


def save_plot(records: list[dict], output_dir: Path, model_names: list[str]) -> None:
    corruptions = sorted(
        {record["corruption"] for record in records if record["corruption"] != "clean"}
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for axis, corruption in zip(axes.flat, corruptions, strict=False):
        for name in model_names:
            severities = sorted(
                record["severity"]
                for record in records
                if record["corruption"] == corruption and record["model"] == name
            )
            values = [
                next(
                    record["accuracy"]
                    for record in records
                    if record["corruption"] == corruption
                    and record["model"] == name
                    and record["severity"] == severity
                )
                for severity in severities
            ]
            axis.plot(severities, values, marker="o", label=name)
        axis.set_title(corruption)
        axis.set_xlabel("severity")
        axis.set_ylim(0, 1)
    axes.flat[0].set_ylabel("accuracy")
    axes.flat[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_vs_severity.png", dpi=130)
    plt.close(figure)


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Lesion YOLO checkpoint not found: {args.yolo_checkpoint}; run make train-yolo-lesion"
        )
    if not args.standalone_yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Disease-aware YOLO checkpoint not found: {args.standalone_yolo_checkpoint}"
        )
    if not all(1 <= severity <= 5 for severity in args.severity_levels):
        raise ValueError("severity levels must be in 1..5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    metadata_path = RAW_DIR / "PlantSeg" / "plantseg" / "Metadata.csv"
    class_names = plantseg_class_names(metadata_path, config["classification"]["num_classes"])
    models = load_classifiers(args.models, args.classifier_dir, class_names, device)
    detector = YOLO(args.yolo_checkpoint)
    lesion_names = [detector.names[index] for index in range(len(detector.names))]
    if lesion_names != ["lesion"]:
        raise RuntimeError(
            f"Crop detector must contain only the 'lesion' class, found: {lesion_names}"
        )
    standalone_detector = YOLO(args.standalone_yolo_checkpoint)
    standalone_names = [
        standalone_detector.names[index] for index in range(len(standalone_detector.names))
    ]
    if standalone_names != class_names:
        raise RuntimeError("Standalone YOLO checkpoint taxonomy differs from PlantSeg")

    test_root = TESTS_DIR / "PlantSeg" / "full"
    with (test_root / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subset = select_subset(rows, args.subset_size, config["project"]["seed"])
    (args.output_dir / "subset.csv").write_text(
        "\n".join(["Name"] + [row["Name"] for row in subset]) + "\n", encoding="utf-8"
    )
    print(f"Evaluating robustness on {device} with a fixed {len(subset)}-image subset")

    transform = classification_transform(config["project"]["image_size"])
    variants: list[tuple[str | None, int | None]] = [(None, None)]
    for corruption in args.corruptions:
        for severity in args.severity_levels:
            variants.append((corruption, severity))

    records: list[dict] = []
    for corruption, severity in tqdm(variants, desc="Robustness variants"):
        label = "clean" if corruption is None else corruption
        variant_results = evaluate_variant(
            detector,
            standalone_detector,
            models,
            subset,
            test_root,
            corruption,
            severity,
            args,
            config,
            device,
            transform,
        )
        meta = variant_results.pop("_meta")
        for model_name, metrics in variant_results.items():
            records.append(
                {
                    "corruption": label,
                    "severity": severity if severity is not None else 0,
                    "model": model_name,
                    "full_image_fallback_rate": meta["full_image_fallback_rate"],
                    **metrics,
                }
            )

    clean_by_model = {
        record["model"]: record for record in records if record["corruption"] == "clean"
    }
    for record in records:
        clean = clean_by_model[record["model"]]
        record["accuracy_drop_absolute"] = clean["accuracy"] - record["accuracy"]
        record["accuracy_drop_relative"] = (
            record["accuracy_drop_absolute"] / clean["accuracy"] if clean["accuracy"] > 0 else 0.0
        )
        record["macro_f1_drop_absolute"] = clean["macro_f1"] - record["macro_f1"]
        record["confidence_drop_absolute"] = clean["mean_confidence"] - record["mean_confidence"]

    fieldnames = list(records[0])
    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    plot_records = [
        {**record, "severity": record["severity"] if record["corruption"] != "clean" else 0}
        for record in records
        if record["corruption"] != "clean"
    ]
    if plot_records:
        save_plot(plot_records, args.output_dir, [*models, "standalone_yolo"])

    summary = {
        "dataset": "PlantSeg test subset (fixed, reproducible)",
        "subset_size": len(subset),
        "yolo_checkpoint": str(args.yolo_checkpoint.resolve()),
        "standalone_yolo_checkpoint": str(args.standalone_yolo_checkpoint.resolve()),
        "classifiers": list(models),
        "systems": [*models, "standalone_yolo"],
        "standalone_policy": (
            "Highest-confidence box; a missed detection is an incorrect prediction with "
            "confidence zero"
        ),
        "corruptions": args.corruptions,
        "severity_levels": args.severity_levels,
        "clean_baseline": clean_by_model,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    for model_name, clean in clean_by_model.items():
        print(f"{model_name} clean accuracy: {clean['accuracy']:.3f}")
    print(f"Saved robustness evaluation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
