#!/usr/bin/env python3
"""Evaluate the PlantSeg pipeline on matched PlantSeg and PlantVillage classes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from evaluate_pipeline import (
    choose_device,
    expanded_union_box,
    load_classifiers,
    yolo_device,
)
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from tqdm import tqdm
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plant_disease.data import plantseg_class_names
from plant_disease.models import CLASSIFICATION_MODELS, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TESTS_DIR


def parse_args(config: dict) -> argparse.Namespace:
    classification = config["classification"]
    detection = config["detection"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=OUTPUTS_DIR / "yolo" / detection["run_name"] / "weights" / "best.pt",
    )
    parser.add_argument("--classifier-dir", type=Path, default=OUTPUTS_DIR / "classification")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIFICATION_MODELS,
        default=classification["models"],
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=classification["batch_size"])
    parser.add_argument("--yolo-batch-size", type=int, default=detection["batch_size"])
    parser.add_argument("--confidence", type=float, default=detection["crop_confidence"])
    parser.add_argument("--margin", type=float, default=detection["crop_margin"])
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "evaluation" / "cross_domain",
    )
    parser.add_argument("--max-images", type=int, help="Limit each domain for a smoke test.")
    return parser.parse_args()


def expected_calibration_error(
    targets: list[int], predictions: list[int], confidences: list[float], bins: int
) -> float:
    """Return top-label expected calibration error using equal-width confidence bins."""
    if bins < 1:
        raise ValueError("calibration-bins must be positive")
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    confidence_array = np.asarray(confidences)
    correct = target_array == prediction_array
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        selected = (confidence_array >= lower) & (
            confidence_array <= upper if index == bins - 1 else confidence_array < upper
        )
        if selected.any():
            error += selected.mean() * abs(
                correct[selected].mean() - confidence_array[selected].mean()
            )
    return float(error)


def load_rows(root: Path, max_images: int | None) -> list[dict[str, str]]:
    metadata_path = root / "Metadata.csv"
    image_dir = root / "images"
    if not metadata_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared overlap data not found below {root}; run make prepare-data"
        )
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["Name"])
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max-images must be positive")
        rows = rows[:max_images]
    missing = [row["Name"] for row in rows if not (image_dir / row["Name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} overlap images are missing below {image_dir}")
    return rows


def classify_pending(
    tensors: list[torch.Tensor],
    records: list[dict[str, object]],
    models: dict[str, torch.nn.Module],
    raw: dict[str, dict[str, list]],
    device: torch.device,
) -> None:
    if not tensors:
        return
    batch = torch.stack(tensors).to(device)
    targets = [int(record["actual_id"]) for record in records]
    with torch.inference_mode():
        for name, model in models.items():
            probabilities = model(batch).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predicted_ids = prediction.cpu().tolist()
            scores = confidence.cpu().tolist()
            raw[name]["targets"].extend(targets)
            raw[name]["predictions"].extend(predicted_ids)
            raw[name]["confidences"].extend(scores)
            for record, predicted_id, score in zip(records, predicted_ids, scores, strict=True):
                record[f"{name}_prediction_id"] = predicted_id
                record[f"{name}_confidence"] = f"{score:.8f}"


def evaluate_domain(
    domain: str,
    root: Path,
    detector: YOLO,
    models: dict[str, torch.nn.Module],
    class_names: list[str],
    shared_ids: list[int],
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    rows = load_rows(root, args.max_images)
    rows_by_name = {row["Name"]: row for row in rows}
    image_dir = root / "images"
    source: str | list[str]
    if args.max_images is None:
        source = str(image_dir)
    else:
        source = [str(image_dir / row["Name"]) for row in rows]

    transform = classification_transform(config["project"]["image_size"])
    raw = {name: {"targets": [], "predictions": [], "confidences": []} for name in models}
    records: list[dict[str, object]] = []
    tensors: list[torch.Tensor] = []
    pending_records: list[dict[str, object]] = []
    fallbacks = 0
    detection_total = 0
    started = time.perf_counter()
    results = detector.predict(
        source=source,
        conf=args.confidence,
        imgsz=config["detection"]["image_size"],
        batch=args.yolo_batch_size,
        device=yolo_device(device),
        stream=True,
        verbose=False,
    )
    progress = tqdm(total=len(rows), desc=domain, unit="image")
    seen: set[str] = set()
    selected_rows = iter(rows) if args.max_images is not None else None
    for result in results:
        if selected_rows is None:
            name = Path(result.path).name
            metadata = rows_by_name.get(name)
            if metadata is None:
                continue
        else:
            metadata = next(selected_rows)
            name = metadata["Name"]
        seen.add(name)
        height, width = result.orig_shape
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        detection_count = len(boxes)
        detection_total += detection_count
        if detection_count:
            left, top, right, bottom = expanded_union_box(boxes, width, height, args.margin)
            crop_source = "yolo"
            detection_confidence = float(result.boxes.conf.max().item())
        else:
            left, top, right, bottom = 0, 0, width, height
            crop_source = "full_image_fallback"
            detection_confidence = 0.0
            fallbacks += 1
        rgb = np.ascontiguousarray(result.orig_img[..., ::-1])
        crop = Image.fromarray(rgb).crop((left, top, right, bottom))
        record: dict[str, object] = {
            "domain": domain,
            "image": name,
            "actual_id": int(metadata["Index"]),
            "actual_class": f"{metadata['Plant']} / {metadata['Disease']}",
            "crop_source": crop_source,
            "detection_count": detection_count,
            "detection_confidence": f"{detection_confidence:.8f}",
            "crop_box": f"{left},{top},{right},{bottom}",
        }
        tensors.append(transform(crop))
        pending_records.append(record)
        records.append(record)
        if len(tensors) == args.batch_size:
            classify_pending(tensors, pending_records, models, raw, device)
            tensors.clear()
            pending_records.clear()
        progress.update(1)
    progress.close()
    classify_pending(tensors, pending_records, models, raw, device)
    missing = set(rows_by_name) - seen
    if missing:
        raise RuntimeError(f"YOLO did not return {len(missing)} images for {domain}")

    metrics: dict[str, dict[str, float | int]] = {}
    per_class_by_model: dict[str, list[dict[str, object]]] = {}
    for name, values in raw.items():
        targets = values["targets"]
        predictions = values["predictions"]
        confidences = values["confidences"]
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=shared_ids, zero_division=0
        )
        per_class_by_model[name] = [
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for class_id, class_precision, class_recall, class_f1, class_support in zip(
                shared_ids, precision, recall, f1, support, strict=True
            )
        ]
        metrics[name] = {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(
                f1_score(targets, predictions, labels=shared_ids, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(
                recall_score(
                    targets, predictions, labels=shared_ids, average="macro", zero_division=0
                )
            ),
            "mean_confidence": float(np.mean(confidences)),
            "expected_calibration_error": expected_calibration_error(
                targets, predictions, confidences, args.calibration_bins
            ),
        }
    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "images": len(rows),
        "seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "full_image_fallbacks": fallbacks,
        "fallback_rate": fallbacks / len(rows),
        "mean_detections_per_image": detection_total / len(rows),
        "classification": metrics,
    }
    return summary, records, per_class_by_model


def save_confusion_matrices(
    output_dir: Path,
    domain: str,
    records: list[dict[str, object]],
    model_names: list[str],
    num_classes: int,
) -> None:
    targets = [int(record["actual_id"]) for record in records]
    labels = list(range(num_classes))
    for name in model_names:
        predictions = [int(record[f"{name}_prediction_id"]) for record in records]
        matrix = confusion_matrix(targets, predictions, labels=labels)
        path = output_dir / f"{domain}_{name}_confusion_matrix.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual_id/predicted_id", *labels])
            writer.writerows([class_id, *row] for class_id, row in zip(labels, matrix, strict=True))


def save_plot(summary: dict[str, object], output_path: Path, model_names: list[str]) -> None:
    domains = ("plantseg_overlap", "plantvillage")
    metrics = ("accuracy", "macro_f1", "balanced_accuracy")
    labels = ("Accuracy", "Macro F1", "Balanced accuracy")
    figure, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=True)
    x = np.arange(len(model_names))
    width = 0.36
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        for offset, domain in zip((-width / 2, width / 2), domains, strict=True):
            values = [
                summary["domains"][domain]["classification"][model][metric] for model in model_names
            ]
            axis.bar(x + offset, values, width, label=domain.replace("_", " "))
        axis.set_title(label)
        axis.set_xticks(x, [name.replace("_", "\n") for name in model_names])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right")
    figure.suptitle("Matched-class PlantSeg vs PlantVillage performance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Lesion YOLO checkpoint not found: {args.yolo_checkpoint}; run make train-yolo-lesion"
        )
    if args.batch_size < 1 or args.yolo_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if not 0 <= args.confidence <= 1 or args.margin < 0:
        raise ValueError("confidence must be in 0..1 and margin cannot be negative")

    mapping = config["plantseg_to_plantvillage"]
    shared_ids = sorted(int(details["class_id"]) for details in mapping.values())
    if len(shared_ids) != len(set(shared_ids)):
        raise ValueError("plantseg_to_plantvillage contains duplicate PlantSeg class IDs")
    metadata = RAW_DIR / "PlantSeg" / "plantseg" / "Metadata.csv"
    class_names = plantseg_class_names(metadata, config["classification"]["num_classes"])
    for shared_name, details in mapping.items():
        class_id = int(details["class_id"])
        if class_names[class_id] != shared_name:
            raise ValueError(
                f"Mapped class {class_id} is {class_names[class_id]!r}, not {shared_name!r}"
            )
    device = choose_device(args.device)
    models, checkpoints = load_classifiers(args.models, args.classifier_dir, class_names, device)
    detector = YOLO(args.yolo_checkpoint)
    lesion_names = [detector.names[index] for index in range(len(detector.names))]
    if lesion_names != ["lesion"]:
        raise RuntimeError(f"Crop detector must contain only 'lesion', found {lesion_names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "plantseg_overlap": TESTS_DIR / "overlap" / "PlantSeg",
        "plantvillage": TESTS_DIR / "overlap" / "PlantVillage",
    }
    domain_summaries = {}
    all_records: list[dict[str, object]] = []
    all_per_class = {}
    for domain, root in roots.items():
        domain_summary, records, per_class = evaluate_domain(
            domain,
            root,
            detector,
            models,
            class_names,
            shared_ids,
            args,
            config,
            device,
        )
        domain_summaries[domain] = domain_summary
        all_records.extend(records)
        all_per_class[domain] = per_class
        save_confusion_matrices(args.output_dir, domain, records, list(models), len(class_names))

    shifts = {}
    for name in models:
        shifts[name] = {}
        for metric in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "mean_confidence",
            "expected_calibration_error",
        ):
            source = domain_summaries["plantseg_overlap"]["classification"][name][metric]
            target = domain_summaries["plantvillage"]["classification"][name][metric]
            shifts[name][f"{metric}_change"] = target - source

    summary = {
        "experiment": "PlantSeg-to-PlantVillage matched-class cross-domain evaluation",
        "shared_classes": len(shared_ids),
        "shared_class_ids": shared_ids,
        "yolo_checkpoint": str(args.yolo_checkpoint.resolve()),
        "classifier_checkpoints": checkpoints,
        "detection_confidence_threshold": args.confidence,
        "crop_margin": args.margin,
        "calibration_bins": args.calibration_bins,
        "domains": domain_summaries,
        "plantvillage_minus_plantseg": shifts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "per_class.json").write_text(
        json.dumps(all_per_class, indent=2) + "\n", encoding="utf-8"
    )
    for record in all_records:
        for name in models:
            predicted_id = int(record[f"{name}_prediction_id"])
            record[f"{name}_prediction_class"] = class_names[predicted_id]
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    save_plot(summary, args.output_dir / "domain_comparison.png", list(models))

    print(f"Cross-domain outputs saved to {args.output_dir.resolve()}")
    for name in models:
        source = domain_summaries["plantseg_overlap"]["classification"][name]
        target = domain_summaries["plantvillage"]["classification"][name]
        print(
            f"{name}: PlantSeg accuracy={source['accuracy']:.3f}, "
            f"PlantVillage accuracy={target['accuracy']:.3f}, "
            f"change={target['accuracy'] - source['accuracy']:+.3f}"
        )


if __name__ == "__main__":
    main()
