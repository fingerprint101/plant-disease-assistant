#!/usr/bin/env python3
"""Compare standalone disease-aware YOLO with the two-stage PlantSeg pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
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

from plant_disease.data import plantseg_class_names
from plant_disease.models import CLASSIFICATION_MODELS, build_classifier, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TESTS_DIR


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    classification = config["classification"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=(
            OUTPUTS_DIR
            / "yolo"
            / detection["run_name"]
            / "weights"
            / "best.pt"
        ),
        help="Class-agnostic lesion detector used by the crop pipeline.",
    )
    parser.add_argument(
        "--standalone-yolo-checkpoint",
        type=Path,
        default=(
            OUTPUTS_DIR
            / "yolo"
            / detection["standalone_run_name"]
            / "weights"
            / "best.pt"
        ),
        help="Disease-aware YOLO checkpoint used for standalone detection and classification.",
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=OUTPUTS_DIR / "classification",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "evaluation" / "plantseg_test",
    )
    parser.add_argument("--max-images", type=int, help="Limit each evaluation for a smoke test.")
    parser.add_argument(
        "--skip-localization",
        action="store_true",
        help="Skip box-level validation for both YOLO models during a quick smoke test.",
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
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    margin: float,
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


def load_classifiers(
    names: list[str],
    classifier_dir: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict]]:
    models = {}
    details = {}
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
        details[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "epoch": int(checkpoint["epoch"]),
            "validation_macro_f1": float(checkpoint["best_macro_f1"]),
        }
        print(f"Loaded {name} from epoch {checkpoint['epoch']}")
    return models, details


def localization_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    smoke_name: str,
) -> tuple[Path, Path | None]:
    dataset_path = TESTS_DIR / "PlantSeg" / dataset_name / "dataset.yaml"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Detection test data not found: {dataset_path}; run make prepare-data"
        )
    if args.max_images is None:
        return dataset_path, None

    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    images = sorted(
        path
        for path in (root / dataset["test"]).iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
    )[: args.max_images]
    subset_dir = args.output_dir / f".{smoke_name}_smoke"
    subset_dir.mkdir(parents=True, exist_ok=True)
    image_list = subset_dir / "test.txt"
    image_list.write_text(
        "".join(f"{path.absolute()}\n" for path in images),
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        dataset[split] = str(image_list.absolute())
    subset_yaml = subset_dir / "dataset.yaml"
    subset_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    cache_path = (root / "labels" / "test").with_suffix(".cache")
    return subset_yaml, cache_path


def evaluate_localization(
    detector: YOLO,
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    *,
    dataset_name: str,
    output_name: str,
    description: str,
) -> dict[str, object]:
    dataset, smoke_cache = localization_dataset(args, dataset_name, output_name)
    metrics = detector.val(
        data=str(dataset.resolve()),
        split="test",
        imgsz=config["detection"]["image_size"],
        batch=args.yolo_batch_size,
        device=yolo_device(device),
        project=str(args.output_dir),
        name=output_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    if smoke_cache is not None and smoke_cache.is_file():
        smoke_cache.unlink()
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    print(
        f"{description} test: "
        f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
        f"mAP50={result['map50']:.3f} mAP50-95={result['map50_95']:.3f}"
    )
    return result


def classify_batch(
    tensors: list[torch.Tensor],
    records: list[dict[str, object]],
    models: dict[str, torch.nn.Module],
    outputs: dict[str, dict[str, list]],
    device: torch.device,
) -> None:
    batch = torch.stack(tensors).to(device)
    targets = [int(record["actual_id"]) for record in records]
    with torch.inference_mode():
        for name, model in models.items():
            probabilities = model(batch).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predicted_ids = prediction.cpu().tolist()
            confidences = confidence.cpu().tolist()
            outputs[name]["targets"].extend(targets)
            outputs[name]["predictions"].extend(predicted_ids)
            outputs[name]["confidences"].extend(confidences)
            for record, class_id, score in zip(
                records,
                predicted_ids,
                confidences,
                strict=True,
            ):
                record[f"{name}_prediction_id"] = class_id
                record[f"{name}_confidence"] = f"{score:.8f}"


def evaluate_classification(
    detector: YOLO,
    models: dict[str, torch.nn.Module],
    class_names: list[str],
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, dict[str, list]]]:
    test_root = TESTS_DIR / "PlantSeg" / "full"
    with (test_root / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["Name"])
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("max-images must be positive")
        rows = rows[: args.max_images]
    rows_by_name = {row["Name"]: row for row in rows}
    expected = set(rows_by_name)
    transform = classification_transform(config["project"]["image_size"])
    outputs = {
        name: {"targets": [], "predictions": [], "confidences": []} for name in models
    }
    prediction_rows: list[dict[str, object]] = []
    pending_tensors = []
    pending_records = []
    fallback_count = 0
    total_detections = 0
    started = time.perf_counter()

    # Directory streaming keeps memory bounded; a list is treated as one batch by Ultralytics.
    results = detector.predict(
        source=str(test_root / "images"),
        conf=args.confidence,
        imgsz=config["detection"]["image_size"],
        batch=args.yolo_batch_size,
        device=yolo_device(device),
        stream=True,
        verbose=False,
    )
    progress = tqdm(total=len(rows), desc="PlantSeg test", unit="image")
    for result in results:
        name = Path(result.path).name
        if name not in rows_by_name:
            continue
        metadata = rows_by_name[name]
        height, width = result.orig_shape
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        detection_count = len(boxes)
        total_detections += detection_count
        if detection_count:
            left, top, right, bottom = expanded_union_box(boxes, width, height, args.margin)
            crop_source = "yolo"
            detection_confidence = float(result.boxes.conf.max().item())
        else:
            left, top, right, bottom = 0, 0, width, height
            crop_source = "full_image_fallback"
            detection_confidence = 0.0
            fallback_count += 1

        rgb = np.ascontiguousarray(result.orig_img[..., ::-1])
        crop = Image.fromarray(rgb).crop((left, top, right, bottom))
        record: dict[str, object] = {
            "image": name,
            "actual_id": int(metadata["Index"]),
            "actual_class": f'{metadata["Plant"]} / {metadata["Disease"]}',
            "crop_source": crop_source,
            "detection_count": detection_count,
            "detection_confidence": f"{detection_confidence:.8f}",
            "crop_box": f"{left},{top},{right},{bottom}",
        }
        pending_tensors.append(transform(crop))
        pending_records.append(record)
        prediction_rows.append(record)
        progress.update()
        if len(pending_tensors) == args.batch_size:
            classify_batch(pending_tensors, pending_records, models, outputs, device)
            pending_tensors.clear()
            pending_records.clear()
        if len(prediction_rows) == len(expected):
            break
    progress.close()

    if pending_tensors:
        classify_batch(pending_tensors, pending_records, models, outputs, device)
    found = {str(row["image"]) for row in prediction_rows}
    if found != expected:
        missing = expected - found
        raise RuntimeError(f"YOLO did not return {len(missing)} test images; first: {min(missing)}")

    elapsed = time.perf_counter() - started
    pipeline = {
        "images": len(prediction_rows),
        "seconds": elapsed,
        "images_per_second": len(prediction_rows) / elapsed,
        "detection_confidence_threshold": args.confidence,
        "crop_margin": args.margin,
        "full_image_fallbacks": fallback_count,
        "fallback_rate": fallback_count / len(prediction_rows),
        "mean_detections_per_image": total_detections / len(prediction_rows),
    }
    return pipeline, prediction_rows, outputs


def evaluate_standalone_yolo(
    detector: YOLO,
    prediction_rows: list[dict[str, object]],
    class_names: list[str],
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
) -> dict[str, object]:
    """Score one disease prediction per image from the highest-confidence YOLO box."""
    rows_by_name = {str(row["image"]): row for row in prediction_rows}
    expected = set(rows_by_name)
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    detected_count = 0
    total_detections = 0
    started = time.perf_counter()

    results = detector.predict(
        source=str(TESTS_DIR / "PlantSeg" / "full" / "images"),
        conf=args.confidence,
        imgsz=config["detection"]["image_size"],
        batch=args.yolo_batch_size,
        device=yolo_device(device),
        stream=True,
        verbose=False,
    )
    progress = tqdm(total=len(expected), desc="Standalone YOLO", unit="image")
    found: set[str] = set()
    for result in results:
        name = Path(result.path).name
        if name not in rows_by_name:
            continue
        row = rows_by_name[name]
        target = int(row["actual_id"])
        detection_count = len(result.boxes)
        total_detections += detection_count
        if detection_count:
            best = int(result.boxes.conf.argmax().item())
            predicted_id = int(result.boxes.cls[best].item())
            confidence = float(result.boxes.conf[best].item())
            box = result.boxes.xyxy[best].detach().cpu().tolist()
            box_text = ",".join(str(int(round(value))) for value in box)
            detected_count += 1
        else:
            predicted_id = -1
            confidence = 0.0
            box_text = ""

        targets.append(target)
        predictions.append(predicted_id)
        confidences.append(confidence)
        row["standalone_yolo_prediction_id"] = predicted_id
        row["standalone_yolo_prediction_class"] = (
            class_names[predicted_id] if predicted_id >= 0 else "<no_detection>"
        )
        row["standalone_yolo_confidence"] = f"{confidence:.8f}"
        row["standalone_yolo_detection_count"] = detection_count
        row["standalone_yolo_box"] = box_text
        found.add(name)
        progress.update()
        if found == expected:
            break
    progress.close()
    if found != expected:
        missing = expected - found
        raise RuntimeError(
            f"Standalone YOLO did not return {len(missing)} test images; first: {min(missing)}"
        )

    present = sorted(set(targets))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=present,
        zero_division=0,
    )
    per_class = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for class_id, class_precision, class_recall, class_f1, class_support in zip(
            present,
            precision,
            recall,
            f1,
            support,
            strict=True,
        )
    ]
    (args.output_dir / "standalone_yolo_per_class.json").write_text(
        json.dumps(per_class, indent=2) + "\n",
        encoding="utf-8",
    )

    matrix_labels = [*range(len(class_names)), -1]
    matrix = confusion_matrix(targets, predictions, labels=matrix_labels)
    with (args.output_dir / "standalone_yolo_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        displayed_labels = [*range(len(class_names)), "no_detection"]
        writer.writerow(["actual_id/predicted_id", *displayed_labels])
        writer.writerows(
            [displayed, *row.tolist()]
            for displayed, row in zip(displayed_labels, matrix, strict=True)
        )

    elapsed = time.perf_counter() - started
    detected_mask = np.asarray(predictions) >= 0
    correct = np.asarray(targets) == np.asarray(predictions)
    summary = {
        "images": len(targets),
        "seconds": elapsed,
        "images_per_second": len(targets) / elapsed,
        "accuracy": float(accuracy_score(targets, predictions)),
        "accuracy_when_detected": (
            float(correct[detected_mask].mean()) if detected_mask.any() else None
        ),
        "macro_f1": float(
            f1_score(targets, predictions, labels=present, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            recall_score(targets, predictions, labels=present, average="macro", zero_division=0)
        ),
        "coverage": detected_count / len(targets),
        "no_detections": len(targets) - detected_count,
        "mean_confidence": float(np.mean(confidences)),
        "mean_detections_per_image": total_detections / len(targets),
        "classes_present": len(present),
    }
    print(
        "standalone_yolo classification test: "
        f"accuracy={summary['accuracy']:.3f} macro-F1={summary['macro_f1']:.3f} "
        f"balanced-accuracy={summary['balanced_accuracy']:.3f} "
        f"coverage={summary['coverage']:.3f}"
    )
    return summary


def save_classification_results(
    output_dir: Path,
    class_names: list[str],
    prediction_rows: list[dict[str, object]],
    outputs: dict[str, dict[str, list]],
) -> dict[str, dict[str, float]]:
    summaries = {}
    all_class_ids = list(range(len(class_names)))
    for name, values in outputs.items():
        targets = values["targets"]
        predictions = values["predictions"]
        confidences = values["confidences"]
        present = sorted(set(targets))
        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            predictions,
            labels=present,
            zero_division=0,
        )
        per_class = [
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for class_id, class_precision, class_recall, class_f1, class_support in zip(
                present,
                precision,
                recall,
                f1,
                support,
                strict=True,
            )
        ]
        (output_dir / f"{name}_per_class.json").write_text(
            json.dumps(per_class, indent=2) + "\n",
            encoding="utf-8",
        )

        matrix = confusion_matrix(targets, predictions, labels=all_class_ids)
        with (output_dir / f"{name}_confusion_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual_id/predicted_id", *all_class_ids])
            writer.writerows(
                [class_id, *row.tolist()] for class_id, row in enumerate(matrix)
            )

        summaries[name] = {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(
                f1_score(targets, predictions, labels=present, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(
                recall_score(
                    targets,
                    predictions,
                    labels=present,
                    average="macro",
                    zero_division=0,
                )
            ),
            "mean_confidence": float(np.mean(confidences)),
            "classes_present": len(present),
        }
        print(
            f"{name} test: accuracy={summaries[name]['accuracy']:.3f} "
            f"macro-F1={summaries[name]['macro_f1']:.3f} "
            f"balanced-accuracy={summaries[name]['balanced_accuracy']:.3f}"
        )

    for row in prediction_rows:
        for name in outputs:
            predicted_id = int(row[f"{name}_prediction_id"])
            row[f"{name}_prediction_class"] = class_names[predicted_id]
    fieldnames = list(prediction_rows[0])
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    return summaries


def main() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    args = parse_args(config)
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Lesion YOLO checkpoint not found: {args.yolo_checkpoint}; "
            "run make train-yolo-lesion"
        )
    if not args.standalone_yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Standalone YOLO checkpoint not found: {args.standalone_yolo_checkpoint}; "
            "run make train-yolo-standalone"
        )
    if args.batch_size < 1 or args.yolo_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if not 0 <= args.confidence <= 1 or args.margin < 0:
        raise ValueError("confidence must be in 0..1 and margin cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    metadata = RAW_DIR / "PlantSeg" / "plantseg" / "Metadata.csv"
    class_names = plantseg_class_names(metadata, config["classification"]["num_classes"])
    models, checkpoints = load_classifiers(
        args.models,
        args.classifier_dir,
        class_names,
        device,
    )
    detector = YOLO(args.yolo_checkpoint)
    standalone_detector = YOLO(args.standalone_yolo_checkpoint)
    lesion_names = [detector.names[index] for index in range(len(detector.names))]
    if lesion_names != ["lesion"]:
        raise RuntimeError(
            f"Crop detector must contain only the 'lesion' class, found: {lesion_names}"
        )
    standalone_names = [
        standalone_detector.names[index] for index in range(len(standalone_detector.names))
    ]
    if standalone_names != class_names:
        raise RuntimeError("Standalone YOLO checkpoint taxonomy differs from PlantSeg")
    print(f"Evaluating on {device}; outputs: {args.output_dir.resolve()}")

    localization = None
    standalone_localization = None
    if not args.skip_localization:
        localization = evaluate_localization(
            detector,
            args,
            config,
            device,
            dataset_name="detection",
            output_name="yolo_localization",
            description="Lesion YOLO",
        )
        standalone_localization = evaluate_localization(
            standalone_detector,
            args,
            config,
            device,
            dataset_name="disease_detection",
            output_name="standalone_yolo_detection",
            description="Standalone disease YOLO",
        )
    pipeline, prediction_rows, raw_outputs = evaluate_classification(
        detector,
        models,
        class_names,
        args,
        config,
        device,
    )
    standalone_classification = evaluate_standalone_yolo(
        standalone_detector,
        prediction_rows,
        class_names,
        args,
        config,
        device,
    )
    classification = save_classification_results(
        args.output_dir,
        class_names,
        prediction_rows,
        raw_outputs,
    )
    summary = {
        "dataset": "PlantSeg official test split",
        "yolo_checkpoint": str(args.yolo_checkpoint.resolve()),
        "standalone_yolo_checkpoint": str(args.standalone_yolo_checkpoint.resolve()),
        "classifier_checkpoints": checkpoints,
        "localization": localization,
        "standalone_yolo": {
            "localization": standalone_localization,
            "classification": standalone_classification,
        },
        "pipeline": pipeline,
        "classification": classification,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Pipeline test complete: {pipeline['images']} images, "
        f"fallback rate={pipeline['fallback_rate']:.3f}"
    )
    print(f"Saved evaluation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
