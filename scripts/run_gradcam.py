#!/usr/bin/env python3
"""Compare classifier Grad-CAM activation with PlantSeg lesion masks.

For each PlantSeg test image, the class-agnostic lesion YOLO detector supplies the
same crop used by the classification pipeline (falling back to the full image when
it finds nothing). Grad-CAM is computed on that crop for the model's predicted
class, then pasted back into full-image coordinates so it can be compared directly
with the authoritative binary lesion mask.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from ultralytics import YOLO

from plant_disease.data import plantseg_class_names
from plant_disease.models import build_classifier, build_gradcam, classification_transform
from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TESTS_DIR

CAM_THRESHOLD = 0.5
CAM_TOP_QUANTILE = 0.20


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "efficientnet_b0", "mobilenet_v3_large"),
        default="mobilenet_v3_large",
        help="Classifier to explain (default: mobilenet_v3_large, best validation macro-F1).",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        help="Override path to the classifier checkpoint (default: outputs/classification/<model>/best.pt).",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=(
            OUTPUTS_DIR / "yolo" / detection["run_name"] / "weights" / "best.pt"
        ),
        help="Class-agnostic lesion detector used to produce the classifier crop.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--confidence", type=float, default=detection["crop_confidence"])
    parser.add_argument("--margin", type=float, default=detection["crop_margin"])
    parser.add_argument("--max-images", type=int, help="Limit the run for a quick check.")
    parser.add_argument(
        "--qualitative-count",
        type=int,
        default=12,
        help="Number of side-by-side image/mask/Grad-CAM figures to save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "gradcam" / "plantseg_test",
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


def load_classifier(name: str, checkpoint_path: Path, class_names: list[str], device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != name or checkpoint["num_classes"] != len(class_names):
        raise RuntimeError(f"Classifier checkpoint metadata is invalid: {checkpoint_path}")
    if checkpoint["class_names"] != class_names:
        raise RuntimeError(f"Classifier taxonomy differs from PlantSeg: {checkpoint_path}")
    model = build_classifier(name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    return model, checkpoint


def cam_to_full_image(
    cam_crop: np.ndarray, box: tuple[int, int, int, int], height: int, width: int
) -> np.ndarray:
    """Paste a crop-sized Grad-CAM map into a full-image-sized zero canvas."""
    left, top, right, bottom = box
    canvas = np.zeros((height, width), dtype=np.float32)
    resized = np.asarray(
        Image.fromarray(cam_crop).resize((right - left, bottom - top), Image.BILINEAR)
    )
    canvas[top:bottom, left:right] = resized
    return canvas


def cam_mask_metrics(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Threshold-based overlap plus threshold-free CAM mass concentration."""
    mask_bool = mask > 0
    mask_area = float(mask_bool.sum())
    total_pixels = float(mask.size)

    metrics: dict[str, float] = {}
    for label, threshold in (("fixed", CAM_THRESHOLD), ("top_quantile", None)):
        if threshold is None:
            cutoff = float(np.quantile(cam, 1.0 - CAM_TOP_QUANTILE))
        else:
            cutoff = threshold
        active = cam >= cutoff
        active_area = float(active.sum())
        intersection = float(np.logical_and(active, mask_bool).sum())
        union = float(np.logical_or(active, mask_bool).sum())
        metrics[f"iou_{label}"] = intersection / union if union > 0 else 0.0
        metrics[f"precision_{label}"] = intersection / active_area if active_area > 0 else 0.0
        metrics[f"recall_{label}"] = intersection / mask_area if mask_area > 0 else 0.0

    cam_sum = float(cam.sum())
    metrics["mask_energy_fraction"] = (
        float(cam[mask_bool].sum()) / cam_sum if cam_sum > 0 else 0.0
    )
    metrics["mask_area_fraction"] = mask_area / total_pixels if total_pixels > 0 else 0.0
    peak_index = int(np.argmax(cam))
    metrics["pointing_game_hit"] = float(mask_bool.flat[peak_index])
    return metrics


def save_qualitative_figure(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    cam: np.ndarray,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap="Reds", alpha=0.45)
    axes[1].set_title("PlantSeg mask")
    axes[2].imshow(image)
    axes[2].imshow(cam, cmap="jet", alpha=0.45)
    axes[2].set_title("Grad-CAM")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if not 0 <= args.confidence <= 1 or args.margin < 0:
        raise ValueError("confidence must be in 0..1 and margin cannot be negative")

    checkpoint_path = args.classifier_checkpoint or (
        OUTPUTS_DIR / "classification" / args.model / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}; run make train-classifiers"
        )
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"Lesion YOLO checkpoint not found: {args.yolo_checkpoint}; run make train-yolo-lesion"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    metadata_path = RAW_DIR / "PlantSeg" / "plantseg" / "Metadata.csv"
    class_names = plantseg_class_names(metadata_path, config["classification"]["num_classes"])
    classifier, checkpoint_info = load_classifier(args.model, checkpoint_path, class_names, device)
    detector = YOLO(args.yolo_checkpoint)
    lesion_names = [detector.names[index] for index in range(len(detector.names))]
    if lesion_names != ["lesion"]:
        raise RuntimeError(f"Crop detector must contain only the 'lesion' class, found: {lesion_names}")

    test_root = TESTS_DIR / "PlantSeg" / "full"
    with (test_root / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["Name"])
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("max-images must be positive")
        rows = rows[: args.max_images]

    transform = classification_transform(config["project"]["image_size"])
    gradcam = build_gradcam(args.model, classifier)

    per_image_rows: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = {}
    saved_figures = 0

    progress = tqdm(rows, desc=f"Grad-CAM ({args.model})", unit="image")
    for row in progress:
        image_name = row["Name"]
        image_path = test_root / "images" / image_name
        mask_path = test_root / "masks" / row["Label file"]
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            mask = (np.asarray(handle) > 0).astype(np.uint8)
        height, width = image.shape[:2]

        results = detector.predict(
            source=image_path,
            conf=args.confidence,
            imgsz=config["detection"]["image_size"],
            device=yolo_device(device),
            verbose=False,
        )
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        if len(boxes):
            box = expanded_union_box(boxes, width, height, args.margin)
            crop_source = "yolo"
        else:
            box = (0, 0, width, height)
            crop_source = "full_image_fallback"
        left, top, right, bottom = box
        crop = Image.fromarray(image[top:bottom, left:right])
        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = classifier(tensor).softmax(dim=1)
        predicted_id = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_id].item())

        cam_crop = gradcam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(predicted_id)],
        )[0]
        cam_full = cam_to_full_image(cam_crop, box, height, width)

        metrics = cam_mask_metrics(cam_full, mask)
        for key, value in metrics.items():
            aggregate.setdefault(key, []).append(value)

        record = {
            "image": image_name,
            "actual_id": int(row["Index"]),
            "actual_class": f'{row["Plant"]} / {row["Disease"]}',
            "predicted_id": predicted_id,
            "predicted_class": class_names[predicted_id],
            "confidence": f"{confidence:.8f}",
            "crop_source": crop_source,
            "crop_box": f"{left},{top},{right},{bottom}",
            **{key: f"{value:.8f}" for key, value in metrics.items()},
        }
        per_image_rows.append(record)

        if saved_figures < args.qualitative_count:
            correct = predicted_id == int(row["Index"])
            title = (
                f"{image_name} | actual={record['actual_class']} | "
                f"predicted={record['predicted_class']} ({'correct' if correct else 'wrong'}) | "
                f"IoU(top-20%)={metrics['iou_top_quantile']:.2f}"
            )
            save_qualitative_figure(
                figures_dir / f"{Path(image_name).stem}.png",
                image,
                mask,
                cam_full,
                title,
            )
            saved_figures += 1

    fieldnames = list(per_image_rows[0])
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary = {
        "dataset": "PlantSeg official test split",
        "model": args.model,
        "classifier_checkpoint": str(checkpoint_path.resolve()),
        "classifier_validation_macro_f1": float(checkpoint_info["best_macro_f1"]),
        "yolo_checkpoint": str(args.yolo_checkpoint.resolve()),
        "images": len(per_image_rows),
        "cam_threshold_fixed": CAM_THRESHOLD,
        "cam_top_quantile": CAM_TOP_QUANTILE,
        "metrics_mean": {key: float(np.mean(values)) for key, values in aggregate.items()},
        "metrics_median": {key: float(np.median(values)) for key, values in aggregate.items()},
        "qualitative_figures": saved_figures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Grad-CAM vs PlantSeg masks ({args.model}, {len(per_image_rows)} images): "
        f"IoU(top-20%)={summary['metrics_mean']['iou_top_quantile']:.3f} "
        f"mask-energy-fraction={summary['metrics_mean']['mask_energy_fraction']:.3f} "
        f"pointing-game={summary['metrics_mean']['pointing_game_hit']:.3f}"
    )
    print(f"Saved evaluation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
