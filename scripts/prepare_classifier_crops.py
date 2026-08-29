#!/usr/bin/env python3
"""Create PlantSeg classifier train/validation crops using the trained YOLO detector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from plant_disease.paths import OUTPUTS_DIR, PROJECT_ROOT, RAW_DIR, TRAINING_DIR

SPLITS = {"Training": "train", "Validation": "val"}


def parse_args(config: dict) -> argparse.Namespace:
    detection = config["detection"]
    default_checkpoint = OUTPUTS_DIR / "yolo" / detection["run_name"] / "weights" / "best.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--confidence", type=float, default=detection["crop_confidence"])
    parser.add_argument("--margin", type=float, default=detection["crop_margin"])
    parser.add_argument("--image-size", type=int, default=detection["image_size"])
    parser.add_argument("--batch-size", type=int, default=detection["batch_size"])
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA device ID")
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DIR / "PlantSegCrops")
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        help="Limit each split for smoke testing.",
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


def prepare_split(
    model: YOLO,
    image_dir: Path,
    destination: Path,
    rows_by_name: dict[str, dict[str, str]],
    confidence: float,
    margin: float,
    image_size: int,
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, str]], int]:
    destination.mkdir(parents=True, exist_ok=True)
    prepared_rows = []
    fallback_count = 0
    expected = set(rows_by_name)

    # A path list is treated as one in-memory batch by Ultralytics. A directory
    # source honors batch_size and keeps full-dataset crop generation bounded.
    results = model.predict(
        source=str(image_dir),
        conf=confidence,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        stream=True,
        verbose=False,
    )
    for result in results:
        name = Path(result.path).name
        if name not in rows_by_name:
            continue
        height, width = result.orig_shape
        boxes = (
            result.boxes.xyxy.detach().cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 4))
        )
        if len(boxes):
            left, top, right, bottom = expanded_union_box(boxes, width, height, margin)
            crop_source = "yolo"
            max_confidence = float(result.boxes.conf.max().item())
        else:
            left, top, right, bottom = 0, 0, width, height
            crop_source = "full_image_fallback"
            max_confidence = 0.0
            fallback_count += 1

        rgb = np.ascontiguousarray(result.orig_img[..., ::-1])
        crop = Image.fromarray(rgb).crop((left, top, right, bottom))
        crop.save(destination / name)
        row = dict(rows_by_name[name])
        row.update(
            {
                "Crop source": crop_source,
                "Detection confidence": f"{max_confidence:.8f}",
                "Crop box": f"{left},{top},{right},{bottom}",
            }
        )
        prepared_rows.append(row)
        if len(prepared_rows) == len(expected):
            break

    prepared_names = {row["Name"] for row in prepared_rows}
    missing = expected - prepared_names
    if missing:
        raise RuntimeError(
            f"YOLO did not return {len(missing)} expected images; first: {min(missing)}"
        )
    for existing in destination.iterdir():
        if existing.is_file() and existing.name not in expected:
            existing.unlink()
    return prepared_rows, fallback_count


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    args = parse_args(config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Trained YOLO checkpoint not found: {args.checkpoint}; run make train-yolo first"
        )
    if not 0 <= args.confidence <= 1 or args.margin < 0:
        raise ValueError("confidence must be in 0..1 and margin cannot be negative")

    plantseg = RAW_DIR / "PlantSeg" / "plantseg"
    metadata_path = plantseg / "Metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    model = YOLO(args.checkpoint)
    destination = args.output_dir
    all_rows = []
    total_fallbacks = 0
    device = yolo_device(args.device)
    for metadata_split, split in SPLITS.items():
        selected_rows = sorted(
            (row for row in rows if row["Split"] == metadata_split),
            key=lambda row: row["Name"],
        )
        if args.max_images_per_split is not None:
            if args.max_images_per_split < 1:
                raise ValueError("max-images-per-split must be positive")
            selected_rows = selected_rows[: args.max_images_per_split]
        split_rows = {row["Name"]: row for row in selected_rows}
        prepared, fallbacks = prepare_split(
            model,
            plantseg / "images" / split,
            destination / "images" / split,
            split_rows,
            args.confidence,
            args.margin,
            args.image_size,
            args.batch_size,
            device,
        )
        all_rows.extend(prepared)
        total_fallbacks += fallbacks
        print(f"Prepared {len(prepared):,} {split} crops ({fallbacks:,} full-image fallbacks)")

    output_fields = [*fieldnames, "Crop source", "Detection confidence", "Crop box"]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "Metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(
        f"Classifier crops ready at {destination}: {len(all_rows):,} images, "
        f"{total_fallbacks:,} full-image fallbacks"
    )


if __name__ == "__main__":
    main()
