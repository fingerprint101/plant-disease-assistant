#!/usr/bin/env python3
"""Run basic integrity checks on the two core project datasets."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from plant_disease.paths import RAW_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def decode_image(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"Invalid image dimensions in {path}")
        return {"path": str(path), "size": [image.width, image.height], "mode": image.mode}


def check_plantvillage(raw_dir: Path) -> dict[str, object]:
    color_dir = raw_dir / "PlantVillage" / "raw" / "color"
    class_dirs = sorted(path for path in color_dir.glob("*") if path.is_dir())
    if len(class_dirs) != 38:
        raise RuntimeError(f"PlantVillage has {len(class_dirs)} color classes; expected 38")

    per_class = {path.name: len(image_files(path)) for path in class_dirs}
    if any(count == 0 for count in per_class.values()):
        raise RuntimeError("At least one PlantVillage class is empty")
    total = sum(per_class.values())
    if total != 54_305:
        raise RuntimeError(f"PlantVillage has {total} color images; expected 54,305")

    samples = [decode_image(image_files(path)[0]) for path in class_dirs]
    return {
        "status": "ok",
        "classes": len(class_dirs),
        "images": total,
        "smallest_class": min(per_class.values()),
        "largest_class": max(per_class.values()),
        "decoded_samples": len(samples),
    }


def validate_yolo_label(path: Path, num_classes: int) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Invalid YOLO row at {path}:{line_number}")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if not 0 <= class_id < num_classes or not all(0.0 <= value <= 1.0 for value in coordinates):
            raise RuntimeError(f"Out-of-range YOLO row at {path}:{line_number}")
        boxes += 1
    return boxes


def check_plantdoc(raw_dir: Path) -> dict[str, object]:
    root = raw_dir / "PlantDoc"
    records = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    classes = (root / "classes.txt").read_text(encoding="utf-8").splitlines()
    extraction = json.loads((root / "extraction_report.json").read_text(encoding="utf-8"))
    expected_images = extraction["statistics"]["images_written"]
    if len(classes) != 29:
        raise RuntimeError(f"PlantDoc has {len(classes)} classes; expected 29")
    if len(records) != expected_images:
        raise RuntimeError(
            f"PlantDoc metadata has {len(records)} records; report says {expected_images}"
        )

    missing = []
    box_count = 0
    for record in records:
        image_path = root / record["image"]
        label_path = root / record["label_file"]
        if not image_path.is_file():
            missing.append(str(image_path))
        if not label_path.is_file():
            missing.append(str(label_path))
        if label_path.is_file():
            box_count += validate_yolo_label(label_path, len(classes))
    if missing:
        raise RuntimeError(f"PlantDoc has {len(missing)} missing files; first: {missing[0]}")
    if box_count != extraction["statistics"]["boxes_written"]:
        raise RuntimeError(
            f"PlantDoc labels contain {box_count} boxes; report says "
            f"{extraction['statistics']['boxes_written']}"
        )

    sample_indices = sorted({0, len(records) // 2, len(records) - 1})
    for index in sample_indices:
        decode_image(root / records[index]["image"])
    return {
        "status": "ok",
        "classes": len(classes),
        "images": len(records),
        "boxes": box_count,
        "decoded_samples": len(sample_indices),
    }


def main() -> None:
    results = {
        "plantvillage": check_plantvillage(RAW_DIR),
        "plantdoc": check_plantdoc(RAW_DIR),
    }
    for name, result in results.items():
        print(
            f"[ok] {name}: {result['images']} images, {result['classes']} classes",
            flush=True,
        )
    print(f"All {len(results)} datasets passed")


if __name__ == "__main__":
    main()
