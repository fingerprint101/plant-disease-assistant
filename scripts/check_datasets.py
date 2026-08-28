#!/usr/bin/env python3
"""Run basic integrity checks on all project datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from plant_disease.paths import RAW_DIR, TESTS_DIR, TRAINING_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def check_plantseg(raw_dir: Path) -> dict[str, object]:
    root = raw_dir / "PlantSeg" / "plantseg"
    metadata_path = root / "Metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7_774:
        raise RuntimeError(f"PlantSeg has {len(rows)} metadata rows; expected 7,774")
    classes = {(row["Plant"], row["Disease"]) for row in rows}
    if len(classes) != 115:
        raise RuntimeError(f"PlantSeg has {len(classes)} classes; expected 115")

    split_names = {"Training": "train", "Validation": "val", "Test": "test"}
    missing = []
    for row in rows:
        split = split_names[row["Split"]]
        image_path = root / "images" / split / row["Name"]
        mask_path = root / "annotations" / split / row["Label file"]
        if not image_path.is_file():
            missing.append(str(image_path))
        if not mask_path.is_file():
            missing.append(str(mask_path))
    if missing:
        raise RuntimeError(f"PlantSeg has {len(missing)} missing files; first: {missing[0]}")

    sample_indices = sorted({0, len(rows) // 2, len(rows) - 1})
    for index in sample_indices:
        row = rows[index]
        split = split_names[row["Split"]]
        image = decode_image(root / "images" / split / row["Name"])
        mask = decode_image(root / "annotations" / split / row["Label file"])
        if image["size"] != mask["size"]:
            raise RuntimeError(f"PlantSeg image/mask size mismatch for {row['Name']}")
    return {
        "status": "ok",
        "classes": len(classes),
        "images": len(rows),
        "masks": len(rows),
        "decoded_samples": len(sample_indices),
    }


def check_plantseg_training() -> None:
    root = TRAINING_DIR / "PlantSeg"
    with (root / "Metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5_367 or any(row["Split"] != "Training" for row in rows):
        raise RuntimeError("PlantSeg training view does not contain the official training split")
    if len(image_files(root / "images")) != len(rows):
        raise RuntimeError("PlantSeg training view image count does not match its metadata")
    if len(image_files(root / "masks")) != len(rows):
        raise RuntimeError("PlantSeg training view mask count does not match its metadata")
    if not (root / "annotations.json").is_file():
        raise RuntimeError("PlantSeg training view is missing annotations.json")
    classes = {(row["Plant"], row["Disease"]) for row in rows}
    if len(classes) != 115:
        raise RuntimeError(f"PlantSeg training view has {len(classes)} classes; expected 115")
    print(f"[ok] plantseg training view: {len(rows)} images, {len(classes)} classes")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_test_datasets() -> None:
    full = TESTS_DIR / "PlantSeg" / "full"
    full_rows = csv_rows(full / "Metadata.csv")
    if len(full_rows) != 1_561 or len(image_files(full / "images")) != len(full_rows):
        raise RuntimeError("Full PlantSeg test view is incomplete")
    if len(image_files(full / "masks")) != len(full_rows):
        raise RuntimeError("Full PlantSeg test mask view is incomplete")

    ps_overlap = TESTS_DIR / "overlap" / "PlantSeg"
    pv_overlap = TESTS_DIR / "overlap" / "PlantVillage"
    ps_rows = csv_rows(ps_overlap / "Metadata.csv")
    pv_rows = csv_rows(pv_overlap / "Metadata.csv")
    if not ps_rows or len(image_files(ps_overlap / "images")) != len(ps_rows):
        raise RuntimeError("PlantSeg overlap test view is incomplete")
    if len(image_files(ps_overlap / "masks")) != len(ps_rows):
        raise RuntimeError("PlantSeg overlap test masks are incomplete")
    if not pv_rows or len(image_files(pv_overlap / "images")) != len(pv_rows):
        raise RuntimeError("PlantVillage overlap test view is incomplete")
    if {row["Index"] for row in ps_rows} != {row["Index"] for row in pv_rows}:
        raise RuntimeError("Overlap test views do not contain the same class IDs")

    variants = csv_rows(TESTS_DIR / "robustness" / "PlantSeg" / "variants.csv")
    if len(variants) != 30:
        raise RuntimeError(f"Robustness view has {len(variants)} variants; expected 30")
    print(
        f"[ok] test views: {len(full_rows)} full PlantSeg images, "
        f"{len(ps_rows)}/{len(pv_rows)} overlap images, {len(variants)} robustness variants"
    )


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


def check_plantseg_detection() -> None:
    root = TRAINING_DIR / "PlantSegDetection"
    expected = {"train": 5_367, "val": 846}
    for split, expected_count in expected.items():
        images = image_files(root / "images" / split)
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if len(images) != expected_count or len(labels) != expected_count:
            raise RuntimeError(
                f"PlantSeg detection {split} view is incomplete: "
                f"{len(images)} images / {len(labels)} labels"
            )
        if sum(validate_yolo_label(path, 1) for path in labels) != expected_count:
            raise RuntimeError(f"PlantSeg detection {split} must have one lesion box per image")
    print("[ok] plantseg detection view: 5,367 train / 846 validation lesion boxes")


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
    check_plantseg_training()
    check_plantseg_detection()
    check_test_datasets()
    results = {
        "plantseg (primary)": check_plantseg(RAW_DIR),
        "plantvillage (secondary)": check_plantvillage(RAW_DIR),
        "plantdoc (backup)": check_plantdoc(RAW_DIR),
    }
    for name, result in results.items():
        print(
            f"[ok] {name}: {result['images']} images, {result['classes']} classes",
            flush=True,
        )
    print(f"All {len(results)} datasets passed")


if __name__ == "__main__":
    main()
