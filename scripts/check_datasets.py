#!/usr/bin/env python3
"""Validate relationships between raw, prepared, and test dataset views."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from plant_disease.paths import PROJECT_ROOT, RAW_DIR, TESTS_DIR, TRAINING_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PLANTSEG_SPLITS = {"Training": "train", "Validation": "val", "Test": "test"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES
    )


def decode_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"Invalid image dimensions in {path}")
        return image.size


def check_plantseg() -> tuple[list[dict[str, str]], int]:
    root = RAW_DIR / "PlantSeg" / "plantseg"
    rows = csv_rows(root / "Metadata.csv")
    if not rows:
        raise RuntimeError("PlantSeg metadata is empty")
    unknown_splits = {row["Split"] for row in rows} - set(PLANTSEG_SPLITS)
    if unknown_splits:
        raise RuntimeError(f"PlantSeg contains unknown splits: {sorted(unknown_splits)}")

    missing = []
    for row in rows:
        split = PLANTSEG_SPLITS[row["Split"]]
        image_path = root / "images" / split / row["Name"]
        mask_path = root / "annotations" / split / row["Label file"]
        if not image_path.is_file():
            missing.append(image_path)
        if not mask_path.is_file():
            missing.append(mask_path)
    if missing:
        raise RuntimeError(f"PlantSeg has {len(missing)} missing files; first: {missing[0]}")

    sample_indices = sorted({0, len(rows) // 2, len(rows) - 1})
    for index in sample_indices:
        row = rows[index]
        split = PLANTSEG_SPLITS[row["Split"]]
        image_size = decode_image(root / "images" / split / row["Name"])
        mask_size = decode_image(root / "annotations" / split / row["Label file"])
        if image_size != mask_size:
            raise RuntimeError(f"PlantSeg image/mask size mismatch for {row['Name']}")

    classes = {(row["Plant"], row["Disease"]) for row in rows}
    print(f"[ok] plantseg raw: {len(rows):,} images, {len(classes)} classes")
    return rows, len(classes)


def check_plantseg_training(raw_rows: list[dict[str, str]], class_count: int) -> None:
    root = TRAINING_DIR / "PlantSeg"
    rows = csv_rows(root / "Metadata.csv")
    expected_rows = [row for row in raw_rows if row["Split"] == "Training"]
    if rows != expected_rows:
        raise RuntimeError("PlantSeg training metadata does not match the official training split")
    if len(image_files(root / "images")) != len(rows):
        raise RuntimeError("PlantSeg training image count does not match its metadata")
    if len(image_files(root / "masks")) != len(rows):
        raise RuntimeError("PlantSeg training mask count does not match its metadata")
    if not (root / "annotations.json").is_file():
        raise RuntimeError("PlantSeg training view is missing annotations.json")
    classes = {(row["Plant"], row["Disease"]) for row in rows}
    if len(classes) != class_count:
        raise RuntimeError("The PlantSeg training split does not cover every raw class")
    print(f"[ok] plantseg training view: {len(rows):,} images, {len(classes)} classes")


def validate_yolo_label(path: Path, num_classes: int) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Invalid YOLO row at {path}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, width, height = (float(value) for value in fields[1:])
        if (
            not 0 <= class_id < num_classes
            or not 0 <= center_x <= 1
            or not 0 <= center_y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            raise RuntimeError(f"Out-of-range YOLO row at {path}:{line_number}")
        boxes += 1
    return boxes


def check_plantseg_detection(raw_rows: list[dict[str, str]]) -> None:
    root = TRAINING_DIR / "PlantSegDetection"
    for metadata_split, split in (("Training", "train"), ("Validation", "val")):
        expected_names = {
            Path(row["Name"]).stem for row in raw_rows if row["Split"] == metadata_split
        }
        images = image_files(root / "images" / split)
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if {path.stem for path in images} != expected_names:
            raise RuntimeError(f"PlantSeg detection {split} images do not match metadata")
        if {path.stem for path in labels} != expected_names:
            raise RuntimeError(f"PlantSeg detection {split} labels do not match metadata")
        if sum(validate_yolo_label(path, 1) for path in labels) != len(expected_names):
            raise RuntimeError(f"PlantSeg detection {split} must have one lesion box per image")
    print("[ok] plantseg detection view matches train/validation metadata")


def check_test_datasets(raw_rows: list[dict[str, str]], config: dict) -> None:
    expected_test = [row for row in raw_rows if row["Split"] == "Test"]
    full = TESTS_DIR / "PlantSeg" / "full"
    full_rows = csv_rows(full / "Metadata.csv")
    if full_rows != expected_test:
        raise RuntimeError("Full PlantSeg test metadata does not match the official test split")
    if len(image_files(full / "images")) != len(full_rows):
        raise RuntimeError("Full PlantSeg test image view is incomplete")
    if len(image_files(full / "masks")) != len(full_rows):
        raise RuntimeError("Full PlantSeg test mask view is incomplete")

    mapping = config["plantseg_to_plantvillage"]
    mapped_names = set(mapping)
    expected_overlap_names = {
        row["Name"]
        for row in expected_test
        if f"{row['Plant']} / {row['Disease']}" in mapped_names
    }
    ps_root = TESTS_DIR / "overlap" / "PlantSeg"
    ps_rows = csv_rows(ps_root / "Metadata.csv")
    if {row["Name"] for row in ps_rows} != expected_overlap_names:
        raise RuntimeError("PlantSeg overlap view does not match the configured taxonomy")
    if len(image_files(ps_root / "images")) != len(ps_rows):
        raise RuntimeError("PlantSeg overlap image view is incomplete")
    if len(image_files(ps_root / "masks")) != len(ps_rows):
        raise RuntimeError("PlantSeg overlap mask view is incomplete")

    pv_root = TESTS_DIR / "overlap" / "PlantVillage"
    pv_rows = csv_rows(pv_root / "Metadata.csv")
    mapped_ids = {str(details["class_id"]) for details in mapping.values()}
    if not pv_rows or {row["Index"] for row in pv_rows} != mapped_ids:
        raise RuntimeError("PlantVillage overlap view does not cover the configured taxonomy")
    if len(image_files(pv_root / "images")) != len(pv_rows):
        raise RuntimeError("PlantVillage overlap image view is incomplete")

    variants = csv_rows(TESTS_DIR / "robustness" / "PlantSeg" / "variants.csv")
    expected_variants = {
        (corruption, str(severity))
        for corruption in config["robustness"]["corruptions"]
        for severity in config["robustness"]["severity_levels"]
    }
    if {(row["corruption"], row["severity"]) for row in variants} != expected_variants:
        raise RuntimeError("PlantSeg robustness variants do not match project.yaml")
    print(
        f"[ok] test views: {len(full_rows):,} full, "
        f"{len(ps_rows):,}/{len(pv_rows):,} overlap, {len(variants)} robustness variants"
    )


def check_plantvillage() -> None:
    color_dir = RAW_DIR / "PlantVillage" / "raw" / "color"
    class_dirs = sorted(path for path in color_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError("PlantVillage contains no class directories")
    counts = {path.name: len(image_files(path)) for path in class_dirs}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError("At least one PlantVillage class is empty")
    for path in class_dirs:
        decode_image(image_files(path)[0])
    print(f"[ok] plantvillage raw: {sum(counts.values()):,} images, {len(counts)} classes")


def check_plantdoc() -> None:
    root = RAW_DIR / "PlantDoc"
    records = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    classes = (root / "classes.txt").read_text(encoding="utf-8").splitlines()
    report = json.loads((root / "extraction_report.json").read_text(encoding="utf-8"))
    if not classes:
        raise RuntimeError("PlantDoc class list is empty")
    if len(records) != report["statistics"]["images_written"]:
        raise RuntimeError("PlantDoc records do not match its extraction report")

    box_count = 0
    for record in records:
        image_path = root / record["image"]
        label_path = root / record["label_file"]
        if not image_path.is_file() or not label_path.is_file():
            raise RuntimeError(f"PlantDoc record has a missing file: {record}")
        box_count += validate_yolo_label(label_path, len(classes))
    if box_count != report["statistics"]["boxes_written"]:
        raise RuntimeError("PlantDoc labels do not match its extraction report")
    for index in sorted({0, len(records) // 2, len(records) - 1}):
        decode_image(root / records[index]["image"])
    print(f"[ok] plantdoc raw: {len(records):,} images, {len(classes)} classes")


def main() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    raw_rows, class_count = check_plantseg()
    check_plantseg_training(raw_rows, class_count)
    check_plantseg_detection(raw_rows)
    check_test_datasets(raw_rows, config)
    check_plantvillage()
    check_plantdoc()
    print("All dataset relationships passed")


if __name__ == "__main__":
    main()
