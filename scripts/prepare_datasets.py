#!/usr/bin/env python3
"""Create reproducible PlantSeg training and project test dataset views."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from plant_disease.paths import (
    DOWNLOADS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    TESTS_DIR,
    TRAINING_DIR,
    ensure_project_directories,
)


def create_relative_link(link: Path, target: Path) -> None:
    """Create an idempotent relative symbolic link without replacing real files."""
    relative_target = Path(os.path.relpath(target, start=link.parent))
    if link.is_symlink() and link.readlink() == relative_target:
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Cannot create dataset link because {link} already exists")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative_target, target_is_directory=target.is_dir())


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sync_image_links(destination: Path, sources: list[Path]) -> None:
    """Make a directory contain exactly the requested generated image links."""
    destination.mkdir(parents=True, exist_ok=True)
    expected = {source.name for source in sources}
    if len(expected) != len(sources):
        raise RuntimeError(f"Duplicate filenames cannot be flattened into {destination}")
    for existing in destination.iterdir():
        if existing.name not in expected:
            if not existing.is_symlink():
                raise RuntimeError(f"Unexpected non-link file in generated view: {existing}")
            existing.unlink()
    for source in sources:
        create_relative_link(destination / source.name, source)


def jpeg_needs_repair(path: Path) -> bool:
    if path.suffix.casefold() not in {".jpg", ".jpeg"}:
        return False
    with path.open("rb") as handle:
        handle.seek(-2, os.SEEK_END)
        return handle.read() != b"\xff\xd9"


def sync_yolo_images(destination: Path, sources: list[Path]) -> int:
    """Link normal images but copy JPEGs that Ultralytics would repair in place."""
    destination.mkdir(parents=True, exist_ok=True)
    expected = {source.name for source in sources}
    if len(expected) != len(sources):
        raise RuntimeError(f"Duplicate filenames cannot be flattened into {destination}")
    for existing in destination.iterdir():
        if existing.name not in expected:
            if not existing.is_symlink() and not existing.is_file():
                raise RuntimeError(f"Unexpected entry in generated YOLO view: {existing}")
            existing.unlink()

    copied = 0
    for source in sources:
        target = destination / source.name
        if jpeg_needs_repair(source):
            copied += 1
            if target.is_symlink():
                target.unlink()
            if not target.exists():
                shutil.copy2(source, target)
            continue
        if target.exists() and not target.is_symlink():
            target.unlink()
        create_relative_link(target, source)
    return copied


def read_plantseg_metadata(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    metadata = root / "Metadata.csv"
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"PlantSeg metadata has no header: {metadata}")
    return rows, fieldnames


def prepare_plantseg_training() -> None:
    """Create a lightweight view containing only PlantSeg's official training split."""
    source = RAW_DIR / "PlantSeg" / "plantseg"
    required = (
        source / "images" / "train",
        source / "annotations" / "train",
        source / "annotation_train.json",
        source / "Metadata.csv",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("PlantSeg must be downloaded before preparing its training view")

    destination = TRAINING_DIR / "PlantSeg"
    create_relative_link(destination / "images", source / "images" / "train")
    create_relative_link(destination / "masks", source / "annotations" / "train")
    create_relative_link(destination / "annotations.json", source / "annotation_train.json")
    rows, fieldnames = read_plantseg_metadata(source)
    training_rows = [row for row in rows if row["Split"] == "Training"]
    write_csv(destination / "Metadata.csv", training_rows, fieldnames)
    print(
        f"PlantSeg training view ready at {destination} "
        f"({len(training_rows):,} images; raw files are linked, not copied)"
    )


def mask_to_yolo_box(mask_path: Path) -> tuple[float, float, float, float]:
    """Return one normalized box enclosing every lesion pixel in a mask."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    rows, columns = np.where(mask > 0)
    if not len(columns):
        raise RuntimeError(f"PlantSeg mask contains no lesion pixels: {mask_path}")

    height, width = mask.shape[:2]
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return (
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def write_detection_labels(
    destination: Path,
    mask_dir: Path,
    rows: list[dict[str, str]],
    *,
    class_aware: bool = False,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination.with_suffix(".cache")
    if cache_path.is_file():
        cache_path.unlink()
    expected = {Path(row["Name"]).with_suffix(".txt").name for row in rows}
    for existing in destination.iterdir():
        if existing.is_file() and existing.name not in expected:
            existing.unlink()

    for row in rows:
        center_x, center_y, width, height = mask_to_yolo_box(mask_dir / row["Label file"])
        class_id = int(row["Index"]) if class_aware else 0
        label_path = destination / Path(row["Name"]).with_suffix(".txt").name
        label_path.write_text(
            f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n",
            encoding="utf-8",
        )
    return len(rows)


def plantseg_class_names(rows: list[dict[str, str]]) -> dict[int, str]:
    """Return PlantSeg's stable disease taxonomy indexed by metadata class ID."""
    names: dict[int, str] = {}
    for row in rows:
        class_id = int(row["Index"])
        class_name = f'{row["Plant"]} / {row["Disease"]}'
        if class_id in names and names[class_id] != class_name:
            raise RuntimeError(f"Conflicting PlantSeg names for class ID {class_id}")
        names[class_id] = class_name
    expected = set(range(len(names)))
    if set(names) != expected:
        raise RuntimeError("PlantSeg class IDs must be contiguous from zero")
    return names


def prepare_detection_view(
    destination: Path,
    rows: list[dict[str, str]],
    split_details: dict[str, tuple[str, Path, Path]],
    names: dict[int, str],
    *,
    class_aware: bool,
) -> tuple[dict[str, int], int]:
    """Build one YOLO view, sharing source images wherever possible."""
    counts = {}
    copied_images = 0
    for split, (metadata_split, image_dir, mask_dir) in split_details.items():
        split_rows = [row for row in rows if row["Split"] == metadata_split]
        image_destination = destination / "images" / split
        if image_destination.is_symlink():
            image_destination.unlink()
        copied_images += sync_yolo_images(
            image_destination,
            [image_dir / row["Name"] for row in split_rows],
        )
        counts[split] = write_detection_labels(
            destination / "labels" / split,
            mask_dir,
            split_rows,
            class_aware=class_aware,
        )

    dataset = {
        "path": str(destination.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (destination / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False),
        encoding="utf-8",
    )
    return counts, copied_images


def prepare_plantseg_detection() -> None:
    """Build class-agnostic and disease-aware YOLO views from PlantSeg masks."""
    source = RAW_DIR / "PlantSeg" / "plantseg"
    rows, _ = read_plantseg_metadata(source)
    split_details = {
        "train": ("Training", source / "images" / "train", source / "annotations" / "train"),
        "val": ("Validation", source / "images" / "val", source / "annotations" / "val"),
    }
    lesion_destination = TRAINING_DIR / "PlantSegDetection"
    lesion_counts, lesion_copies = prepare_detection_view(
        lesion_destination,
        rows,
        split_details,
        {0: "lesion"},
        class_aware=False,
    )
    print(
        f"PlantSeg lesion YOLO view ready at {lesion_destination} "
        f"({lesion_counts['train']:,} train / {lesion_counts['val']:,} validation labels; "
        f"one lesion class; {lesion_copies} repair-safe image copies)"
    )

    disease_destination = TRAINING_DIR / "PlantSegDiseaseDetection"
    disease_names = plantseg_class_names(rows)
    disease_counts, disease_copies = prepare_detection_view(
        disease_destination,
        rows,
        split_details,
        disease_names,
        class_aware=True,
    )
    print(
        f"PlantSeg standalone YOLO view ready at {disease_destination} "
        f"({disease_counts['train']:,} train / {disease_counts['val']:,} validation labels; "
        f"{len(disease_names)} disease classes; {disease_copies} repair-safe image copies)"
    )


def prepare_test_datasets() -> None:
    """Create clean, overlap, and robustness test views without copying images."""
    plantseg = RAW_DIR / "PlantSeg" / "plantseg"
    plantvillage = RAW_DIR / "PlantVillage"
    pv_split = DOWNLOADS_DIR / "plantvillage" / "splits" / "color_test.txt"
    config_path = PROJECT_ROOT / "configs" / "project.yaml"
    required = (plantseg / "Metadata.csv", pv_split, config_path)
    if not all(path.is_file() for path in required):
        raise RuntimeError("PlantSeg and PlantVillage must be downloaded before test preparation")

    plantseg_rows, plantseg_fields = read_plantseg_metadata(plantseg)
    test_rows = [row for row in plantseg_rows if row["Split"] == "Test"]
    full = TESTS_DIR / "PlantSeg" / "full"
    create_relative_link(full / "images", plantseg / "images" / "test")
    create_relative_link(full / "masks", plantseg / "annotations" / "test")
    create_relative_link(full / "annotations.json", plantseg / "annotation_test.json")
    write_csv(full / "Metadata.csv", test_rows, plantseg_fields)

    detection = TESTS_DIR / "PlantSeg" / "detection"
    detection_images = detection / "images" / "test"
    if detection_images.is_symlink():
        detection_images.unlink()
    copied_images = sync_yolo_images(
        detection_images,
        [plantseg / "images" / "test" / row["Name"] for row in test_rows],
    )
    detection_labels = write_detection_labels(
        detection / "labels" / "test",
        plantseg / "annotations" / "test",
        test_rows,
    )
    detection_config = {
        "path": str(detection.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {0: "lesion"},
    }
    (detection / "dataset.yaml").write_text(
        yaml.safe_dump(detection_config, sort_keys=False),
        encoding="utf-8",
    )

    disease_detection = TESTS_DIR / "PlantSeg" / "disease_detection"
    disease_images = disease_detection / "images" / "test"
    if disease_images.is_symlink():
        disease_images.unlink()
    disease_copies = sync_yolo_images(
        disease_images,
        [plantseg / "images" / "test" / row["Name"] for row in test_rows],
    )
    disease_labels = write_detection_labels(
        disease_detection / "labels" / "test",
        plantseg / "annotations" / "test",
        test_rows,
        class_aware=True,
    )
    disease_config = {
        "path": str(disease_detection.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": plantseg_class_names(plantseg_rows),
    }
    (disease_detection / "dataset.yaml").write_text(
        yaml.safe_dump(disease_config, sort_keys=False),
        encoding="utf-8",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = config["plantseg_to_plantvillage"]
    shared_names = set(mapping)
    plantseg_overlap_rows = [
        row for row in test_rows if f"{row['Plant']} / {row['Disease']}" in shared_names
    ]
    ps_overlap = TESTS_DIR / "overlap" / "PlantSeg"
    sync_image_links(
        ps_overlap / "images",
        [plantseg / "images" / "test" / row["Name"] for row in plantseg_overlap_rows],
    )
    sync_image_links(
        ps_overlap / "masks",
        [plantseg / "annotations" / "test" / row["Label file"] for row in plantseg_overlap_rows],
    )
    write_csv(ps_overlap / "Metadata.csv", plantseg_overlap_rows, plantseg_fields)

    village_to_shared = {
        details["plantvillage_class"]: (shared_name, int(details["class_id"]))
        for shared_name, details in mapping.items()
    }
    pv_rows = []
    pv_sources = []
    for relative_name in pv_split.read_text(encoding="utf-8").splitlines():
        relative_path = Path(relative_name)
        village_class = relative_path.parent.name
        if village_class not in village_to_shared:
            continue
        shared_name, class_id = village_to_shared[village_class]
        plant, disease = shared_name.split(" / ", maxsplit=1)
        source = plantvillage / relative_path
        pv_sources.append(source)
        pv_rows.append(
            {
                "Name": source.name,
                "Index": str(class_id),
                "Plant": plant,
                "Disease": disease,
                "PlantVillage class": village_class,
                "Split": "Test",
                "Source path": relative_name,
            }
        )
    pv_overlap = TESTS_DIR / "overlap" / "PlantVillage"
    sync_image_links(pv_overlap / "images", pv_sources)
    pv_fields = [
        "Name",
        "Index",
        "Plant",
        "Disease",
        "PlantVillage class",
        "Split",
        "Source path",
    ]
    write_csv(pv_overlap / "Metadata.csv", pv_rows, pv_fields)

    robustness = TESTS_DIR / "robustness" / "PlantSeg"
    create_relative_link(robustness / "clean", full)
    variant_rows = [
        {"corruption": corruption, "severity": str(severity), "generation": "on_demand"}
        for corruption in config["robustness"]["corruptions"]
        for severity in config["robustness"]["severity_levels"]
    ]
    write_csv(
        robustness / "variants.csv",
        variant_rows,
        ["corruption", "severity", "generation"],
    )
    print(
        "Test views ready at "
        f"{TESTS_DIR} (PlantSeg full: {len(test_rows):,}; overlap: "
        f"{len(plantseg_overlap_rows):,} PlantSeg / {len(pv_rows):,} PlantVillage; "
        f"lesion/disease detection labels: {detection_labels:,}/{disease_labels:,}; "
        f"robustness variants: {len(variant_rows)}; repair-safe test copies: "
        f"{copied_images + disease_copies})"
    )


def main() -> None:
    ensure_project_directories()
    prepare_plantseg_training()
    prepare_plantseg_detection()
    prepare_test_datasets()


if __name__ == "__main__":
    main()
