"""Audit the extracted PlantSeg release and report annotation inconsistencies."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from plant_disease.paths import OUTPUTS_DIR, RAW_DIR

SPLIT_NAMES = {"Training": "train", "Validation": "val", "Test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_DIR / "PlantSeg" / "plantseg")
    parser.add_argument(
        "--output", type=Path, default=OUTPUTS_DIR / "plantseg_audit.json"
    )
    return parser.parse_args()


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask > 0)
    if not len(columns):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def coco_union_bbox(annotations: list[dict]) -> tuple[float, float, float, float] | None:
    if not annotations:
        return None
    return (
        min(annotation["bbox"][0] for annotation in annotations),
        min(annotation["bbox"][1] for annotation in annotations),
        max(annotation["bbox"][0] + annotation["bbox"][2] for annotation in annotations),
        max(annotation["bbox"][1] + annotation["bbox"][3] for annotation in annotations),
    )


def bboxes_agree(left: tuple | None, right: tuple | None, tolerance: float = 1.01) -> bool:
    if left is None or right is None:
        return left == right
    return all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))


def append_example(examples: list, value: object, limit: int = 10) -> None:
    if len(examples) < limit:
        examples.append(value)


def main() -> None:
    args = parse_args()
    metadata_path = args.root / "Metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        metadata_fields = reader.fieldnames or []

    rows_by_name = {row["Name"]: row for row in rows}
    class_mapping: dict[int, set[tuple[str, str]]] = defaultdict(set)
    class_counts = Counter()
    split_counts = Counter()
    license_counts = Counter()
    for row in rows:
        class_id = int(row["Index"])
        class_mapping[class_id].add((row["Plant"], row["Disease"]))
        class_counts[class_id] += 1
        split_counts[row["Split"]] += 1
        license_counts[row["License"]] += 1

    checks = Counter()
    examples: dict[str, list] = defaultdict(list)
    mask_details = {}
    for row in rows:
        split = SPLIT_NAMES[row["Split"]]
        image_path = args.root / "images" / split / row["Name"]
        mask_path = args.root / "annotations" / split / row["Label file"]
        if not image_path.exists():
            checks["missing_images"] += 1
            append_example(examples["missing_images"], row["Name"])
            continue
        if not mask_path.exists():
            checks["missing_masks"] += 1
            append_example(examples["missing_masks"], row["Label file"])
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                raw_image_size = image.size
                visual_image_size = ImageOps.exif_transpose(image).size
            with Image.open(mask_path) as mask_image:
                mask_image.load()
                mask = np.asarray(mask_image)
                mask_size = mask_image.size
        except Exception as error:
            checks["decode_failures"] += 1
            append_example(examples["decode_failures"], [row["Name"], str(error)])
            continue

        if raw_image_size != mask_size:
            checks["raw_image_mask_size_mismatches"] += 1
            append_example(
                examples["raw_image_mask_size_mismatches"],
                [row["Name"], raw_image_size, mask_size],
            )
        if visual_image_size != mask_size:
            checks["exif_corrected_size_mismatches"] += 1
            append_example(
                examples["exif_corrected_size_mismatches"],
                [row["Name"], visual_image_size, mask_size],
            )

        class_id = int(row["Index"])
        values = set(np.unique(mask).tolist())
        expected_mask_value = class_id + 1
        if not values.issubset({0, expected_mask_value}) or expected_mask_value not in values:
            checks["unexpected_mask_values"] += 1
            append_example(examples["unexpected_mask_values"], [row["Name"], sorted(values)])
        measured_ratio = float((mask > 0).mean())
        if abs(measured_ratio - float(row["Mask ratio"])) > 1e-9:
            checks["mask_ratio_mismatches"] += 1
            append_example(
                examples["mask_ratio_mismatches"],
                [row["Name"], float(row["Mask ratio"]), measured_ratio],
            )
        mask_details[row["Name"]] = {
            "bbox": mask_bbox(mask),
            "size": mask_size,
            "class_id": class_id,
        }

    coco_splits = {}
    for split in ("train", "val", "test"):
        coco_path = args.root / f"annotation_{split}.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        annotations_by_image = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        split_checks = Counter()
        for image in coco["images"]:
            name = image["file_name"]
            annotations = annotations_by_image[image["id"]]
            metadata = rows_by_name.get(name)
            if metadata is None:
                split_checks["images_missing_from_metadata"] += 1
                append_example(examples["images_missing_from_metadata"], name)
                continue
            if not annotations:
                split_checks["images_without_annotations"] += 1
                append_example(examples["images_without_annotations"], name)
                continue
            expected_class_id = int(metadata["Index"])
            if any(annotation["category_id"] != expected_class_id for annotation in annotations):
                split_checks["category_id_mismatches"] += 1
                append_example(examples["category_id_mismatches"], name)
            if any(
                annotation["bbox"][0] < 0
                or annotation["bbox"][1] < 0
                or annotation["bbox"][0] + annotation["bbox"][2] > image["width"] + 1
                or annotation["bbox"][1] + annotation["bbox"][3] > image["height"] + 1
                for annotation in annotations
            ):
                split_checks["images_with_out_of_bounds_boxes"] += 1
                append_example(examples["images_with_out_of_bounds_boxes"], name)
            details = mask_details.get(name)
            if details and not bboxes_agree(
                details["bbox"], coco_union_bbox(annotations)
            ):
                split_checks["images_where_coco_and_mask_boxes_disagree"] += 1
                append_example(
                    examples["images_where_coco_and_mask_boxes_disagree"], name
                )

        metadata_names = {
            row["Name"] for row in rows if SPLIT_NAMES[row["Split"]] == split
        }
        coco_names = {image["file_name"] for image in coco["images"]}
        coco_splits[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
            "metadata_images_missing_from_coco": len(metadata_names - coco_names),
            **dict(split_checks),
        }

    file_check_names = (
        "missing_images",
        "missing_masks",
        "decode_failures",
        "raw_image_mask_size_mismatches",
        "exif_corrected_size_mismatches",
        "unexpected_mask_values",
        "mask_ratio_mismatches",
    )
    file_checks = {name: checks[name] for name in file_check_names}
    all_class_ids = set(class_mapping)
    split_class_coverage = {}
    for source_split, split in SPLIT_NAMES.items():
        present = {int(row["Index"]) for row in rows if row["Split"] == source_split}
        missing = sorted(all_class_ids - present)
        split_class_coverage[split] = {
            "classes_present": len(present),
            "missing_class_ids": missing,
        }

    ordered_mapping = {
        str(class_id): {
            "plant": next(iter(class_mapping[class_id]))[0],
            "disease": next(iter(class_mapping[class_id]))[1],
            "images": class_counts[class_id],
        }
        for class_id in sorted(class_mapping)
    }
    counts = list(class_counts.values())
    report = {
        "release": {
            "doi": "10.5281/zenodo.17719108",
            "archive_size": 1_057_281_724,
            "archive_md5": "9358a66dff88cdd15c4fe009763c40a3",
        },
        "metadata": {
            "fields": metadata_fields,
            "rows": len(rows),
            "plants": len({row["Plant"] for row in rows}),
            "plant_disease_classes": len(class_mapping),
            "split_counts": dict(split_counts),
            "license_counts": dict(license_counts),
            "class_images_min": min(counts),
            "class_images_median": statistics.median(counts),
            "class_images_max": max(counts),
            "class_mapping_conflicts": sum(len(values) != 1 for values in class_mapping.values()),
            "split_class_coverage": split_class_coverage,
        },
        "file_and_mask_checks": file_checks,
        "coco_splits": coco_splits,
        "examples": dict(examples),
        "class_mapping": ordered_mapping,
        "interpretation": {
            "mask_encoding": "Normally 0 is background; Index + 1 is the class",
            "authoritative_localization": "Use PNG masks and derive boxes from them",
            "coco_warning": "COCO categories are empty and some COCO boxes disagree with masks",
            "class_value_warning": (
                "Use masks as binary lesion masks and Metadata.csv as the class authority"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "class_mapping"}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
