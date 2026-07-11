"""Extract PlantDoc Parquet shards to images, JSON metadata, and YOLO labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep rows with invalid boxes after clipping them to image boundaries.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def all_categories(files: list[Path]) -> list[str]:
    categories = set()
    for path in files:
        table = pq.read_table(path, columns=["objects"])
        for objects in table.column("objects").to_pylist():
            categories.update(objects["category"] or [])
    return sorted(categories)


def normalize_box(box: list[float], width: int, height: int) -> tuple[list[float], bool]:
    x, y, box_width, box_height = map(float, box)
    valid = (
        box_width > 0
        and box_height > 0
        and x >= 0
        and y >= 0
        and x + box_width <= width + 1
        and y + box_height <= height + 1
    )
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], valid


def main() -> None:
    args = parse_args()
    files = sorted(args.parquet_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in {args.parquet_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.output_dir} is not empty; pass --overwrite to continue")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    classes = all_categories(files)
    class_to_id = {name: index for index, name in enumerate(classes)}
    for split in ("train", "test"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records = []
    statistics = Counter()
    seen_hashes = {}

    for parquet_path in files:
        split = "test" if parquet_path.name.startswith("test") else "train"
        rows = pq.read_table(parquet_path).to_pylist()
        for row_index, row in enumerate(tqdm(rows, desc=parquet_path.name)):
            statistics["rows"] += 1
            try:
                payload = row["image"]["bytes"]
                image = Image.open(BytesIO(payload)).convert("RGB")
                image.load()
            except Exception:
                statistics["decode_failures"] += 1
                continue

            digest = hashlib.sha1(payload).hexdigest()
            if digest in seen_hashes:
                statistics["duplicate_images"] += 1
            else:
                seen_hashes[digest] = split
            if seen_hashes[digest] != split:
                statistics["cross_split_duplicates"] += 1

            labels = row["objects"]["category"] or []
            source_boxes = row["objects"]["bbox"] or []
            converted = [normalize_box(box, image.width, image.height) for box in source_boxes]
            has_invalid = any(not valid for _, valid in converted)
            if has_invalid and not args.keep_invalid:
                statistics["rows_skipped_invalid_boxes"] += 1
                continue

            boxes = [box for box, _ in converted]
            if any(box[2] <= 0 or box[3] <= 0 for box in boxes):
                statistics["rows_skipped_empty_boxes"] += 1
                continue

            stem = f"{split}_{parquet_path.stem}_{row_index:05d}_{row['image_id']}"
            image_path = args.output_dir / "images" / split / f"{stem}.jpg"
            label_path = args.output_dir / "labels" / split / f"{stem}.txt"
            image.save(image_path, quality=95)

            yolo_lines = []
            annotations = []
            for label, (x, y, box_width, box_height) in zip(labels, boxes, strict=True):
                class_id = class_to_id[label]
                cx = (x + box_width / 2) / image.width
                cy = (y + box_height / 2) / image.height
                nw = box_width / image.width
                nh = box_height / image.height
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                annotations.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "bbox_xywh": [x, y, box_width, box_height],
                    }
                )
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            records.append(
                {
                    "image": str(image_path.relative_to(args.output_dir)),
                    "label_file": str(label_path.relative_to(args.output_dir)),
                    "source_file": parquet_path.name,
                    "source_image_id": row["image_id"],
                    "source_split": split,
                    "sha1": digest,
                    "width": image.width,
                    "height": image.height,
                    "annotations": annotations,
                }
            )
            statistics["images_written"] += 1
            statistics["boxes_written"] += len(annotations)

    (args.output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (args.output_dir / "annotations.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    report = {
        "classes": classes,
        "statistics": dict(statistics),
        "note": (
            "Duplicates are reported but retained. Create the final grouped split "
            "after deduplication."
        ),
    }
    (args.output_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":
    main()
