"""Export complete dataset records with visual previews."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from plant_disease.paths import DOWNLOADS_DIR, OUTPUTS_DIR, RAW_DIR

OUTPUT_DIR = OUTPUTS_DIR / "dataset_examples"


def plantvillage_class_names() -> list[str]:
    readme = (DOWNLOADS_DIR / "plantvillage" / "README.md").read_text(encoding="utf-8")
    _, front_matter, _ = readme.split("---", 2)
    metadata = yaml.safe_load(front_matter)
    features = metadata["dataset_info"]["features"]
    label_feature = next(feature for feature in features if feature["name"] == "label")
    names = label_feature["dtype"]["class_label"]["names"]
    return [names[str(index)] for index in range(len(names))]


def plantvillage_leaf_id(class_name: str, filename: str, leaf_map: dict) -> str:
    image_identifier = filename.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    image_identifier = Path(image_identifier).stem.strip()
    suggestions = leaf_map.get(image_identifier.lower(), [])
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback_{image_identifier}"


def export_plantvillage() -> dict:
    split_path = DOWNLOADS_DIR / "plantvillage" / "splits" / "color_train.txt"
    image_path = next(line.strip() for line in split_path.read_text().splitlines() if line.strip())
    source_path = RAW_DIR / "PlantVillage" / image_path
    image = Image.open(source_path)
    class_name = Path(image_path).parent.name
    crop, disease = class_name.split("___", maxsplit=1)
    leaf_map = json.loads(
        (DOWNLOADS_DIR / "plantvillage" / "leaf_grouping" / "leaf-map.json").read_text()
    )
    classes = plantvillage_class_names()

    preview_path = OUTPUT_DIR / "plantvillage_example.jpg"
    image.convert("RGB").save(preview_path, quality=95)
    return {
        "source_schema_fields": ["image", "image_path", "label", "crop", "disease", "leaf_id"],
        "split": "train",
        "image": {
            "source_path": str(source_path),
            "preview_path": str(preview_path),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        },
        "image_path": image_path,
        "label": {"id": classes.index(class_name), "name": class_name},
        "crop": crop,
        "disease": disease,
        "leaf_id": plantvillage_leaf_id(class_name, source_path.name, leaf_map),
    }


def boxes_are_valid(boxes: list[list[float]], width: int, height: int) -> bool:
    return all(
        box_width > 0
        and box_height > 0
        and x >= 0
        and y >= 0
        and x + box_width <= width + 1
        and y + box_height <= height + 1
        for x, y, box_width, box_height in boxes
    )


def export_plantdoc() -> dict:
    parquet_path = sorted((DOWNLOADS_DIR / "PlantDoc-parquet").glob("train*.parquet"))[0]
    selected = None
    selected_image = None
    for row in pq.read_table(parquet_path).to_pylist():
        image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        boxes = row["objects"]["bbox"] or []
        if len(boxes) >= 2 and boxes_are_valid(boxes, image.width, image.height):
            selected = row
            selected_image = image
            break
    if selected is None or selected_image is None:
        raise RuntimeError("No suitable PlantDoc example found")

    labels = selected["objects"]["category"] or []
    boxes = selected["objects"]["bbox"] or []
    annotated = selected_image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=16)
    for label, (x, y, width, height) in zip(labels, boxes, strict=True):
        draw.rectangle((x, y, x + width, y + height), outline="#ff2d2d", width=4)
        text_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(text_box, fill="#ff2d2d")
        draw.text((x, y), label, fill="white", font=font)

    original_path = OUTPUT_DIR / "plantdoc_example.jpg"
    annotated_path = OUTPUT_DIR / "plantdoc_example_annotated.jpg"
    selected_image.save(original_path, quality=95)
    annotated.save(annotated_path, quality=95)
    payload = selected["image"]["bytes"]

    return {
        "source_schema_fields": ["image_id", "image", "width", "height", "objects"],
        "split": "train",
        "source_parquet": str(parquet_path),
        "image_id": selected["image_id"],
        "image": {
            "bytes": "<binary omitted from JSON display>",
            "byte_length": len(payload),
            "sha1": hashlib.sha1(payload).hexdigest(),
            "path": selected["image"]["path"],
            "preview_path": str(original_path),
            "annotated_preview_path": str(annotated_path),
            "decoded_mode": selected_image.mode,
            "decoded_width": selected_image.width,
            "decoded_height": selected_image.height,
        },
        "width": selected["width"],
        "height": selected["height"],
        "objects": selected["objects"],
    }


def export_plantseg() -> dict:
    root = RAW_DIR / "PlantSeg" / "plantseg"
    with (root / "Metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        metadata_rows = {row["Name"]: row for row in reader}
        metadata_fields = reader.fieldnames or []
    coco = json.loads((root / "annotation_train.json").read_text(encoding="utf-8"))
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    selected_image_record = None
    selected_annotations = None
    selected_image = None
    selected_mask = None
    for image_record in coco["images"]:
        annotations = annotations_by_image[image_record["id"]]
        metadata = metadata_rows[image_record["file_name"]]
        image_path = root / "images" / "train" / image_record["file_name"]
        mask_path = root / "annotations" / "train" / metadata["Label file"]
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if (
            len(annotations) == 1
            and image.size == mask.size
            and annotations[0]["category_id"] == int(metadata["Index"])
        ):
            pixels = np.asarray(mask)
            rows, columns = np.where(pixels > 0)
            mask_box = [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ]
            x, y, width, height = annotations[0]["bbox"]
            coco_box = [x, y, x + width, y + height]
            if all(abs(a - b) <= 1.01 for a, b in zip(mask_box, coco_box, strict=True)):
                selected_image_record = image_record
                selected_annotations = annotations
                selected_image = image
                selected_mask = mask
                break
    if not all((selected_image_record, selected_annotations, selected_image, selected_mask)):
        raise RuntimeError("No internally consistent PlantSeg example found")

    metadata = metadata_rows[selected_image_record["file_name"]]
    pixels = np.asarray(selected_mask)
    overlay = selected_image.convert("RGBA")
    highlight = Image.new("RGBA", overlay.size, (228, 45, 45, 0))
    highlight.putalpha(Image.fromarray(np.where(pixels > 0, 120, 0).astype(np.uint8)))
    overlay = Image.alpha_composite(overlay, highlight).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for annotation in selected_annotations:
        x, y, width, height = annotation["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline="#ffd43b", width=4)

    original_path = OUTPUT_DIR / "plantseg_example.jpg"
    mask_path = OUTPUT_DIR / "plantseg_example_mask.png"
    annotated_path = OUTPUT_DIR / "plantseg_example_annotated.jpg"
    selected_image.save(original_path, quality=95)
    selected_mask.save(mask_path)
    overlay.save(annotated_path, quality=95)
    return {
        "source_schema_fields": {
            "metadata_csv": metadata_fields,
            "coco_image": list(selected_image_record),
            "coco_annotation": list(selected_annotations[0]),
            "coco_top_level": list(coco),
        },
        "split": "train",
        "metadata": metadata,
        "coco_image": selected_image_record,
        "coco_annotations": selected_annotations,
        "mask": {
            "path": str(root / "annotations" / "train" / metadata["Label file"]),
            "preview_path": str(mask_path),
            "mode": selected_mask.mode,
            "width": selected_mask.width,
            "height": selected_mask.height,
            "pixel_values": np.unique(pixels).tolist(),
            "nonzero_pixels": int((pixels > 0).sum()),
        },
        "image_previews": {
            "original": str(original_path),
            "mask_and_box_overlay": str(annotated_path),
        },
        "dataset_note": (
            "The source COCO categories array is empty; use Metadata.csv for class names."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {
        "plantseg": export_plantseg(),
        "plantvillage": export_plantvillage(),
        "plantdoc": export_plantdoc(),
    }
    output_path = OUTPUT_DIR / "records.json"
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(f"\nSaved examples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
