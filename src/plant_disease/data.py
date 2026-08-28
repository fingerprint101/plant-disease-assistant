from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class PlantSegClassificationDataset(Dataset):
    """PlantSeg images and class IDs read from the authoritative metadata."""

    def __init__(
        self,
        metadata_path: Path,
        image_dir: Path,
        transform: Callable | None = None,
        split: str | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        if not metadata_path.is_file():
            raise FileNotFoundError(f"PlantSeg metadata not found: {metadata_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"PlantSeg image directory not found: {image_dir}")

        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if split is not None:
            rows = [row for row in rows if row["Split"].casefold() == split.casefold()]
        if not rows:
            suffix = f" for split {split!r}" if split else ""
            raise ValueError(f"No PlantSeg rows found in {metadata_path}{suffix}")

        self.samples = [(row["Name"], int(row["Index"])) for row in rows]
        missing = [name for name, _ in self.samples if not (image_dir / name).is_file()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} images referenced by {metadata_path} are missing from "
                f"{image_dir} (for example: {preview})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_name, target = self.samples[index]
        with Image.open(self.image_dir / image_name) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def plantseg_class_names(metadata_path: Path, num_classes: int) -> list[str]:
    """Build the stable class-ID-to-name mapping encoded in PlantSeg metadata."""
    names: list[str | None] = [None] * num_classes
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["Index"])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"PlantSeg class ID {class_id} is outside 0..{num_classes - 1}")
            class_name = f'{row["Plant"]} / {row["Disease"]}'
            if names[class_id] not in {None, class_name}:
                raise ValueError(f"Conflicting names found for PlantSeg class ID {class_id}")
            names[class_id] = class_name

    missing = [str(index) for index, name in enumerate(names) if name is None]
    if missing:
        raise ValueError(f"PlantSeg metadata is missing class IDs: {', '.join(missing)}")
    return [name for name in names if name is not None]


def training_transform(image_size: int = 224) -> Callable:
    """ImageNet-normalized augmentation used for PlantSeg classifier training."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
