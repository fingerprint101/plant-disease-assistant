from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)
from torchvision.transforms import v2

from plant_disease.paths import MODELS_DIR

if TYPE_CHECKING:
    from pytorch_grad_cam import GradCAM
    from ultralytics import YOLO

CLASSIFICATION_MODELS = ("baseline_cnn", "efficientnet_b0", "mobilenet_v2")
DEFAULT_YOLO_MODEL = "yolo11n.pt"


class BaselineCNN(nn.Module):
    """Small classifier trained from scratch for the project baseline."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a project classifier with a task-specific output layer."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    choices = ", ".join(CLASSIFICATION_MODELS)
    raise ValueError(f"Unknown classifier {name!r}; choose one of: {choices}")


def classification_transform(image_size: int = 224) -> Callable:
    """Return the shared ImageNet preprocessing used by all initial experiments."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def gradcam_target_layer(name: str, model: nn.Module) -> nn.Module:
    """Return the final convolutional feature layer for a project classifier."""
    if name == "baseline_cnn":
        return model.features[-1][0]
    if name in {"efficientnet_b0", "mobilenet_v2"}:
        return model.features[-1]
    choices = ", ".join(CLASSIFICATION_MODELS)
    raise ValueError(f"Grad-CAM is not configured for {name!r}; choose one of: {choices}")


def build_gradcam(name: str, model: nn.Module) -> GradCAM:
    """Create Grad-CAM with the correct target layer for a project classifier."""
    from pytorch_grad_cam import GradCAM

    return GradCAM(model=model, target_layers=[gradcam_target_layer(name, model)])


def load_yolo(
    model_name: str = DEFAULT_YOLO_MODEL,
    model_dir: Path = MODELS_DIR,
    task: str = "detect",
) -> YOLO:
    """Load a cached YOLO model, downloading its official checkpoint when absent."""
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    return YOLO(str(model_dir / model_name), task=task)


def model_parameter_count(model: nn.Module) -> int:
    """Return the total number of parameters in a PyTorch model."""
    return sum(parameter.numel() for parameter in model.parameters())
