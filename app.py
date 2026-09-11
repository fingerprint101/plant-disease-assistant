#!/usr/bin/env python3
"""Streamlit prototype for the Plant Disease Assistant."""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO

from plant_disease.models import (
    CLASSIFICATION_MODELS,
    build_classifier,
    build_gradcam,
    classification_transform,
)
from plant_disease.paths import MODELS_DIR, PROJECT_ROOT

TWO_STAGE = "Lesion YOLO + classifier"
STANDALONE = "Standalone disease-aware YOLO"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expanded_union_box(
    boxes: np.ndarray, image_width: int, image_height: int, margin: float
) -> tuple[int, int, int, int]:
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = boxes[:, 2:].max(axis=0)
    padding_x = (right - left) * margin
    padding_y = (bottom - top) * margin
    return (
        max(0, int(np.floor(left - padding_x))),
        max(0, int(np.floor(top - padding_y))),
        min(image_width, int(np.ceil(right + padding_x))),
        min(image_height, int(np.ceil(bottom + padding_y))),
    )


@st.cache_resource(show_spinner="Loading two-stage models...")
def load_two_stage_models(classifier_name: str):
    config = yaml.safe_load((PROJECT_ROOT / "configs/project.yaml").read_text())
    detector_path = MODELS_DIR / "lesion_yolo_best.pt"
    classifier_path = MODELS_DIR / f"{classifier_name}_best.pt"
    for path in (detector_path, classifier_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint not found: {path}")

    device = choose_device()
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    if len(class_names) != config["classification"]["num_classes"]:
        raise RuntimeError("Classifier checkpoint does not contain the expected PlantSeg taxonomy")
    classifier = build_classifier(classifier_name, len(class_names), pretrained=False)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval().to(device)
    return YOLO(detector_path), classifier, class_names, config, device


@st.cache_resource(show_spinner="Loading standalone YOLO...")
def load_standalone_model():
    config = yaml.safe_load((PROJECT_ROOT / "configs/project.yaml").read_text())
    checkpoint_path = MODELS_DIR / "standalone_disease_yolo_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    detector = YOLO(checkpoint_path)
    class_names = [detector.names[index] for index in range(len(detector.names))]
    if len(class_names) != config["classification"]["num_classes"]:
        raise RuntimeError("Standalone YOLO does not contain the expected PlantSeg taxonomy")
    return detector, class_names, config, choose_device()


def gradcam_overlay(
    image: Image.Image,
    classifier: torch.nn.Module,
    classifier_name: str,
    tensor: torch.Tensor,
    predicted_id: int,
) -> Image.Image:
    with build_gradcam(classifier_name, classifier) as gradcam:
        activation = gradcam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(predicted_id)],
        )[0]
    activation = np.asarray(
        Image.fromarray(activation).resize(image.size, Image.Resampling.BILINEAR)
    )
    base = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = np.stack(
        [activation, np.clip(1.0 - np.abs(2.0 * activation - 1.0), 0.0, 1.0), 1.0 - activation],
        axis=-1,
    )
    overlay = np.clip(0.55 * base + 0.45 * heatmap, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def run_two_stage(image: Image.Image, classifier_name: str, warning_threshold: float) -> None:
    detector, classifier, class_names, config, device = load_two_stage_models(classifier_name)
    detection = detector.predict(
        source=image,
        conf=config["detection"]["crop_confidence"],
        imgsz=config["detection"]["image_size"],
        device="0" if device.type == "cuda" else device.type,
        verbose=False,
    )[0]
    boxes = detection.boxes.xyxy.detach().cpu().numpy()
    if len(boxes):
        bounds = expanded_union_box(
            boxes, image.width, image.height, config["detection"]["crop_margin"]
        )
        crop_source = "Detected lesion region"
    else:
        bounds = (0, 0, image.width, image.height)
        crop_source = "Full-image fallback (no lesion detected)"

    crop = image.crop(bounds)
    transform = classification_transform(config["project"]["image_size"])
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = classifier(tensor).softmax(dim=1)[0]
    scores, indices = probabilities.topk(3)
    predicted_id = int(indices[0].item())
    prediction_confidence = float(scores[0].item())

    annotated = image.copy()
    if len(boxes):
        ImageDraw.Draw(annotated).rectangle(bounds, outline=(255, 80, 40), width=4)
    overlay = gradcam_overlay(crop, classifier, classifier_name, tensor, predicted_id)

    left, middle, right = st.columns(3)
    left.image(annotated, caption="Input and lesion region", width="stretch")
    middle.image(crop, caption=crop_source, width="stretch")
    right.image(overlay, caption="Grad-CAM explanation", width="stretch")

    st.subheader(class_names[predicted_id])
    st.metric("Classifier confidence", f"{prediction_confidence:.1%}")
    st.write("Alternative classifier predictions")
    for score, index in zip(scores[1:].tolist(), indices[1:].tolist(), strict=True):
        st.write(f"- {class_names[index]}: {score:.1%}")
    if prediction_confidence < warning_threshold:
        st.warning("Low-confidence result. Seek assessment from a qualified plant-health expert.")


def run_standalone(image: Image.Image, warning_threshold: float) -> None:
    detector, class_names, config, device = load_standalone_model()
    result = detector.predict(
        source=image,
        conf=config["detection"]["crop_confidence"],
        imgsz=config["detection"]["image_size"],
        device="0" if device.type == "cuda" else device.type,
        verbose=False,
    )[0]
    if not len(result.boxes):
        st.image(image, caption="Input image", width="stretch")
        st.error("Standalone YOLO did not detect a disease region, so no diagnosis was produced.")
        return

    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    order = np.argsort(confidences)[::-1]
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for rank, detection_index in enumerate(order):
        left, top, right, bottom = boxes[detection_index]
        color = (255, 80, 40) if rank == 0 else (255, 190, 40)
        draw.rectangle((left, top, right, bottom), outline=color, width=4 if rank == 0 else 2)
        label = f"{class_names[class_ids[detection_index]]} {confidences[detection_index]:.1%}"
        text_box = draw.textbbox((left, top), label)
        draw.rectangle(text_box, fill=color)
        draw.text((left, top), label, fill=(0, 0, 0))

    best = int(order[0])
    predicted_id = int(class_ids[best])
    prediction_confidence = float(confidences[best])
    left, right = st.columns((2, 1))
    left.image(annotated, caption="Standalone YOLO detections", width="stretch")
    with right:
        st.subheader(class_names[predicted_id])
        st.metric("Detection confidence", f"{prediction_confidence:.1%}")
        st.write(f"Detections: {len(order)}")
        if len(order) > 1:
            st.write("Additional detections")
            for detection_index in order[1:3]:
                st.write(
                    f"- {class_names[class_ids[detection_index]]}: "
                    f"{confidences[detection_index]:.1%}"
                )
        st.caption("Grad-CAM is available only for the two-stage classifier pipeline.")
    if prediction_confidence < warning_threshold:
        st.warning("Low-confidence result. Seek assessment from a qualified plant-health expert.")


def main() -> None:
    st.set_page_config(page_title="Plant Disease Assistant", page_icon="🌿", layout="wide")
    st.title("Plant Disease Assistant")
    st.caption(
        "Course prototype for preliminary screening only. Predictions are not agronomic advice."
    )

    approach = st.sidebar.selectbox(
        "Inference approach",
        (TWO_STAGE, STANDALONE),
    )
    classifier_name = "efficientnet_b0"
    if approach == TWO_STAGE:
        classifier_name = st.sidebar.selectbox(
            "Classifier",
            CLASSIFICATION_MODELS,
            index=CLASSIFICATION_MODELS.index("efficientnet_b0"),
            format_func=lambda name: name.replace("_", " ").title(),
        )
    confidence_threshold = st.sidebar.slider("Low-confidence warning", 0.0, 1.0, 0.60, 0.05)
    uploaded = st.file_uploader("Upload a plant image", type=("jpg", "jpeg", "png"))
    if uploaded is None:
        st.info("Upload a JPG or PNG image to run lesion localization and classification.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        if approach == TWO_STAGE:
            run_two_stage(image, classifier_name, confidence_threshold)
        else:
            run_standalone(image, confidence_threshold)
    except Exception as error:
        st.error(f"The prototype could not process this image: {error}")


if __name__ == "__main__":
    main()
