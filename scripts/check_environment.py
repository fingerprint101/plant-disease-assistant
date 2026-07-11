"""Check the Python environment and report available project data."""

from __future__ import annotations

import importlib
import platform
import sys

from plant_disease.paths import RAW_DIR, SAMPLE_DIR, ensure_project_directories

PACKAGES = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("PIL", "Pillow"),
    ("pyarrow", "PyArrow"),
    ("sklearn", "scikit-learn"),
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "Ultralytics"),
    ("pytorch_grad_cam", "Grad-CAM"),
    ("jupyterlab", "JupyterLab"),
]


def main() -> None:
    ensure_project_directories()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    failures = []
    for module_name, display_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[ok] {display_name}: {version}")
        except Exception as error:
            failures.append(display_name)
            print(f"[missing] {display_name}: {error}")

    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device = "mps (Apple Metal)"
        else:
            device = "cpu"
        print(f"Selected training device: {device}")
    except ImportError:
        pass

    sample_images = list(SAMPLE_DIR.rglob("*.jpg")) + list(SAMPLE_DIR.rglob("*.JPG"))
    raw_images = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.JPG"))
    print(f"Bundled sample images: {len(sample_images)}")
    print(f"Extracted raw images: {len(raw_images)}")

    if failures:
        raise SystemExit("Environment check failed. Run `make setup` before continuing.")


if __name__ == "__main__":
    main()
