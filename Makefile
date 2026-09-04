UV_CACHE_DIR := $(CURDIR)/.uv-cache
PYTHON := .venv/bin/python
RUNTIME_ENV := XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib YOLO_CONFIG_DIR=$(CURDIR)/.cache

.PHONY: init setup kernel check-datasets check-models train-yolo train-yolo-lesion train-yolo-standalone prepare-crops train-classifiers train-pipeline evaluate gradcam notebook data prepare-data data-plantvillage data-plantdoc data-plantseg clean-cache

init:
	@if [ ! -x "$(PYTHON)" ]; then $(MAKE) setup; else echo "Python environment already exists; skipping setup"; fi
	$(MAKE) data
	$(MAKE) prepare-data
	$(MAKE) check-datasets
	$(MAKE) check-models

setup:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv venv --python 3.13
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip install --python $(PYTHON) -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name plant-disease-assistant --display-name "Python (plant-disease-assistant)"

kernel:
	$(PYTHON) -m ipykernel install --user --name plant-disease-assistant --display-name "Python (plant-disease-assistant)"

check-datasets:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/check_datasets.py

check-models:
	$(RUNTIME_ENV) TORCH_HOME=$(CURDIR)/models PYTHONPATH=src $(PYTHON) scripts/check_models.py

train-yolo:
	$(MAKE) train-yolo-lesion
	$(MAKE) train-yolo-standalone

train-yolo-lesion:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/train_yolo.py --mode lesion

train-yolo-standalone:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/train_yolo.py --mode disease

prepare-crops:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/prepare_classifier_crops.py

train-classifiers:
	$(RUNTIME_ENV) TORCH_HOME=$(CURDIR)/models PYTHONPATH=src $(PYTHON) scripts/train_classifiers.py

train-pipeline:
	$(MAKE) train-yolo
	$(MAKE) prepare-crops
	$(MAKE) train-classifiers

evaluate:
	$(RUNTIME_ENV) TORCH_HOME=$(CURDIR)/models PYTHONPATH=src $(PYTHON) scripts/evaluate_pipeline.py

gradcam:
	$(RUNTIME_ENV) TORCH_HOME=$(CURDIR)/models PYTHONPATH=src $(PYTHON) scripts/run_gradcam.py

notebook:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) -m jupyter lab

data:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset all

prepare-data:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/prepare_datasets.py

data-plantseg:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantseg

data-plantvillage:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantvillage

data-plantdoc:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantdoc

clean-cache:
	rm -rf .uv-cache
