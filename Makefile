UV_CACHE_DIR := $(CURDIR)/.uv-cache
PYTHON := .venv/bin/python
RUNTIME_ENV := XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib YOLO_CONFIG_DIR=$(CURDIR)/.cache

.PHONY: setup kernel check notebook data data-plantvillage data-plantdoc data-plantseg clean-cache

setup:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv venv --python 3.13
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip install --python $(PYTHON) -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name plant-disease-assistant --display-name "Python (plant-disease-assistant)"

kernel:
	$(PYTHON) -m ipykernel install --user --name plant-disease-assistant --display-name "Python (plant-disease-assistant)"

check:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/check_environment.py

notebook:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) -m jupyter lab

data:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset all

data-plantvillage:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantvillage

data-plantdoc:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantdoc

data-plantseg:
	$(RUNTIME_ENV) PYTHONPATH=src $(PYTHON) scripts/download_datasets.py --dataset plantseg

clean-cache:
	rm -rf .uv-cache
