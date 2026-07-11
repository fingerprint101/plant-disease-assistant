# Dataset Sample

This directory contains a small subset of the data inspected for the project. It is provided for team discussion and visual inspection, not as a complete training set.

## PlantVillage

- 24 images in six class directories.
- Four images are included for each class.
- The directory name is the classification label.
- Source: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
- License stated by the source repository: CC BY-SA 3.0.

## PlantDoc

- 12 field images covering six selected disease labels.
- `annotations.json` contains source identifiers, image dimensions and bounding boxes in `[x, y, width, height]` format.
- `labels/*.txt` contains normalized YOLO labels in `class x_center y_center width height` format.
- `classes.txt` defines the class order used by the YOLO labels.
- Original source: [PlantDoc Object Detection Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset).
- Downloaded representation: [PlantDoc Hugging Face mirror](https://huggingface.co/datasets/agyaatcoder/PlantDoc).

## Manifest

`manifest.csv` lists the dataset, relative path, original split, labels and number of boxes for every included image.

The PlantDoc sample was restricted to rows with decodable images and bounding boxes inside the decoded image boundaries. This does not replace the full deduplication and grouped split required before model training.
