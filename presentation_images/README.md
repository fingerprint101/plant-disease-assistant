# Presentation images — pick what fits

- **gradcam_correct_high_iou.png** — best example: correct prediction, highest overlap (IoU 0.45)
  between Grad-CAM attention and the real lesion mask. Use for Slide 11 (Grad-CAM).
- **gradcam_correct_clean_background.png** — correct prediction, clean single-leaf photo, good if
  you want a simpler/less busy image than the one above.
- **gradcam_wrong_prediction_example.png** — a genuine failure case (wrong prediction, attention
  off-target). Useful if you want an honest "here's where it fails" slide instead of only showing
  wins.
- **lesion_detection_boxes.jpg** — a real validation batch mosaic (16 images) with the lesion
  detector's predicted boxes and confidence scores drawn on. Busy/crowded but shows real variety
  of plants and diseases. Crop to one tile if you want a cleaner single image for Slide 9.
- **robustness_accuracy_vs_severity.png** — clean 6-panel chart, accuracy vs corruption severity
  for all 3 classifiers, all 6 corruption types. Ready to use as-is for Slide 12.

All pulled directly from `outputs/` (gitignored, not tracked in the repo) — these copies are just
for convenience when building the slide deck.
