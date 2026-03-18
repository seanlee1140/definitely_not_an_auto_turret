# DETR Tennis Ball Detection Scaffold

This project keeps the same live-camera CLI shape as the YOLO app, but it intentionally fails fast because the current local folder at `../model/detr-finetuned-tennis-ball-v2/` is not a complete DETR object-detection checkpoint.

## Current status

- Validates `config.json` and `model.safetensors` before touching the camera loop.
- Refuses to run if class metadata or detection-head tensors are missing.
- Uses the same Jetson CSI camera flags as the YOLO app so the runtime shape is already in place.

## Why it fails today

The current checkpoint folder is missing:

- Class metadata such as `id2label`, `label2id`, and `num_labels`
- Detection-head tensors such as `class_labels_classifier`, `bbox_predictor`, `class_embed`, or `bbox_embed`

That means the folder contains backbone and transformer weights, but not the full object-detection head needed for live inference.

## Run

From the repository root:

```bash
python -m tennis_ball_detection.object_detection_DETR --sensor-id 0
```

You should currently see a validation error that explains why the checkpoint is incomplete.
