---
title: VisionGuard Waste Classifier
emoji: "♻️"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: MobileNetV2 waste image classifier (TrashNet), FastAPI
---

# VisionGuard (FastAPI service)

Fine-tuned MobileNetV2 trained on TrashNet (6 classes). Held-out test macro-F1: 89.75%.

- `GET /health` reports `"ready": true` once the model is loaded.
- `POST /predict` takes a multipart image upload (max 5 MiB) and returns the top-3 scores and an `abstained` flag.
- Interactive docs: `/docs`.

Source code, evaluation and tests: https://github.com/AryaShah800/visionguard

Trained on the TrashNet dataset (Yang and Thung, Stanford). The free CPU tier sleeps when idle, so the first request after a pause can be slow.
