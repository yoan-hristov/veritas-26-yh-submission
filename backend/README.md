---
title: Veritas AI Chest Radiograph Consensus Cascade
emoji: ❎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# 🫁 Veritas AI: Chest Radiograph Consensus Cascade

Clinical second-reader decision support service powered by:
- 5 ResNet-18 architectures (Parts I–V)
- Google Gemini Decision 1 second-reader consensus
- Automated scope & diagnostic quality audits
- Temperature-calibrated thresholds & explainable Grad-CAM spatial attributions

### Endpoints Supported:
- **Interactive UI**: Web GUI powered by Gradio at root `/`
- **Uptime Health**: `GET /health`
- **Model Metadata**: `GET /models`
- **Inference Analysis**: `POST /analyze` (Default Model 5) and `POST /analyze/{model_id}` (Models 1–5)
