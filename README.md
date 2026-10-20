# Gloves Compliance & Detection System 🧤

An automated, real-time Computer Vision system for jewelry sales counters and high-security retail environments to detect and enforce hand glove compliance using state-of-the-art YOLO models and human pose estimation.

---

## 🌟 Key Features

- **Dual-Model Perception Pipeline**:
  - **Pose Estimation**: YOLO26L-Pose / YOLO26N-Pose / YOLOv8N-Pose for human body keypoint detection and wrist tracking.
  - **Hand State Classifier**: Fine-tuned YOLO26L / YOLOv8s detecting `gloved_hand` vs `bare_hand` with high-confidence thresholds.
- **Hand-to-Wrist Spatial Association**:
  - Associates detected hands with the closest staff member's wrists.
  - Enforces operational policies (e.g., Mandatory Right-Hand Rule).
- **Temporal State Tracking & Compliance Engine**:
  - Integrated with **ByteTrack** for persistent multi-staff identity tracking.
  - Temporal state smoothing prevents false alarms from brief occlusions.
  - Automated violation event generation with cooldown periods.
- **Active Learning & Labeler Studio**:
  - Web-based labeling studio for active learning and dataset expansion.
  - Automated champion/challenger model training and fine-tuning pipelines.
- **Audit & Evidence Management**:
  - Automated high-resolution snapshot capture with audit overlay banners.
  - Persistent storage in SQLite / PostgreSQL with complete violation logging.
- **Interactive Web Dashboards**:
  - Live detector web interface (`detector_app.py`) with RTSP/video streaming.
  - Labeler studio web interface (`labeler_studio.py`).

---

## 📁 Repository Structure

```text
├── core/
│   ├── association/       # Spatial wrist-to-hand association algorithms
│   ├── config.py          # Unified system configuration & model path resolvers
│   ├── database/          # Database models & service layer (SQLite / Postgres)
│   ├── detector/          # YOLO hand detector & pose estimator modules
│   ├── evidence/          # Violation evidence snapshot capture
│   ├── pipeline.py        # End-to-end vision inference pipeline
│   ├── temporal_logic/    # Temporal smoothing & state machine
│   ├── tracking/          # ByteTrack multi-object tracking integration
│   ├── training/          # Dataset manager & model training orchestrators
│   └── violation/         # Violation evaluation & alert dispatch
├── runs/
│   └── detect/            # Training runs, validation plots, and weights
│       ├── active_model/  # Currently active fine-tuned weights
│       └── champion_model/# Champion production model weights
├── static/                # Web styling, CSS, and UI assets
├── templates/             # HTML templates for the detector dashboard
├── tools/                 # CCTV streaming, RTSP tools, and utilities
├── detector_app.py        # Main web application & live CCTV detection server
├── labeler_studio.py      # Active learning labeling interface
├── train_gloves.py        # Model training script
├── finetune_active_model.py # Automated finetuning loop
├── evaluate_system.py     # Evaluation & benchmark script
├── cctv_inference.py      # Standalone CCTV stream inference script
├── yolo26l-pose.pt        # Base YOLO26L pose estimation weights
├── yolo26l.pt             # Base YOLO26L detection weights
├── yolo26n-pose.pt        # Base YOLO26N pose weights
├── yolov8n-pose.pt        # Fallback YOLOv8N pose weights
└── yolov8s.pt             # Base YOLOv8S weights
```

---

## 🚀 Getting Started

### 1. Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: RTX / GTX series with FP16 support)
- PyTorch with CUDA support

### 2. Installation

```bash
git clone https://github.com/HABEEL007/GLOVES-.git
cd GLOVES-
pip install -r requirements.txt  # or install torch torchvision ultralytics opencv-python flask
```

### 3. Running the Live Detector

```bash
python detector_app.py
```
Open your browser and navigate to `http://localhost:5000` to view the live detection feed, configure cameras, and monitor violation alerts.

### 4. Running the Labeler Studio (Active Learning)

```bash
python labeler_studio.py
```

---

## 🎯 Model Zoo Included

All model weights are included directly in this repository:
- **Champion Model**: `runs/detect/champion_model/weights/best.pt`
- **Active Fine-Tuned Model**: `runs/detect/active_model/weights/best.pt`
- **Candidate Models**: `runs/detect/candidate_yolo26l_*/weights/best.pt`
- **Pose Estimators**: `yolo26l-pose.pt`, `yolo26n-pose.pt`, `yolov8n-pose.pt`
- **Base Detectors**: `yolo26l.pt`, `yolov8s.pt`
