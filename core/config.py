import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class SystemConfig:
    # Model Configurations
    BASE_DIR: Path = BASE_DIR
    POSE_MODEL_PATH: Path = BASE_DIR / "yolo26n-pose.pt"
    FALLBACK_POSE_MODEL_PATH: Path = BASE_DIR / "yolov8n-pose.pt"
    
    # Hand Detector Weights
    CHAMPION_MODEL_PATH: Path = BASE_DIR / "runs" / "detect" / "champion_model" / "weights" / "best.pt"
    ACTIVE_MODEL_PATH: Path = BASE_DIR / "runs" / "detect" / "active_model" / "weights" / "best.pt"
    YOLO26L_BASE_PATH: Path = BASE_DIR / "yolo26l.pt"
    
    # Classes - exactly two classes
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
        0: "gloved_hand",
        1: "bare_hand"
    })
    
    # Hardware & Inference
    DEVICE: str = "0" if torch.cuda.is_available() else "cpu"
    IMGSZ: int = 640
    HALF_PRECISION: bool = True # 1660Ti FP16 3x GPU acceleration
    
    # Detection Thresholds
    HAND_CONF_THRESHOLD: float = 0.25
    BARE_VIOLATION_CONF_THRESHOLD: float = 0.70  # Higher bar for violation confirmation
    WRIST_CONF_THRESHOLD: float = 0.20
    PERSON_POSE_CONF_THRESHOLD: float = 0.25
    
    # Association Parameters
    ASSOCIATION_DISTANCE_MAX_PX: float = 240.0
    ASSOCIATION_CONFIDENCE_MIN: float = 0.45
    RIGHT_HAND_RULE_DEFAULT: bool = True  # Right hand mandatory, left exempt
    
    # Temporal Compliance Parameters
    BARE_PERSISTENCE_SECONDS: float = 1.8
    MINIMUM_STABLE_FRAMES: int = 20
    VIOLATION_COOLDOWN_SECONDS: float = 15.0
    TRACK_EXPIRY_SECONDS: float = 5.0
    
    # ByteTrack Parameters
    TRACK_HIGH_THRESH: float = 0.45
    TRACK_LOW_THRESH: float = 0.15
    TRACK_BUFFER_FRAMES: int = 45
    MATCH_IOU_THRESH: float = 0.75
    
    # RTSP Streaming
    DEFAULT_RTSP_URL: str = "rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/301"
    RTSP_TRANSPORT: str = "tcp"
    RTSP_RECONNECT_INTERVAL_SEC: float = 2.0
    DISPLAY_WIDTH: int = 1280
    DISPLAY_HEIGHT: int = 720
    
    # Database
    DATABASE_TYPE: str = os.getenv("DB_TYPE", "sqlite") # "sqlite" or "postgres"
    SQLITE_PATH: Path = BASE_DIR / "gloves_system.db"
    POSTGRES_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gloves_db")
    
    # Directories
    EVIDENCE_DIR: Path = BASE_DIR / "evidence"
    SNAPSHOTS_DIR: Path = BASE_DIR / "cctv_dataset" / "images" / "cctv_snapshots"
    ACTIVE_DATASET_DIR: Path = BASE_DIR / "active_learning_dataset"
    CCTV_DATASET_DIR: Path = BASE_DIR / "cctv_dataset"
    CPPE_DATASET_DIR: Path = BASE_DIR / "CPPE-Dataset"
    VIDEOS_DIR: Path = BASE_DIR / "uploads" / "videos"
    
    def resolve_hand_model_path(self) -> Path:
        """Returns the best available hand detector weights."""
        if self.CHAMPION_MODEL_PATH.exists():
            return self.CHAMPION_MODEL_PATH
        if self.ACTIVE_MODEL_PATH.exists():
            return self.ACTIVE_MODEL_PATH
        candidates = list(self.BASE_DIR.glob("runs/detect/**/weights/best.pt"))
        if candidates:
            return candidates[0]
        return self.YOLO26L_BASE_PATH

    def resolve_pose_model_path(self) -> Path:
        """Returns the preferred pose weights (YOLO26L-Pose with fallback)."""
        if self.POSE_MODEL_PATH.exists():
            return self.POSE_MODEL_PATH
        if self.FALLBACK_POSE_MODEL_PATH.exists():
            return self.FALLBACK_POSE_MODEL_PATH
        return Path("yolo26l-pose.pt")

config = SystemConfig()
config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
config.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
