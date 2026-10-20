import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class CameraRecord:
    camera_id: str
    name: str
    stream_url: str
    location: str = "Gold Store Sales Counter"
    is_active: bool = True

@dataclass
class StaffTrackRecord:
    track_id: int
    camera_id: str
    first_seen: str
    last_seen: str
    is_staff: bool = True
    is_active: bool = True

@dataclass
class DetectionRecord:
    detection_id: Optional[int]
    frame_number: int
    camera_id: str
    track_id: Optional[int]
    class_id: int # 0=gloved_hand, 1=bare_hand
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    assigned_side: str # "RIGHT", "LEFT", "UNCERTAIN"
    association_confidence: float
    wrist_distance: float
    timestamp: str

@dataclass
class HandStateRecord:
    track_id: int
    side: str # "RIGHT" or "LEFT"
    state: str # "UNKNOWN", "OBSERVING", "GLOVED", "BARE_CANDIDATE", "CONFIRMING", "VIOLATION", "UNCERTAIN"
    confidence: float
    first_seen: str
    last_seen: str
    stable_frames: int
    bare_duration: float
    last_transition: str

@dataclass
class ViolationEventRecord:
    violation_id: Optional[int]
    camera_id: str
    track_id: int
    start_time: str
    end_time: Optional[str]
    confidence: float
    violation_type: str # "BARE_RIGHT_HAND"
    evidence_path: str
    model_version: str
    status: str = "CONFIRMED" # "CONFIRMED", "RESOLVED", "DISMISSED"

@dataclass
class EvidenceRecord:
    evidence_id: Optional[int]
    violation_id: int
    image_path: str
    width: int
    height: int
    metadata_json: str
    created_at: str

@dataclass
class ModelVersionRecord:
    model_version: str
    training_run_id: str
    architecture: str
    image_size: int
    epochs: int
    batch_size: int
    precision: float
    recall: float
    map50: float
    map50_95: float
    f1: float
    right_hand_acc: float
    false_violation_rate: float
    created_at: str
    approved_at: Optional[str] = None
    is_champion: bool = False
