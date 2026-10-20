import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..config import config

class EvidenceRecorder:
    """
    Saves high-resolution annotated evidence snapshots when a violation is confirmed,
    saving to the evidence directory and creating audit trails.
    """
    def __init__(self, evidence_dir: Optional[Path] = None):
        self.evidence_dir = evidence_dir or config.EVIDENCE_DIR
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def capture_evidence(
        self,
        raw_frame: np.ndarray,
        track_id: int,
        camera_id: str,
        confidence: float,
        hand_box: List[float],
        model_version: str = "YOLO26L-v1"
    ) -> str:
        """
        Generates and writes an evidence snapshot with legal audit banner.
        Returns the saved file path string.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"violation_cam{camera_id}_trk{track_id}_{timestamp_str}.jpg"
        filepath = self.evidence_dir / filename

        annotated = raw_frame.copy()
        h, w = annotated.shape[:2]

        # Draw red alert bounding box on bare hand
        x1, y1 = max(0, int(hand_box[0])), max(0, int(hand_box[1]))
        x2, y2 = min(w, int(hand_box[2])), min(h, int(hand_box[3]))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 240), 4)

        # Audit banner at bottom
        cv2.rectangle(annotated, (0, h - 55), (w, h), (15, 18, 25), -1)
        cv2.line(annotated, (0, h - 55), (w, h - 55), (0, 0, 240), 2)

        audit_text = f"EVIDENCE [CONFIRMED VIOLATION] | CAM-{camera_id} | TRACK ID: {track_id} | CONF: {confidence*100:.1f}% | {now.strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(annotated, audit_text, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[EvidenceRecorder] Saved verified evidence snapshot: {filepath}")
        return str(filepath)
