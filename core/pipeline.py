import time
import math
import threading
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import config
from .pose.pose_detector import YOLO26PoseDetector, PoseDetectionResult
from .tracking.byte_tracker import PersonByteTracker, PersonTrack
from .hand_detection.hand_detector import YOLO26HandDetector, HandDetectionBox
from .association.hand_association import HandAssociationEngine, AssociatedHand
from .violation.violation_manager import ViolationManager
from .database.db_service import db_service
from .active_learning.model_registry import ModelRegistry

class GloveCompliancePipeline:
    """
    Unified end-to-end inference & compliance pipeline:
    RTSP Frame -> YOLO26L-Pose -> ByteTrack -> Right Wrist ->
    YOLO26L Hand Detector -> Person Association -> Temporal Verification ->
    Violation Trigger -> Evidence Capture -> Database -> Dashboard.
    """
    def __init__(self, camera_id: str = "301"):
        self.camera_id = camera_id
        
        # Core Models & Processors
        self.pose_detector = YOLO26PoseDetector()
        self.tracker = PersonByteTracker()
        self.hand_detector = YOLO26HandDetector()
        self.association_engine = HandAssociationEngine()
        self.violation_manager = ViolationManager()
        self.registry = ModelRegistry()
        
        self.model_version = self.registry.get_registry().get("active_champion", "YOLO26L-Pose-ByteTrack-v1")
        self.right_hand_only_rule = config.RIGHT_HAND_RULE_DEFAULT
        self.conf_threshold = config.HAND_CONF_THRESHOLD
        
        # Telemetry tracking
        self.prev_time = time.time()
        self.fps_buffer = []
        self.known_tracks = set()
        self.last_track_sync = time.time()

        # Pose cadence optimization: run pose estimation every 2 frames
        self.pose_stride = 2
        self.cached_pose_results = []
        self.cached_active_tracks: List[PersonTrack] = []
        self.cached_dt_pose = 0.0
        self.cached_dt_track = 0.0

    def set_hand_conf(self, conf: float):
        self.conf_threshold = max(0.05, min(0.95, float(conf)))

    def set_rule_mode(self, right_hand_only: bool):
        self.right_hand_only_rule = bool(right_hand_only)

    def reload_hand_detector(self, weights_path):
        self.hand_detector.reload_weights(weights_path)
        self.model_version = Path(weights_path).stem

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        img_h, img_w = frame.shape[:2]

        # 1. Pose Estimation & 2. ByteTrack Tracking (with cadence skip)
        run_pose = (frame_number % self.pose_stride == 0) or (not self.cached_active_tracks)
        if run_pose:
            t_pose_0 = time.time()
            pose_results = self.pose_detector.predict(frame)
            dt_pose = time.time() - t_pose_0

            t_track_0 = time.time()
            active_tracks = self.tracker.update(pose_results)
            dt_track = time.time() - t_track_0

            self.cached_pose_results = pose_results
            self.cached_active_tracks = active_tracks
            self.cached_dt_pose = dt_pose
            self.cached_dt_track = dt_track
        else:
            pose_results = self.cached_pose_results
            active_tracks = self.cached_active_tracks
            dt_pose = 0.001
            dt_track = 0.0005

        # Record new active tracks to DB asynchronously/on-discovery, avoiding per-frame disk locks
        now_ts = time.time()
        for trk in active_tracks:
            if trk.track_id not in self.known_tracks or (now_ts - getattr(self, "last_track_sync", 0)) > 10.0:
                self.known_tracks.add(trk.track_id)
                self.last_track_sync = now_ts
                threading.Thread(target=db_service.record_track, args=(trk.track_id, self.camera_id), daemon=True).start()

        # 3. Hand Detection (Gloved vs Bare Hand - Full Frame Single-Pass for Maximum Speed & Smoothness)
        t_det_0 = time.time()
        hand_boxes = self.hand_detector.predict(frame, conf_threshold=self.conf_threshold)
        dt_det = time.time() - t_det_0

        # 4. Person-Aware Geometric Association
        t_assoc_0 = time.time()
        associated_hands = self.association_engine.associate_hands(
            hands=hand_boxes,
            tracks=active_tracks,
            img_width=img_w,
            img_height=img_h
        )
        dt_assoc = time.time() - t_assoc_0

        # 5. Temporal Compliance & Business Rule Evaluation
        active_track_ids = [t.track_id for t in active_tracks]
        compliance_eval = self.violation_manager.evaluate_compliance(
            associated_hands=associated_hands,
            tracked_ids=active_track_ids,
            raw_frame=frame,
            camera_id=self.camera_id,
            model_version=self.model_version
        )

        # 6. Database Logging (Detections periodically every 10 frames to optimize disk I/O)
        if frame_number % 10 == 0:
            db_dets = []
            for ah in associated_hands:
                db_dets.append({
                    "track_id": ah.track_id,
                    "class_id": ah.class_id,
                    "conf": ah.confidence,
                    "box": ah.box,
                    "association_confidence": ah.association_confidence,
                    "wrist_distance": ah.wrist_distance
                })
            threading.Thread(target=db_service.record_detections, args=(db_dets, frame_number, self.camera_id), daemon=True).start()

        # 7. Render Display Frame (720p HD with telemetry & HUD)
        annotated_frame, telemetry = self._render_hud(
            frame=frame,
            active_tracks=active_tracks,
            associated_hands=associated_hands,
            compliance=compliance_eval,
            t0=t0,
            dt_pose=dt_pose,
            dt_det=dt_det,
            dt_track=dt_track
        )

        # Collect detected gloves for dataset auto-capture and reporting
        detected_gloves = []
        for ah in associated_hands:
            if ah.class_id == 0:
                g_col = getattr(ah.hand_box, "glove_color", "unknown")
                detected_gloves.append({
                    "box": [float(c) for c in ah.box],
                    "conf": float(ah.confidence),
                    "side": str(ah.side),
                    "color": g_col,
                    "track_id": ah.track_id
                })

        stats = {
            "status": compliance_eval["status_text"],
            "badge_type": compliance_eval["badge_type"],
            "active_violations": compliance_eval["active_violations"],
            "right_hand_violations": compliance_eval["active_violations"],
            "total_violations_today": compliance_eval["total_violations_today"],
            "right_gloves_count": compliance_eval["right_gloves_count"],
            "left_hand_bare_count": compliance_eval["left_hand_bare_count"],
            "left_hand_count": compliance_eval["left_hand_bare_count"],
            "detected_gloves": detected_gloves,
            "has_gloves": len(detected_gloves) > 0,
            "has_green_gloves": any(g["color"] == "green" for g in detected_gloves),
            "gloves_count": len(detected_gloves),
            "fps": round(telemetry["fps"], 1),
            "latency_ms": round(telemetry["latency_ms"], 1),
            "pose_ms": round(dt_pose * 1000, 1),
            "det_ms": round(dt_det * 1000, 1),
            "gpu_ms": round((dt_pose + dt_det) * 1000, 1),
            "track_ms": round(dt_track * 1000, 1),
            "tracked_people": len(active_tracks),
            "tracks": [
                {
                    "track_id": trk.track_id,
                    "state": compliance_eval["track_states"].get(trk.track_id, "OBSERVING")
                }
                for trk in active_tracks
            ],
            "model_version": self.model_version,
            "channel": self.camera_id,
            "rule_mode": "Either Hand Glove Compliant (Right or Left)",
            "recent_evidence_path": compliance_eval["recent_evidence_path"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return annotated_frame, stats

    def _render_hud(
        self,
        frame: np.ndarray,
        active_tracks: List[PersonTrack],
        associated_hands: List[AssociatedHand],
        compliance: Dict[str, Any],
        t0: float,
        dt_pose: float,
        dt_det: float,
        dt_track: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        img_h, img_w = frame.shape[:2]
        disp_w, disp_h = config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
        scale_x = disp_w / img_w
        scale_y = disp_h / img_h
        
        canvas = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # 1. Draw Person Track Bounding Boxes and Wrist Points
        for trk in active_tracks:
            p_box = trk.box
            px1, py1 = int(p_box[0] * scale_x), int(p_box[1] * scale_y)
            px2, py2 = int(p_box[2] * scale_x), int(p_box[3] * scale_y)
            
            # Person boundary in subtle blue
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (180, 150, 60), 2)
            
            # Track ID Banner
            trk_label = f"ID: {trk.track_id}"
            (tw, th), _ = cv2.getTextSize(trk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (px1, py1 - th - 6), (px1 + tw + 6, py1), (180, 150, 60), -1)
            cv2.putText(canvas, trk_label, (px1 + 3, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Wrists
            if trk.pose_data:
                rw = trk.pose_data.right_wrist
                if rw:
                    rwx, rwy = int(rw["x"] * scale_x), int(rw["y"] * scale_y)
                    cv2.circle(canvas, (rwx, rwy), 5, (0, 140, 255), -1)
                    cv2.putText(canvas, "R-Wrist", (rwx + 6, rwy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1)

        # Check which workers are wearing a compliant glove on either hand
        gloved_track_ids = set()
        for ah in associated_hands:
            if ah.class_id == 0 and ah.track_id is not None:
                color = getattr(ah.hand_box, "glove_color", "unknown")
                if color in ("black", "green", "unknown"):
                    gloved_track_ids.add(ah.track_id)

        # 2. Draw Hand Detections with Compliance Colors
        for ah in associated_hands:
            hx1 = max(0, int(ah.box[0] * scale_x))
            hy1 = max(0, int(ah.box[1] * scale_y))
            hx2 = min(disp_w, int(ah.box[2] * scale_x))
            hy2 = min(disp_h, int(ah.box[3] * scale_y))

            # Color logic based on compliance + glove color
            glove_color = getattr(ah.hand_box, "glove_color", "unknown")
            if ah.side == "UNCERTAIN":
                color = (160, 160, 160)  # Neutral Gray
                tag = f"? Uncertain {ah.confidence*100:.0f}%"
            elif ah.class_id == 0:  # gloved_hand detected
                if glove_color == "green":
                    color = (34, 197, 94)   # Bright Green = GREEN GLOVE OK
                    tag = f"[GREEN GLOVE OK] {ah.side} {ah.confidence*100:.0f}%"
                elif glove_color == "black":
                    color = (34, 197, 94)   # Bright Green = BLACK GLOVE OK
                    tag = f"[BLACK GLOVE OK] {ah.side} {ah.confidence*100:.0f}%"
                elif glove_color == "other":
                    color = (0, 165, 255)   # Orange = WRONG COLOR
                    tag = f"[WRONG COLOR] {ah.side} Glove {ah.confidence*100:.0f}% - Compliant Required"
                else:
                    color = (34, 197, 94)   # Default green if color unknown
                    tag = f"[GLOVE OK] {ah.side} {ah.confidence*100:.0f}%"
            else:  # bare_hand
                # If this person already has a glove on their other hand -> EXEMPT!
                if ah.track_id is not None and ah.track_id in gloved_track_ids:
                    color = (240, 180, 45)  # Soft amber = OK because other hand has glove!
                    tag = f"[EXEMPT] {ah.side} Hand (Other Hand Gloved)"
                else:
                    color = (0, 0, 240)     # Bright Red Alert = No gloves on either hand
                    tag = f"[ALERT] BARE {ah.side} HAND {ah.confidence*100:.0f}%"

            border_w = 4 if (ah.class_id == 1 and (ah.track_id is None or ah.track_id not in gloved_track_ids)) else 2
            cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), color, border_w)

            # Tag Box
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            tag_y1 = max(0, hy1 - th - 6)
            cv2.rectangle(canvas, (hx1, tag_y1), (hx1 + tw + 8, hy1), color, -1)
            cv2.putText(canvas, tag, (hx1 + 4, hy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top Telemetry Header
        cv2.rectangle(canvas, (0, 0), (disp_w, 42), (15, 18, 26), -1)
        cv2.line(canvas, (0, 42), (disp_w, 42), (50, 60, 75), 1)

        cv2.circle(canvas, (20, 21), 6, (0, 0, 240), -1)
        cv2.putText(canvas, f"LIVE CH-{self.camera_id}", (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"| YOLO26L + ByteTrack | EITHER-HAND GLOVE RULE (Right or Left)", (180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 240, 255), 1, cv2.LINE_AA)

        # FPS calculation
        curr_time = time.time()
        frame_fps = 1.0 / max(curr_time - self.prev_time, 1e-4)
        self.prev_time = curr_time
        self.fps_buffer.append(frame_fps)
        if len(self.fps_buffer) > 15:
            self.fps_buffer.pop(0)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

        total_latency_ms = (time.time() - t0) * 1000
        telemetry_str = f"ZERO-LAG: {total_latency_ms:.0f}ms | GPU: {(dt_pose+dt_det)*1000:.0f}ms | {avg_fps:.1f} FPS"
        (tw_t, _), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(canvas, telemetry_str, (disp_w - tw_t - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)

        # 4. Status Banner Pill
        pill_w = 560
        hud_bg = (20, 120, 35) if compliance["badge_type"] == "success" else ((0, 0, 200) if compliance["badge_type"] == "danger" else (25, 30, 40))
        cv2.rectangle(canvas, (15, 52), (15 + pill_w, 94), hud_bg, -1)
        cv2.rectangle(canvas, (15, 52), (15 + pill_w, 94), (255, 255, 255), 1)
        cv2.putText(canvas, compliance["status_text"], (24, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas, {"fps": avg_fps, "latency_ms": total_latency_ms}
