import math
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ultralytics import YOLO

from ..config import config

class HandDetectionBox:
    def __init__(
        self,
        box: List[float], # [x1, y1, x2, y2]
        class_id: int,    # 0 = gloved_hand, 1 = bare_hand
        confidence: float,
        class_name: str,
        glove_color: str = "unknown"  # "black", "other", "unknown"
    ):
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name
        self.glove_color = glove_color  # Color of detected glove
        self.cx = (box[0] + box[2]) / 2.0
        self.cy = (box[1] + box[3]) / 2.0
        self.width = max(1.0, box[2] - box[0])
        self.height = max(1.0, box[3] - box[1])


def evaluate_hand_color(frame: np.ndarray, box: List[float]) -> str:
    """
    Analyzes the hand region to accurately distinguish:
    - 'black': Black / Dark Glove
    - 'bare': Human skin / Bare hand
    - 'other': Non-black glove (blue, white, nitrile)
    - 'unknown': If crop is invalid or too small
    """
    try:
        img_h, img_w = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))

        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return "unknown"

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # Sample central 70% of crop to exclude sleeve background
        cy1, cy2 = int(ch * 0.15), int(ch * 0.85)
        cx1, cx2 = int(cw * 0.15), int(cw * 0.85)
        center = crop[cy1:cy2, cx1:cx2] if (cy2 - cy1 >= 8 and cx2 - cx1 >= 8) else crop

        # --- SPEED OPTIMIZATION: Downsample to max 48x48 before color math ---
        # Color classification only needs rough pixel statistics — detail is wasteful.
        h_, w_ = center.shape[:2]
        if h_ > 48 or w_ > 48:
            scale = 48.0 / max(h_, w_)
            center = cv2.resize(center, (max(1, int(w_ * scale)), max(1, int(h_ * scale))),
                                interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(center, cv2.COLOR_BGR2YCrCb)

        # Skin tone: human skin has high Cr and moderate Cb with brightness
        skin_mask = (ycrcb[:,:,1] >= 133) & (ycrcb[:,:,1] <= 173) & (ycrcb[:,:,2] >= 77) & (ycrcb[:,:,2] <= 127) & (ycrcb[:,:,0] > 75) & (hsv[:,:,2] > 70)
        skin_ratio = float(np.mean(skin_mask))

        # Green Glove Detection: HSV Hue 32-88 with saturation, or Green channel dominant
        green_hsv = (hsv[:,:,0] >= 32) & (hsv[:,:,0] <= 88) & (hsv[:,:,1] > 35) & (hsv[:,:,2] > 35)
        green_bgr = (center[:,:,1] > center[:,:,0] * 1.15) & (center[:,:,1] > center[:,:,2] * 1.15) & (center[:,:,1] > 40)
        green_ratio = float(np.mean(green_hsv | green_bgr))

        # Dark / Black Glove: V < 125, S < 140 or BGR all < 125
        dark_hsv = (hsv[:,:,2] < 125) & (hsv[:,:,1] < 140)
        dark_rgb = (center[:,:,0] < 125) & (center[:,:,1] < 125) & (center[:,:,2] < 125)
        dark_ratio = float(np.mean(dark_hsv | dark_rgb))

        if green_ratio >= 0.18:
            return "green"
        elif dark_ratio >= 0.28 and (skin_ratio < 0.35 or dark_ratio > skin_ratio * 1.3):
            return "black"
        elif skin_ratio >= 0.35:
            return "bare"
        else:
            return "other"
    except Exception:
        return "unknown"

def detect_glove_color(frame: np.ndarray, box: List[float], black_pixel_ratio_threshold: float = 0.35) -> str:
    """Backward compatibility wrapper."""
    return evaluate_hand_color(frame, box)

class YOLO26HandDetector:
    """
    YOLO26L Custom Hand Detector specializing in gloved_hand (0) and bare_hand (1),
    with robust support for black gloves, dark clothing hard negatives, and adaptive ROIs.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.resolve_hand_model_path()
        print(f"[YOLO26HandDetector] Loading Hand Detector from: {self.model_path} on device {config.DEVICE}")
        self.model = YOLO(str(self.model_path))

    def reload_weights(self, new_path: Path):
        """Safely reloads new champion weights without downtime."""
        print(f"[YOLO26HandDetector] Reloading model weights from: {new_path}")
        self.model = YOLO(str(new_path))
        self.model_path = new_path

    def predict(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[HandDetectionBox]:
        """
        Runs object detection on full frame with robust black glove color verification.
        """
        conf = conf_threshold if conf_threshold is not None else config.HAND_CONF_THRESHOLD
        res = self.model.predict(
            source=frame,
            device=config.DEVICE,
            conf=conf,
            imgsz=config.IMGSZ,
            verbose=False,
            half=config.HALF_PRECISION
        )[0]

        detections = []
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            cid = int(classes[i])
            c_score = float(confs[i])
            b = boxes[i].tolist()

            # Analyze hand crop color
            hand_color = evaluate_hand_color(frame, b)

            # Robust Glove Recognition:
            # If the detected hand region is dark/black or green, it is a GLOVE!
            if hand_color == "green":
                final_cid = 0
                final_cname = "gloved_hand"
                glove_color = "green"
            elif hand_color == "black":
                final_cid = 0
                final_cname = "gloved_hand"
                glove_color = "black"
            elif cid == 0:
                final_cid = 0
                final_cname = "gloved_hand"
                glove_color = hand_color if hand_color in ("black", "green") else "other"
            else:
                final_cid = 1
                final_cname = "bare_hand"
                glove_color = "unknown"

            detections.append(HandDetectionBox(
                box=b,
                class_id=final_cid,
                confidence=c_score,
                class_name=final_cname,
                glove_color=glove_color
            ))

        return detections

    def predict_adaptive_roi(
        self,
        frame: np.ndarray,
        wrist_pt: Dict[str, float],
        person_box: List[float],
        conf_threshold: Optional[float] = None
    ) -> List[HandDetectionBox]:
        """
        Adaptive ROI search around wrist for small or distant hands (Requirement #17).
        Scales with person bounding box height.
        """
        img_h, img_w = frame.shape[:2]
        p_h = max(20.0, person_box[3] - person_box[1])
        
        # Adaptive search radius proportional to person scale (approx 25% of body height)
        radius = max(80.0, min(300.0, p_h * 0.28))
        
        x1 = max(0, int(wrist_pt["x"] - radius))
        y1 = max(0, int(wrist_pt["y"] - radius))
        x2 = min(img_w, int(wrist_pt["x"] + radius))
        y2 = min(img_h, int(wrist_pt["y"] + radius))

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            return []

        roi = frame[y1:y2, x1:x2]
        roi_detections = self.predict(roi, conf_threshold=conf_threshold)

        # Map ROI coords back to global frame coordinates
        mapped_detections = []
        for det in roi_detections:
            g_box = [
                det.box[0] + x1,
                det.box[1] + y1,
                det.box[2] + x1,
                det.box[3] + y1
            ]
            mapped_detections.append(HandDetectionBox(
                box=g_box,
                class_id=det.class_id,
                confidence=det.confidence,
                class_name=det.class_name,
                glove_color=det.glove_color
            ))

        return mapped_detections
