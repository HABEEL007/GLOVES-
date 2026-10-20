import math
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from ultralytics import YOLO
from ..config import config

class PoseDetectionResult:
    def __init__(
        self,
        person_idx: int,
        person_box: List[float],
        person_conf: float,
        left_wrist: Optional[Dict[str, float]] = None,
        right_wrist: Optional[Dict[str, float]] = None,
        left_shoulder: Optional[Dict[str, float]] = None,
        right_shoulder: Optional[Dict[str, float]] = None,
        left_elbow: Optional[Dict[str, float]] = None,
        right_elbow: Optional[Dict[str, float]] = None,
        torso_x: Optional[float] = None
    ):
        self.person_idx = person_idx
        self.person_box = person_box
        self.person_conf = person_conf
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_elbow = left_elbow
        self.right_elbow = right_elbow
        self.torso_x = torso_x

class YOLO26PoseDetector:
    """
    YOLO26L-Pose pipeline for extracting person bboxes and body keypoints (wrists, shoulders, elbows).
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.resolve_pose_model_path()
        print(f"[YOLO26PoseDetector] Loading YOLO26L-Pose from: {self.model_path} on device {config.DEVICE}")
        self.model = YOLO(str(self.model_path))

    def predict(self, frame: np.ndarray) -> List[PoseDetectionResult]:
        """
        Runs pose estimation and returns structured person & keypoint data.
        """
        res = self.model.predict(
            source=frame,
            device=config.DEVICE,
            conf=config.PERSON_POSE_CONF_THRESHOLD,
            imgsz=config.IMGSZ,
            verbose=False,
            half=config.HALF_PRECISION
        )[0]

        results = []
        if res.boxes is None or len(res.boxes) == 0:
            return results

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        keypoints = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        for p_idx in range(len(boxes)):
            box = boxes[p_idx].tolist()
            p_conf = float(confs[p_idx])
            
            lw, rw, ls, rs, le, re = None, None, None, None, None, None
            torso_x = None

            if keypoints is not None and p_idx < len(keypoints):
                kps = keypoints[p_idx]
                # COCO Keypoints:
                # 5: LS, 6: RS, 7: LE, 8: RE, 9: LW, 10: RW
                if len(kps) > 10:
                    ls_data = kps[5]
                    rs_data = kps[6]
                    le_data = kps[7]
                    re_data = kps[8]
                    lw_data = kps[9]
                    rw_data = kps[10]

                    if ls_data[2] > config.WRIST_CONF_THRESHOLD:
                        ls = {"x": float(ls_data[0]), "y": float(ls_data[1]), "conf": float(ls_data[2])}
                    if rs_data[2] > config.WRIST_CONF_THRESHOLD:
                        rs = {"x": float(rs_data[0]), "y": float(rs_data[1]), "conf": float(rs_data[2])}
                    if le_data[2] > config.WRIST_CONF_THRESHOLD:
                        le = {"x": float(le_data[0]), "y": float(le_data[1]), "conf": float(le_data[2])}
                    if re_data[2] > config.WRIST_CONF_THRESHOLD:
                        re = {"x": float(re_data[0]), "y": float(re_data[1]), "conf": float(re_data[2])}
                    if lw_data[2] > config.WRIST_CONF_THRESHOLD:
                        lw = {"x": float(lw_data[0]), "y": float(lw_data[1]), "conf": float(lw_data[2])}
                    if rw_data[2] > config.WRIST_CONF_THRESHOLD:
                        rw = {"x": float(rw_data[0]), "y": float(rw_data[1]), "conf": float(rw_data[2])}

                    if ls and rs:
                        torso_x = (ls["x"] + rs["x"]) / 2.0

            results.append(PoseDetectionResult(
                person_idx=p_idx,
                person_box=box,
                person_conf=p_conf,
                left_wrist=lw,
                right_wrist=rw,
                left_shoulder=ls,
                right_shoulder=rs,
                left_elbow=le,
                right_elbow=re,
                torso_x=torso_x
            ))

        return results
