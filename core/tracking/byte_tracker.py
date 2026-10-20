import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from scipy.optimize import linear_sum_assignment

from ..config import config
from ..pose.pose_detector import PoseDetectionResult

def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    b1 = np.asarray(box1, dtype=np.float32).flatten()
    b2 = np.asarray(box2, dtype=np.float32).flatten()
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (float(b1[2]) - float(b1[0])) * (float(b1[3]) - float(b1[1])))
    area2 = max(0.0, (float(b2[2]) - float(b2[0])) * (float(b2[3]) - float(b2[1])))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

def iou_batch(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU between all track-detection pairs.
    track_boxes: (T, 4)  det_boxes: (D, 4)  -> returns (T, D) IoU matrix.
    """
    # Expand dims for broadcasting: (T,1,4) vs (1,D,4)
    tb = track_boxes[:, np.newaxis, :]   # (T, 1, 4)
    db = det_boxes[np.newaxis, :, :]     # (1, D, 4)

    ix1 = np.maximum(tb[..., 0], db[..., 0])
    iy1 = np.maximum(tb[..., 1], db[..., 1])
    ix2 = np.minimum(tb[..., 2], db[..., 2])
    iy2 = np.minimum(tb[..., 3], db[..., 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)  # (T, D)

    area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])  # (T,)
    area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])          # (D,)
    union = area_t[:, np.newaxis] + area_d[np.newaxis, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union   # (T, D)


def iou_distance(tracks: List["PersonTrack"], detections: List[PoseDetectionResult]) -> np.ndarray:
    """Cost matrix (1 - IoU) with distance fallback, fully vectorized."""
    T, D = len(tracks), len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    trk_boxes = np.array([t.get_box() for t in tracks], dtype=np.float32)       # (T, 4)
    det_boxes = np.array([d.person_box for d in detections], dtype=np.float32)  # (D, 4)

    iou_mat = iou_batch(trk_boxes, det_boxes)   # (T, D)

    # Where IoU > threshold use (1 - IoU), elsewhere use distance-based cost
    high_iou = iou_mat > 0.05

    # Distance-based fallback (vectorized)
    trk_cx = (trk_boxes[:, 0] + trk_boxes[:, 2]) / 2.0   # (T,)
    trk_cy = (trk_boxes[:, 1] + trk_boxes[:, 3]) / 2.0
    trk_h  = np.maximum(20.0, trk_boxes[:, 3] - trk_boxes[:, 1])  # (T,)

    det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0   # (D,)
    det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0

    # Broadcast: (T, D)
    dist = np.sqrt((trk_cx[:, np.newaxis] - det_cx[np.newaxis, :]) ** 2 +
                   (trk_cy[:, np.newaxis] - det_cy[np.newaxis, :]) ** 2)
    dist_cost = np.minimum(1.0, 0.70 + (dist / (trk_h[:, np.newaxis] * 2.0)) * 0.30)

    cost = np.where(high_iou, 1.0 - iou_mat, dist_cost)
    return cost.astype(np.float32)

class KalmanBoxTracker:
    """Kalman Filter for tracking bounding boxes [x, y, a, h, vx, vy, va, vh]."""
    count = 0

    def __init__(self, bbox: List[float]):
        # State: [cx, cy, aspect_ratio, height, v_cx, v_cy, v_a, v_h]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        for i in range(4, 8):
            self.kf.processNoiseCov[i, i] *= 5.0

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        # Initialize state with measured box
        z = self._convert_bbox_to_z(bbox).flatten()
        self.kf.statePost = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

    def _convert_bbox_to_z(self, bbox: List[float]) -> np.ndarray:
        b = np.asarray(bbox, dtype=np.float32).flatten()
        w = max(1.0, float(b[2]) - float(b[0]))
        h = max(1.0, float(b[3]) - float(b[1]))
        cx = float(b[0]) + w / 2.0
        cy = float(b[1]) + h / 2.0
        a = w / h
        return np.array([[cx], [cy], [a], [h]], dtype=np.float32)

    def _convert_x_to_bbox(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).flatten()
        cx, cy, a, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        w = max(1.0, a * h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        state = self.kf.predict()
        return self._convert_x_to_bbox(state)

    def update(self, bbox: List[float]):
        z = self._convert_bbox_to_z(bbox)
        self.kf.correct(z)

class PersonTrack:
    _id_counter = 0

    @classmethod
    def get_next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, pose_det: PoseDetectionResult, frame_id: int):
        self.track_id = self.get_next_id()
        self.kf = KalmanBoxTracker(pose_det.person_box)
        self.box = np.array(pose_det.person_box, dtype=np.float32)
        self.score = pose_det.person_conf
        self.pose_data = pose_det
        
        self.state = "TRACKED" # "TRACKED", "LOST", "REMOVED"
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.history = []

    def get_box(self) -> np.ndarray:
        return self.box

    def predict(self):
        predicted_box = self.kf.predict()
        self.box = predicted_box
        self.time_since_update += 1

    def update(self, pose_det: PoseDetectionResult, frame_id: int):
        self.kf.update(pose_det.person_box)
        self.box = np.array(pose_det.person_box, dtype=np.float32)
        self.score = pose_det.person_conf
        self.pose_data = pose_det
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = "TRACKED"

    def mark_lost(self):
        self.state = "LOST"

    def mark_removed(self):
        self.state = "REMOVED"

class PersonByteTracker:
    """
    ByteTrack implementation for multi-person tracking using OpenCV Kalman Filter
    and SciPy Linear Sum Assignment. Completely independent of 'lap' library!
    """
    def __init__(
        self,
        high_thresh: float = config.TRACK_HIGH_THRESH,
        low_thresh: float = config.TRACK_LOW_THRESH,
        match_thresh: float = config.MATCH_IOU_THRESH,
        buffer_frames: int = config.TRACK_BUFFER_FRAMES
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.buffer_frames = buffer_frames
        
        self.tracked_tracks: List[PersonTrack] = []
        self.lost_tracks: List[PersonTrack] = []
        self.frame_id = 0

    def update(self, pose_detections: List[PoseDetectionResult]) -> List[PersonTrack]:
        self.frame_id += 1
        
        # 1. Predict locations of existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # 2. Split detections into high and low confidence
        dets_high: List[PoseDetectionResult] = []
        dets_low: List[PoseDetectionResult] = []
        for det in pose_detections:
            if det.person_conf >= self.high_thresh:
                dets_high.append(det)
            elif det.person_conf >= self.low_thresh:
                dets_low.append(det)

        # 3. Match high-conf detections with active tracked tracks
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._linear_assignment(
            self.tracked_tracks, dets_high, self.match_thresh
        )

        for trk, det in matched_tracks_1:
            trk.update(det, self.frame_id)

        # 4. Match remaining unmatched tracks with low-conf detections (ByteTrack second stage)
        remaining_tracks = [self.tracked_tracks[i] for i in unmatched_tracks_1]
        matched_tracks_2, unmatched_tracks_2, _ = self._linear_assignment(
            remaining_tracks, dets_low, 0.50
        )

        for trk, det in matched_tracks_2:
            trk.update(det, self.frame_id)

        # 5. Deal with lost tracks
        unmatched_remain_tracks = [remaining_tracks[i] for i in unmatched_tracks_2]
        for trk in unmatched_remain_tracks:
            trk.mark_lost()
            if trk not in self.lost_tracks:
                self.lost_tracks.append(trk)

        # 6. Re-identify / recover tracks from lost_tracks before spawning new tracks
        unmatched_high_dets = [dets_high[j] for j in unmatched_dets_1]
        recovered_tracks = []
        still_unmatched_dets = unmatched_high_dets

        if len(self.lost_tracks) > 0 and len(unmatched_high_dets) > 0:
            matched_lost, _, unmatched_lost_dets_idx = self._linear_assignment(
                self.lost_tracks, unmatched_high_dets, thresh=0.70
            )
            for trk, det in matched_lost:
                trk.update(det, self.frame_id)
                recovered_tracks.append(trk)
                if trk in self.lost_tracks:
                    self.lost_tracks.remove(trk)
            still_unmatched_dets = [unmatched_high_dets[k] for k in unmatched_lost_dets_idx]

        # 7. Initialize new tracks ONLY from genuinely new, unmatched high-confidence detections
        new_tracks = []
        for det in still_unmatched_dets:
            if det.person_conf >= self.high_thresh:
                new_trk = PersonTrack(det, self.frame_id)
                new_tracks.append(new_trk)

        # 8. Update tracked list
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == "TRACKED"] + recovered_tracks + new_tracks

        # 9. Clean up old lost tracks exceeding buffer
        active_lost = []
        for t in self.lost_tracks:
            if self.frame_id - t.frame_id > self.buffer_frames:
                t.mark_removed()
            else:
                active_lost.append(t)
        self.lost_tracks = active_lost

        return [t for t in self.tracked_tracks if t.state == "TRACKED"]

    def _linear_assignment(
        self,
        tracks: List[PersonTrack],
        detections: List[PoseDetectionResult],
        thresh: float
    ) -> Tuple[List[Tuple[PersonTrack, PoseDetectionResult]], List[int], List[int]]:
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = iou_distance(tracks, detections)

        # Fast greedy matching for tiny matrices (avoids scipy overhead)
        if cost_matrix.shape[0] <= 6 and cost_matrix.shape[1] <= 6:
            row_ind, col_ind = self._greedy_assign(cost_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < thresh:
                matched.append((tracks[r], detections[c]))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    @staticmethod
    def _greedy_assign(cost: np.ndarray) -> Tuple[List[int], List[int]]:
        """Greedy O(n²) assignment — faster than scipy for matrices ≤6×6."""
        used_rows, used_cols = set(), set()
        rows, cols = [], []
        flat_order = np.argsort(cost.ravel())
        for idx in flat_order:
            r, c = divmod(int(idx), cost.shape[1])
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
        return rows, cols
