import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..config import config
from ..tracking.byte_tracker import PersonTrack
from ..hand_detection.hand_detector import HandDetectionBox

class AssociatedHand:
    def __init__(
        self,
        hand_box: HandDetectionBox,
        track_id: Optional[int],
        side: str, # "RIGHT", "LEFT", "UNCERTAIN"
        association_confidence: float,
        wrist_distance: float,
        is_staff: bool = True,
        notes: str = ""
    ):
        self.hand_box = hand_box
        self.track_id = track_id
        self.side = side
        self.association_confidence = association_confidence
        self.wrist_distance = wrist_distance
        self.is_staff = is_staff
        self.notes = notes

    @property
    def class_id(self) -> int:
        return self.hand_box.class_id

    @property
    def class_name(self) -> str:
        return self.hand_box.class_name

    @property
    def confidence(self) -> float:
        return self.hand_box.confidence

    @property
    def box(self) -> List[float]:
        return self.hand_box.box

class HandAssociationEngine:
    """
    Robust person-aware hand association module.
    Prevents single-point Euclidean flaws, multi-person confusion,
    and returns UNCERTAIN when geometry or confidence is ambiguous.
    """
    def __init__(
        self,
        max_dist_px: float = config.ASSOCIATION_DISTANCE_MAX_PX,
        min_assoc_conf: float = config.ASSOCIATION_CONFIDENCE_MIN
    ):
        self.max_dist_px = max_dist_px
        self.min_assoc_conf = min_assoc_conf

    def associate_hands(
        self,
        hands: List[HandDetectionBox],
        tracks: List[PersonTrack],
        img_width: int,
        img_height: int
    ) -> List[AssociatedHand]:
        associated_results: List[AssociatedHand] = []

        if not hands:
            return associated_results

        # If no tracked persons in view, all detected hands are untracked / customer hands
        if not tracks:
            for h in hands:
                associated_results.append(AssociatedHand(
                    hand_box=h,
                    track_id=None,
                    side="UNCERTAIN",
                    association_confidence=0.0,
                    wrist_distance=999.0,
                    is_staff=False,
                    notes="No tracked person detected in frame"
                ))
            return associated_results

        for h in hands:
            hx, hy = h.cx, h.cy
            
            # Step 1: Candidate Person Search & Distance Evaluation
            candidates = []
            for trk in tracks:
                p_box = trk.box
                pw = max(10.0, p_box[2] - p_box[0])
                ph = max(20.0, p_box[3] - p_box[1])

                # Check expanded body envelope (arm reach zone)
                env_x1 = p_box[0] - pw * 0.40
                env_y1 = p_box[1] - ph * 0.10
                env_x2 = p_box[2] + pw * 0.40
                env_y2 = p_box[3] + ph * 0.15

                in_body_zone = (env_x1 <= hx <= env_x2 and env_y1 <= hy <= env_y2)

                pose_data = trk.pose_data
                rw = pose_data.right_wrist if pose_data else None
                lw = pose_data.left_wrist if pose_data else None
                torso_x = pose_data.torso_x if pose_data else None

                # Calculate distances to wrists
                d_rw = math.hypot(hx - rw["x"], hy - rw["y"]) if rw else float("inf")
                d_lw = math.hypot(hx - lw["x"], hy - lw["y"]) if lw else float("inf")

                candidates.append({
                    "track": trk,
                    "in_zone": in_body_zone,
                    "d_rw": d_rw,
                    "d_lw": d_lw,
                    "rw": rw,
                    "lw": lw,
                    "torso_x": torso_x,
                    "ph": ph,
                    "pw": pw
                })

            # Find best candidate person
            best_cand = None
            min_dist = float("inf")
            chosen_side = "UNCERTAIN"

            for c in candidates:
                cand_min = min(c["d_rw"], c["d_lw"])
                if cand_min < min_dist:
                    min_dist = cand_min
                    best_cand = c
                    chosen_side = "RIGHT" if c["d_rw"] <= c["d_lw"] else "LEFT"

            # Check if hand is too far or unassociated
            if best_cand is None or min_dist == float("inf"):
                associated_results.append(AssociatedHand(
                    hand_box=h,
                    track_id=None,
                    side="UNCERTAIN",
                    association_confidence=0.0,
                    wrist_distance=999.0,
                    is_staff=False,
                    notes="No visible wrists found"
                ))
                continue

            # Scale max search distance to person height
            adaptive_max_dist = max(100.0, min(self.max_dist_px, best_cand["ph"] * 0.35))

            # Step 2: Multi-person Ambiguity Check
            # If second-best person is nearly as close, flag as UNCERTAIN
            sorted_by_dist = sorted(candidates, key=lambda x: min(x["d_rw"], x["d_lw"]))
            if len(sorted_by_dist) > 1:
                d1 = min(sorted_by_dist[0]["d_rw"], sorted_by_dist[0]["d_lw"])
                d2 = min(sorted_by_dist[1]["d_rw"], sorted_by_dist[1]["d_lw"])
                if abs(d1 - d2) < 35.0 and d1 < adaptive_max_dist:
                    # Ambiguous overlap between two workers or worker and customer
                    associated_results.append(AssociatedHand(
                        hand_box=h,
                        track_id=best_cand["track"].track_id,
                        side="UNCERTAIN",
                        association_confidence=0.30,
                        wrist_distance=round(d1, 1),
                        notes="Multi-person hand proximity conflict"
                    ))
                    continue

            # Step 3: Calculate Association Confidence Score (Requirement #18)
            # Formula: (0.45 * s_prox) + (0.30 * s_wrist) + (0.15 * s_geom) + (0.10 * s_track)
            s_prox = max(0.0, 1.0 - (min_dist / adaptive_max_dist))
            
            target_wrist = best_cand["rw"] if chosen_side == "RIGHT" else best_cand["lw"]
            s_wrist = target_wrist["conf"] if target_wrist else 0.0
            
            s_geom = 1.0 if best_cand["in_zone"] else 0.40
            
            tracklet_len = getattr(best_cand["track"], "time_since_update", 0)
            s_track = 1.0 if tracklet_len == 0 else 0.70

            assoc_conf = round(
                (0.45 * s_prox) + (0.30 * s_wrist) + (0.15 * s_geom) + (0.10 * s_track),
                2
            )

            # Step 4: Fallback side orientation check via Torso X if wrist is far
            if min_dist > adaptive_max_dist:
                if best_cand["torso_x"] is not None and best_cand["in_zone"]:
                    # In front camera: employee right hand is to the left of their torso center
                    chosen_side = "RIGHT" if hx < best_cand["torso_x"] else "LEFT"
                    assoc_conf = max(0.20, assoc_conf * 0.5)
                else:
                    chosen_side = "UNCERTAIN"
                    assoc_conf = 0.10

            # Step 5: Enforce UNCERTAIN when below threshold
            if assoc_conf < self.min_assoc_conf:
                chosen_side = "UNCERTAIN"

            associated_results.append(AssociatedHand(
                hand_box=h,
                track_id=best_cand["track"].track_id,
                side=chosen_side,
                association_confidence=assoc_conf,
                wrist_distance=round(min_dist, 1),
                is_staff=True,
                notes=f"Associated with Track {best_cand['track'].track_id}"
            ))

        return associated_results
