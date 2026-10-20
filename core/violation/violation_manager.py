from typing import Optional, List, Dict, Any
import numpy as np

from ..config import config
from ..association.hand_association import AssociatedHand
from ..temporal_logic.state_machine import TemporalComplianceStateMachine
from ..evidence.evidence_recorder import EvidenceRecorder
from ..database.db_service import db_service

class ViolationManager:
    """
    Evaluates Phase 1 Compliance Business Rules:
    - Valid Staff Track
    - Valid Right Wrist
    - Valid Right Hand Association
    - Right Hand = bare_hand
    - Temporal Persistence Satisfied
    - Confidence Threshold Satisfied
    Left Hand = Bare -> ALLOWED.
    """
    def __init__(
        self,
        state_machine: Optional[TemporalComplianceStateMachine] = None,
        evidence_recorder: Optional[EvidenceRecorder] = None
    ):
        self.state_machine = state_machine or TemporalComplianceStateMachine()
        self.evidence_recorder = evidence_recorder or EvidenceRecorder()
        self.total_violations_today = 0
        self.recent_evidence_path: Optional[str] = None

    def evaluate_compliance(
        self,
        associated_hands: List[AssociatedHand],
        tracked_ids: List[int],
        raw_frame: np.ndarray,
        camera_id: str = "301",
        model_version: str = "YOLO26L-v1"
    ) -> Dict[str, Any]:
        """
        Evaluates Either-Hand Glove Compliance:
        - If a person has a Black Glove on EITHER hand (Right or Left):
            -> COMPLIANT (No Alert)
        - If a person has a glove but color is NOT black:
            -> WRONG COLOR WARNING
        - If all visible hands for this person are bare (no gloves on either hand):
            -> SUSTAINED BARE HAND VIOLATION
        """
        # Group hands by track_id
        track_to_hands: Dict[int, List[AssociatedHand]] = {tid: [] for tid in tracked_ids}
        for ah in associated_hands:
            if ah.track_id is not None and ah.track_id in track_to_hands:
                track_to_hands[ah.track_id].append(ah)

        active_violations = 0
        track_states = {}
        person_gloves_count = 0
        person_bare_count = 0
        total_wrong_color_count = 0

        # Evaluate each active tracked worker
        for tid in tracked_ids:
            hands = track_to_hands.get(tid, [])
            
            gloved_hands = [h for h in hands if h.class_id == 0]
            bare_hands = [h for h in hands if h.class_id == 1]

            black_gloves = [
                h for h in gloved_hands 
                if getattr(h.hand_box, "glove_color", "unknown") == "black"
            ]
            wrong_color_gloves = [
                h for h in gloved_hands 
                if getattr(h.hand_box, "glove_color", "unknown") not in ("black", "unknown")
            ]

            representative_hand: Optional[AssociatedHand] = None

            if len(black_gloves) > 0:
                # ✅ Person has at least one black glove (Right or Left) -> Fully Compliant!
                representative_hand = black_gloves[0]
                person_gloves_count += 1
            elif len(wrong_color_gloves) > 0:
                # ⚠️ Glove detected on at least one hand, but not black
                representative_hand = wrong_color_gloves[0]
                total_wrong_color_count += len(wrong_color_gloves)
            elif len(bare_hands) > 0:
                # 🚨 No glove on either hand, but bare hand(s) observed!
                representative_hand = bare_hands[0]
                person_bare_count += 1
            else:
                representative_hand = None

            # Feed representative hand into state machine for temporal smoothing
            state, newly_triggered = self.state_machine.update_track(tid, representative_hand)
            track_states[tid] = state

            if state == "VIOLATION":
                active_violations += 1

            if newly_triggered and representative_hand is not None:
                # Capture evidence snapshot and record to DB
                ev_path = self.evidence_recorder.capture_evidence(
                    raw_frame=raw_frame,
                    track_id=tid,
                    camera_id=camera_id,
                    confidence=representative_hand.confidence,
                    hand_box=representative_hand.box,
                    model_version=model_version
                )
                self.recent_evidence_path = ev_path
                self.total_violations_today += 1
                
                # Persist to database
                db_service.create_violation(
                    camera_id=camera_id,
                    track_id=tid,
                    conf=representative_hand.confidence,
                    evidence_path=ev_path,
                    model_ver=model_version
                )

        # Cleanup tracks no longer seen
        self.state_machine.clean_expired_tracks()

        # Determine overall frame compliance status
        if active_violations > 0:
            status_text = f"🚨 VIOLATION CONFIRMED: {active_violations} Worker(s) with No Gloves"
            badge_type = "danger"
        elif total_wrong_color_count > 0:
            status_text = f"⚠️ WRONG GLOVE COLOR: Black gloves required ({total_wrong_color_count} non-black observed)"
            badge_type = "warning"
        elif person_bare_count > 0:
            status_text = f"⏳ CONFIRMING: Bare hands observed on {person_bare_count} worker(s)"
            badge_type = "warning"
        elif person_gloves_count > 0:
            status_text = f"✅ COMPLIANT: Protected with black glove ({person_gloves_count} worker gloved)"
            badge_type = "success"
        elif len(tracked_ids) > 0:
            status_text = f"👀 OBSERVING: {len(tracked_ids)} worker(s) in view"
            badge_type = "neutral"
        else:
            status_text = "🛡️ MONITORING: Area Clear"
            badge_type = "neutral"

        return {
            "status_text": status_text,
            "badge_type": badge_type,
            "active_violations": active_violations,
            "total_violations_today": self.total_violations_today,
            "right_gloves_count": person_gloves_count,
            "left_hand_bare_count": person_bare_count,
            "track_states": track_states,
            "recent_evidence_path": self.recent_evidence_path
        }
