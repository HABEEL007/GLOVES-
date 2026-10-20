import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from ..config import config
from ..association.hand_association import AssociatedHand

@dataclass
class TrackHandState:
    track_id: int
    side: str = "RIGHT"
    state: str = "OBSERVING" # UNKNOWN, OBSERVING, GLOVED, BARE_CANDIDATE, CONFIRMING, VIOLATION, UNCERTAIN
    current_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Temporal counters
    consecutive_bare_frames: int = 0
    consecutive_glove_frames: int = 0
    bare_start_time: Optional[float] = None
    
    # Violation lifecycle
    violation_active: bool = False
    last_violation_time: float = 0.0
    violation_count: int = 0

class TemporalComplianceStateMachine:
    """
    Per-track temporal verification state machine.
    Enforces multi-frame observation persistence (1.5-2.0s / 15-30 frames)
    before declaring violations, and deduplicates events via cooldown.
    """
    def __init__(
        self,
        persistence_seconds: float = config.BARE_PERSISTENCE_SECONDS,
        min_stable_frames: int = config.MINIMUM_STABLE_FRAMES,
        cooldown_seconds: float = config.VIOLATION_COOLDOWN_SECONDS,
        bare_conf_thresh: float = config.BARE_VIOLATION_CONF_THRESHOLD
    ):
        self.persistence_seconds = persistence_seconds
        self.min_stable_frames = min_stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.bare_conf_thresh = bare_conf_thresh
        
        self.tracks_state: Dict[int, TrackHandState] = {}

    def update_track(
        self,
        track_id: int,
        associated_hand: Optional[AssociatedHand],
        current_time: Optional[float] = None
    ) -> Tuple[str, bool]:
        """
        Updates the state of a tracked person's right hand.
        Returns:
            (current_state_str, is_newly_confirmed_violation_bool)
        """
        now = current_time or time.time()
        
        if track_id not in self.tracks_state:
            self.tracks_state[track_id] = TrackHandState(
                track_id=track_id,
                side="RIGHT",
                state="OBSERVING",
                first_seen=now,
                last_seen=now
            )

        ts = self.tracks_state[track_id]
        ts.last_seen = now
        new_violation_triggered = False

        # Case 1: No hand detection or UNCERTAIN association
        if associated_hand is None or associated_hand.side == "UNCERTAIN":
            # Requirement #19: UNCERTAIN != BARE
            # Does NOT increment bare frames, stays in current state or relaxes to OBSERVING
            if ts.state not in ["VIOLATION", "CONFIRMING"]:
                ts.state = "UNCERTAIN" if associated_hand and associated_hand.side == "UNCERTAIN" else "OBSERVING"
            ts.consecutive_bare_frames = max(0, ts.consecutive_bare_frames - 1)
            return ts.state, False

        # Update confidence
        ts.current_confidence = associated_hand.confidence

        # Case 2: Gloved Hand Observed
        if associated_hand.class_id == 0: # 0 = gloved_hand
            ts.consecutive_glove_frames += 1
            ts.consecutive_bare_frames = 0
            ts.bare_start_time = None
            ts.state = "GLOVED"
            ts.violation_active = False
            return ts.state, False

        # Case 3: Bare Hand Observed (Candidate)
        # Class 1 = bare_hand
        if associated_hand.class_id == 1:
            # Check confidence bar for violation evaluation
            if associated_hand.confidence >= self.bare_conf_thresh:
                if ts.bare_start_time is None:
                    ts.bare_start_time = now
                    ts.consecutive_bare_frames = 1
                    ts.state = "BARE_CANDIDATE"
                else:
                    ts.consecutive_bare_frames += 1
                    elapsed = now - ts.bare_start_time

                    if ts.consecutive_bare_frames < self.min_stable_frames or elapsed < (self.persistence_seconds * 0.5):
                        ts.state = "BARE_CANDIDATE"
                    elif elapsed < self.persistence_seconds:
                        ts.state = "CONFIRMING"
                    else:
                        # Full persistence requirement met!
                        # Check cooldown / deduplication (Requirement #22)
                        time_since_last_vio = now - ts.last_violation_time
                        if not ts.violation_active and time_since_last_vio > self.cooldown_seconds:
                            ts.state = "VIOLATION"
                            ts.violation_active = True
                            ts.last_violation_time = now
                            ts.violation_count += 1
                            new_violation_triggered = True
                        else:
                            ts.state = "VIOLATION" # Still in violation episode, but cooldown suppresses duplicate alert
            else:
                # Bare hand below confidence threshold -> OBSERVING
                ts.consecutive_bare_frames = max(0, ts.consecutive_bare_frames - 1)
                if ts.state not in ["VIOLATION"]:
                    ts.state = "OBSERVING"

            return ts.state, new_violation_triggered

        return ts.state, False

    def clean_expired_tracks(self, timeout_sec: float = config.TRACK_EXPIRY_SECONDS):
        """Removes tracks that have not been observed recently."""
        now = time.time()
        expired = [tid for tid, st in self.tracks_state.items() if (now - st.last_seen) > timeout_sec]
        for tid in expired:
            del self.tracks_state[tid]

    def get_track_status_summary(self) -> List[Dict]:
        return [
            {
                "track_id": st.track_id,
                "state": st.state,
                "confidence": round(st.current_confidence, 2),
                "bare_frames": st.consecutive_bare_frames,
                "violation_active": st.violation_active,
                "violation_count": st.violation_count
            }
            for st in self.tracks_state.values()
        ]
