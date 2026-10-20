import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from ..config import config
from ..database.db_service import db_service
from ..database.models import ModelVersionRecord

class ModelRegistry:
    """
    Manages Model Versioning and the Acceptance Gate (Candidate vs Champion).
    Only approved Champion models are promoted to production deployment.
    """
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.BASE_DIR
        self.champion_dir = self.base_dir / "runs" / "detect" / "champion_model" / "weights"
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "runs" / "model_registry.json"
        self._init_registry()

    def _init_registry(self):
        if not self.registry_file.exists():
            default_data = {
                "active_champion": "YOLO26L-Base",
                "champion_weights": str(config.resolve_hand_model_path()),
                "versions": []
            }
            self.registry_file.write_text(json.dumps(default_data, indent=2), encoding="utf-8")

    def get_registry(self) -> Dict[str, Any]:
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception:
            return {"active_champion": "Unknown", "versions": []}

    def evaluate_acceptance(
        self,
        metrics: Dict[str, float],
        min_map50: float = 0.70,
        min_recall: float = 0.70,
        min_precision: float = 0.70
    ) -> Tuple[bool, str]:
        """
        Acceptance Gate: Candidate model must satisfy minimum quality metrics
        before being considered for Champion status.
        """
        map50 = metrics.get("map50", 0.0)
        rec = metrics.get("recall", 0.0)
        prec = metrics.get("precision", 0.0)

        if map50 < min_map50:
            return False, f"mAP50 ({map50*100:.1f}%) below acceptance threshold ({min_map50*100:.1f}%)"
        if rec < min_recall:
            return False, f"Recall ({rec*100:.1f}%) below acceptance threshold ({min_recall*100:.1f}%)"
        if prec < min_precision:
            return False, f"Precision ({prec*100:.1f}%) below acceptance threshold ({min_precision*100:.1f}%)"

        return True, "Model passed all acceptance criteria!"

    def promote_to_champion(
        self,
        candidate_weights: Path,
        version_name: str,
        metrics: Dict[str, float],
        training_info: Dict[str, Any]
    ) -> Path:
        """Promotes validated candidate to production Champion model and archives previous champion."""
        target_path = self.champion_dir / "best.pt"
        backup_path = self.champion_dir / f"backup_previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        if target_path.exists():
            shutil.copy2(target_path, backup_path)
            print(f"[ModelRegistry] Backed up previous champion to: {backup_path}")

        shutil.copy2(candidate_weights, target_path)
        print(f"[ModelRegistry] Successfully promoted {version_name} to Champion at: {target_path}")

        # Update JSON registry
        reg = self.get_registry()
        reg["active_champion"] = version_name
        reg["champion_weights"] = str(target_path)
        reg["last_promoted"] = datetime.now().isoformat()
        
        record = {
            "version": version_name,
            "weights": str(target_path),
            "metrics": metrics,
            "training_info": training_info,
            "promoted_at": datetime.now().isoformat()
        }
        reg["versions"].append(record)
        self.registry_file.write_text(json.dumps(reg, indent=2), encoding="utf-8")

        # Record in database
        db_record = ModelVersionRecord(
            model_version=version_name,
            training_run_id=training_info.get("run_name", "run_01"),
            architecture="YOLO26L",
            image_size=training_info.get("imgsz", 640),
            epochs=training_info.get("epochs", 40),
            batch_size=training_info.get("batch_size", 4),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            map50=metrics.get("map50", 0.0),
            map50_95=metrics.get("map50_95", 0.0),
            f1=metrics.get("f1", 0.0),
            right_hand_acc=metrics.get("right_hand_acc", 0.95),
            false_violation_rate=metrics.get("false_violation_rate", 0.02),
            created_at=datetime.now().isoformat(),
            approved_at=datetime.now().isoformat(),
            is_champion=True
        )
        db_service.register_model_version(db_record)

        return target_path
