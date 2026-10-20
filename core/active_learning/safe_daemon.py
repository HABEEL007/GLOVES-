import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import config
from ..training.dataset_manager import DatasetManager
from ..training.train_engine import YOLO26TrainEngine
from .model_registry import ModelRegistry

STATUS_FILE = config.BASE_DIR / "runs" / "active_learning_status.json"

class SafeActiveLearningDaemon:
    """
    Safely triggers candidate model training and validation gates.
    Never overwrites production champion model without passing acceptance tests.
    """
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.BASE_DIR
        self.dataset_manager = DatasetManager()
        self.train_engine = YOLO26TrainEngine()
        self.registry = ModelRegistry()
        self.status_file = STATUS_FILE
        self.is_training = False
        self.lock = threading.Lock()
        self._init_status()

    def _init_status(self):
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.status_file.exists():
            self._write_status({
                "is_training": False,
                "current_epoch": 0,
                "total_epochs": 40,
                "message": "Idle - Ready for candidate evaluation",
                "active_champion": self.registry.get_registry().get("active_champion", "YOLO26L-Base"),
                "last_trained": None
            })

    def get_status(self) -> Dict[str, Any]:
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"is_training": self.is_training, "message": "Idle"}

    def _write_status(self, data: Dict[str, Any]):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SafeActiveLearningDaemon] Status write error: {e}")

    def trigger_candidate_training_async(
        self,
        epochs: int = 40,
        batch_size: int = 4,
        on_champion_promoted: Optional[Any] = None
    ) -> bool:
        with self.lock:
            if self.is_training:
                return False
            self.is_training = True

        thread = threading.Thread(
            target=self._run_training_pipeline,
            args=(epochs, batch_size, on_champion_promoted),
            daemon=True,
            name="ActiveLearningCandidateWorker"
        )
        thread.start()
        return True

    def _run_training_pipeline(self, epochs: int, batch_size: int, on_champion_promoted: Optional[Any]):
        try:
            # Step 1: Dataset Quality Check & Session-Based Split
            self._write_status({
                "is_training": True,
                "stage": "1/4: Dataset Verification & Session Split",
                "message": "Building balanced dataset with black glove focus and hard negatives..."
            })
            yaml_path, counts = self.dataset_manager.prepare_combined_dataset()

            # Step 2: Train Candidate Model
            self._write_status({
                "is_training": True,
                "stage": "2/4: Candidate Model Training",
                "message": f"Training YOLO26L for {epochs} epochs (Batch: {batch_size})..."
            })
            run_name = f"candidate_yolo26l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            train_res = self.train_engine.train_hand_detector(
                data_yaml=yaml_path,
                model_name=str(config.YOLO26L_BASE_PATH),
                epochs=epochs,
                batch_size=batch_size,
                run_name=run_name
            )

            candidate_weights = train_res["best_weights"]

            # Step 3: Automated Evaluation
            self._write_status({
                "is_training": True,
                "stage": "3/4: Acceptance Gate Evaluation",
                "message": "Validating candidate model against acceptance criteria..."
            })
            metrics = self.train_engine.validate_model(candidate_weights, yaml_path)

            # Step 4: Acceptance Gate Decision
            passed, reason = self.registry.evaluate_acceptance(metrics)

            if passed:
                new_champion_path = self.registry.promote_to_champion(
                    candidate_weights=candidate_weights,
                    version_name=run_name,
                    metrics=metrics,
                    training_info={"run_name": run_name, "epochs": epochs, "batch_size": batch_size}
                )
                if on_champion_promoted:
                    try:
                        on_champion_promoted(new_champion_path)
                    except Exception as e:
                        print(f"[SafeActiveLearningDaemon] Callback error: {e}")

                self._write_status({
                    "is_training": False,
                    "stage": "4/4: Completed",
                    "status": "PROMOTED_CHAMPION",
                    "message": f"Candidate passed acceptance! New Champion: {run_name}",
                    "metrics": metrics,
                    "champion_path": str(new_champion_path)
                })
            else:
                self._write_status({
                    "is_training": False,
                    "stage": "4/4: Completed",
                    "status": "REJECTED",
                    "message": f"Candidate rejected by acceptance gate: {reason}",
                    "metrics": metrics
                })

        except Exception as e:
            print(f"[SafeActiveLearningDaemon] Pipeline error: {e}")
            self._write_status({
                "is_training": False,
                "status": "ERROR",
                "message": f"Pipeline failed: {str(e)}"
            })
        finally:
            with self.lock:
                self.is_training = False

safe_active_daemon = SafeActiveLearningDaemon()
