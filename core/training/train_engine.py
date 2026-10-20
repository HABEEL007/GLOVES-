import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO

from ..config import config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class YOLO26TrainEngine:
    """
    Manages controlled training and validation of the YOLO26L Hand Detector.
    Ensures safe VRAM allocation on 6GB GTX 1660 Ti and logs evaluation metrics.
    """
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.BASE_DIR

    def train_hand_detector(
        self,
        data_yaml: Path,
        model_name: str = "yolo26l.pt",
        epochs: int = 100,
        batch_size: int = 4, # 4 is optimal for safe VRAM on 6GB GPU
        imgsz: int = config.IMGSZ,
        run_name: str = "candidate_yolo26l",
        resume: bool = False
    ) -> Dict[str, Any]:
        print("=" * 65)
        print(f"[YOLO26TrainEngine] Launching YOLO26L Hand Detector Training")
        print(f"  Model       : {model_name}")
        print(f"  Dataset     : {data_yaml}")
        print(f"  Epochs      : {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
        print(f"  Device      : {config.DEVICE}")
        print(f"  Classes     : 0=gloved_hand, 1=bare_hand")
        print("=" * 65)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = YOLO(model_name)

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=config.DEVICE,
            workers=0, # Prevents Windows error 1455
            name=run_name,
            project=str(self.base_dir / "runs" / "detect"),
            exist_ok=True,
            optimizer="AdamW",
            patience=25,           # Give more room before early stopping
            save=True,
            val=True,
            half=True,             # FP16: 3x speedup on GTX 1660 Ti
            plots=True,
            resume=resume,
            close_mosaic=15,       # Disable mosaic last 15 epochs for clean convergence
            # CCTV-domain augmentations
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            degrees=15.0,
            translate=0.15,
            scale=0.5,             # Wider scale range for distance variation in CCTV
            shear=5.0,
            perspective=0.0005,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,            # Full mosaic to boost small-object variety
            mixup=0.15,            # Mix gloved + bare examples to reduce false positives
            copy_paste=0.1         # Copy-paste augmentation for gloved_hand bootstrap
        )

        best_weights = self.base_dir / "runs" / "detect" / run_name / "weights" / "best.pt"
        return {
            "run_name": run_name,
            "best_weights": best_weights,
            "results": results
        }

    def validate_model(self, weights_path: Path, data_yaml: Path) -> Dict[str, float]:
        """Runs evaluation on validation or test split and extracts metrics."""
        print(f"[YOLO26TrainEngine] Evaluating candidate weights: {weights_path}")
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=config.IMGSZ,
            device=config.DEVICE,
            workers=0,
            verbose=False
        )

        box = metrics.box
        p = float(box.mp) if hasattr(box, "mp") else 0.0
        r = float(box.mr) if hasattr(box, "mr") else 0.0
        map50 = float(box.map50) if hasattr(box, "map50") else 0.0
        map50_95 = float(box.map) if hasattr(box, "map") else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "f1": round(f1, 4)
        }
