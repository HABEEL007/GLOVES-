import os
import sys
import argparse
from pathlib import Path
import torch

# Safe UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.config import config
from core.training.dataset_manager import DatasetManager
from core.training.train_engine import YOLO26TrainEngine
from core.active_learning.model_registry import ModelRegistry

def run_finetuning(
    epochs: int = 40,
    batch: int = 4, # 4 is safe for 6GB VRAM on GTX 1660 Ti
    imgsz: int = config.IMGSZ,
    base_model: str = None
):
    print("=" * 65)
    print("  YOLO26L Hand Detector Fine-Tuning & Acceptance Pipeline")
    print(f"  Device: {config.DEVICE} | Target Epochs: {epochs} | Batch: {batch}")
    print("=" * 65)

    # 1. Dataset Preparation with Session-based Split (70% train, 15% val, 15% test)
    dm = DatasetManager()
    yaml_path, counts = dm.prepare_combined_dataset()

    # 2. Train Engine
    engine = YOLO26TrainEngine()
    # Always start from clean pretrained weights unless user explicitly overrides.
    # Using a previously-rejected active_model risks amplifying its biases.
    start_weights = base_model or str(config.YOLO26L_BASE_PATH)
    print(f"  Base Weights: {start_weights}")

    train_res = engine.train_hand_detector(
        data_yaml=yaml_path,
        model_name=start_weights,
        epochs=epochs,
        batch_size=batch,
        imgsz=imgsz,
        run_name="active_model"
    )

    candidate_weights = train_res["best_weights"]

    # 3. Model Acceptance Gate
    registry = ModelRegistry()
    metrics = engine.validate_model(candidate_weights, yaml_path)
    passed, reason = registry.evaluate_acceptance(metrics)

    print("\n" + "=" * 65)
    print("  EVALUATION & ACCEPTANCE REPORT")
    print(f"  mAP50:     {metrics.get('map50', 0)*100:.1f}%")
    print(f"  Precision: {metrics.get('precision', 0)*100:.1f}%")
    print(f"  Recall:    {metrics.get('recall', 0)*100:.1f}%")
    print(f"  F1-Score:  {metrics.get('f1', 0)*100:.1f}%")
    print(f"  Acceptance Result: {'PASSED (Promoted to Champion)' if passed else 'REJECTED'}")
    if not passed:
        print(f"  Reason: {reason}")
    print("=" * 65)

    if passed:
        champion_path = registry.promote_to_champion(
            candidate_weights=candidate_weights,
            version_name="YOLO26L-CCTV-Champion",
            metrics=metrics,
            training_info={"epochs": epochs, "batch_size": batch, "imgsz": imgsz}
        )
        return str(champion_path)

    return str(candidate_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26L Gloves & Hand Detection Model on CCTV Dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of fine-tuning epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4 for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--model", type=str, default=None, help="Base weights path")
    args = parser.parse_args()

    run_finetuning(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, base_model=args.model)
