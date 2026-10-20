import sys
import os
import argparse
from pathlib import Path

# Add CPPE-Dataset to sys.path
cppe_dir = Path(__file__).resolve().parent / "CPPE-Dataset"
sys.path.insert(0, str(cppe_dir))

import train_gloves as tg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Gloves Detection on GPU")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=12, help="Batch size (12 for safe max GPU utilization on 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 prevents Windows error 1455)")
    parser.add_argument("--model", type=str, default=None, help="Weights to fine-tune (defaults to best.pt or yolov8s.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset yaml")
    parser.add_argument("--name", type=str, default="gloves_detector_cctv_50ep", help="Run folder name")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint (last.pt)")

    args = parser.parse_args()
    tg.train_gloves(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        workers=args.workers,
        resume=args.resume
    )
