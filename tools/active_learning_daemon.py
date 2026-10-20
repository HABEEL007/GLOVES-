import os
import sys
import json
import time
import shutil
import random
import threading
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
STATUS_FILE = BASE_DIR / "runs" / "active_learning_status.json"

class ActiveLearningTrainer:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.dataset_dir = self.base_dir / "active_learning_dataset"
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
                "total_epochs": 5,
                "message": "Idle - Ready to train",
                "last_trained": None,
                "best_weights": None,
                "sample_count": 0
            })

    def get_status(self):
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"is_training": self.is_training, "message": "Idle"}

    def _write_status(self, data):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ActiveLearning] Error writing status: {e}")

    def prepare_dataset(self, gloves_sample_limit=200):
        """Prepares a balanced 2-class dataset (0: Gloves, 1: Hand) from CCTV + CPPE augmented."""
        print("[ActiveLearning] Preparing dataset...")
        
        train_img = self.dataset_dir / "images" / "train"
        train_lbl = self.dataset_dir / "labels" / "train"
        val_img = self.dataset_dir / "images" / "val"
        val_lbl = self.dataset_dir / "labels" / "val"

        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
            for old_file in d.glob("*"):
                try:
                    old_file.unlink()
                except Exception:
                    pass

        # 1. Collect CCTV labeled pairs (Hands & Negatives)
        cctv_dir = self.base_dir / "cctv_dataset"
        cctv_pairs = []

        for cat in ["negatives", "gloves"]:
            img_cat = cctv_dir / "images" / cat
            lbl_cat = cctv_dir / "labels" / cat
            if not img_cat.exists():
                continue
            for img_path in img_cat.glob("*.jpg"):
                lbl_path = lbl_cat / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    cctv_pairs.append((img_path, lbl_path))

        # 2. Collect Gloves (Class 0) from Augmented Dataset to prevent forgetting
        cppe_augmented_train = self.base_dir / "CPPE-Dataset" / "yolo_dataset_augmented" / "images" / "train"
        cppe_augmented_lbl = self.base_dir / "CPPE-Dataset" / "yolo_dataset_augmented" / "labels" / "train"
        gloves_pairs = []
        if cppe_augmented_train.exists():
            for g_img in cppe_augmented_train.glob("*.jpg"):
                g_lbl = cppe_augmented_lbl / f"{g_img.stem}.txt"
                if g_lbl.exists() and g_lbl.stat().st_size > 0:
                    gloves_pairs.append((g_img, g_lbl))

        # Subsample gloves to maintain balanced ratio
        random.seed(42)
        if len(gloves_pairs) > gloves_sample_limit:
            gloves_pairs = random.sample(gloves_pairs, gloves_sample_limit)

        print(f"[ActiveLearning] Found {len(cctv_pairs)} CCTV frames and {len(gloves_pairs)} Gloves images.")

        all_pairs = cctv_pairs + gloves_pairs
        if len(all_pairs) < 5:
            print("[ActiveLearning] Not enough samples to train yet (min 5 required).")
            return False, 0

        random.shuffle(all_pairs)
        val_count = max(2, int(len(all_pairs) * 0.15))
        train_pairs = all_pairs[val_count:]
        val_pairs = all_pairs[:val_count]

        # Copy train pairs
        for src_img, src_lbl in train_pairs:
            shutil.copy2(src_img, train_img / src_img.name)
            shutil.copy2(src_lbl, train_lbl / f"{src_img.stem}.txt")

        # Copy val pairs
        for src_img, src_lbl in val_pairs:
            shutil.copy2(src_img, val_img / src_img.name)
            shutil.copy2(src_lbl, val_lbl / f"{src_img.stem}.txt")

        # Invalidate YOLO cache files so it rescans
        for cache in self.dataset_dir.glob("labels/**/*.cache"):
            try:
                cache.unlink()
            except Exception:
                pass

        # Write 2-Class data.yaml
        data_yaml_path = self.dataset_dir / "active_dataset.yaml"
        yaml_content = f"""path: {self.dataset_dir.as_posix()}
train: images/train
val: images/val

names:
  0: Gloves
  1: Hand
nc: 2
"""
        with open(data_yaml_path, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)

        print(f"[ActiveLearning] Dataset ready: {len(train_pairs)} train, {len(val_pairs)} val.")
        return True, len(all_pairs)

    def train_background(self, epochs=20):
        with self.lock:
            if self.is_training:
                print("[ActiveLearning] Training already in progress, skipping trigger.")
                return False
    def run_training(self, epochs=20, batch_size=12, imgsz=640):
        """Runs the active learning training job directly in current process."""
        try:
            import torch
            from ultralytics import YOLO

            self._write_status({
                "is_training": True,
                "current_epoch": 0,
                "total_epochs": epochs,
                "message": "Preparing active learning dataset...",
                "last_trained": self.get_status().get("last_trained")
            })

            success, total_samples = self.prepare_dataset(gloves_sample_limit=250)
            if not success:
                self._write_status({
                    "is_training": False,
                    "message": "Skipped: Not enough labeled samples to train (need at least 5)."
                })
                return False

            # Choose starting weights: cctv_50ep, active_best, or aug_best
            cctv_50ep_best = self.base_dir / "CPPE-Dataset" / "runs" / "detect" / "gloves_detector_cctv_50ep" / "weights" / "best.pt"
            active_best = self.base_dir / "runs" / "detect" / "active_model" / "weights" / "best.pt"
            prev_best = self.base_dir / "CPPE-Dataset" / "runs" / "detect" / "gloves_detector_aug" / "weights" / "best.pt"
            if cctv_50ep_best.exists():
                model_path = str(cctv_50ep_best)
            elif active_best.exists():
                model_path = str(active_best)
            elif prev_best.exists():
                model_path = str(prev_best)
            else:
                model_path = "yolov8s.pt"

            print(f"[ActiveLearning] Starting fine-tuning from: {model_path} for {epochs} epochs...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(model_path)

            def on_train_epoch_end(trainer):
                ep = trainer.epoch + 1
                tot = trainer.epochs
                msg = f"Epoch {ep}/{tot} completed"
                self._write_status({
                    "is_training": True,
                    "current_epoch": ep,
                    "total_epochs": tot,
                    "message": msg,
                    "sample_count": total_samples
                })

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            project_dir = self.base_dir / "runs" / "detect"
            results = model.train(
                data=str(self.dataset_dir / "active_dataset.yaml"),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                project=str(project_dir),
                name="active_model",
                exist_ok=True,
                lr0=0.002,
                lrf=0.01,
                verbose=False,
                plots=False,
                workers=0
            )

            best_pt = project_dir / "active_model" / "weights" / "best.pt"
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")

            print(f"[ActiveLearning] Training finished successfully! Saved to: {best_pt}")
            self._write_status({
                "is_training": False,
                "current_epoch": epochs,
                "total_epochs": epochs,
                "message": f"Successfully updated at {now_str}!",
                "last_trained": now_str,
                "best_weights": str(best_pt),
                "sample_count": total_samples
            })
            return True

        except Exception as e:
            print(f"[ActiveLearning] Training error: {e}")
            import traceback
            traceback.print_exc()
            self._write_status({
                "is_training": False,
                "message": f"Training failed: {str(e)[:100]}"
            })
            return False

    def train_background(self, epochs=5):
        """Spawns an isolated background process to run training smoothly without blocking Flask."""
        import subprocess
        current_status = self.get_status()
        if current_status.get("is_training", False):
            print("[ActiveLearning] Training already in progress, skipping trigger.")
            return False

        self._write_status({
            "is_training": True,
            "current_epoch": 0,
            "total_epochs": epochs,
            "message": "Starting background trainer...",
            "last_trained": current_status.get("last_trained")
        })

        script_path = Path(__file__).resolve()
        cmd = [sys.executable, str(script_path), "--run", "--epochs", str(epochs)]
        print(f"[ActiveLearning] Spawning trainer subprocess: {' '.join(cmd)}")
        log_file = self.base_dir / "runs" / "active_learning.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_file, "a", encoding="utf-8")
        subprocess.Popen(cmd, cwd=str(self.base_dir), stdout=log_f, stderr=subprocess.STDOUT)
        return True

active_trainer = ActiveLearningTrainer()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Active Learning Trainer")
    parser.add_argument("--run", action="store_true", help="Run training immediately")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs")
    args = parser.parse_args()

    if args.run:
        print(f"[ActiveLearning] Running training process for {args.epochs} epochs...")
        active_trainer.run_training(epochs=args.epochs)
    else:
        print("Testing Active Learning dataset preparation...")
        success, total = active_trainer.prepare_dataset()
        print("Dataset preparation result:", success, "Total samples:", total)
