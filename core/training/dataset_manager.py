import os
import re
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..config import config


def _count_class_in_labels(label_files: List[Path]) -> Dict[int, int]:
    """Count annotation instances per class across a list of label files."""
    counts: Dict[int, int] = defaultdict(int)
    for lbl in label_files:
        try:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        except Exception:
            pass
    return dict(counts)

class DatasetManager:
    """
    Builds versioned datasets with session-based splitting (70% train, 15% val, 15% test).
    Ensures black glove specialization and includes hard negatives (dark clothing, shadows, jewelry trays).
    """
    def __init__(self, target_dir: Optional[Path] = None):
        self.target_dir = target_dir or config.ACTIVE_DATASET_DIR
        self.cctv_dir = config.CCTV_DATASET_DIR
        self.cppe_dir = config.CPPE_DATASET_DIR

    def extract_session_id(self, filename: str) -> str:
        """
        Extracts session or video identifier from filename.
        Example: cctv_negatives_20260923_164009_00001.jpg -> '20260923_164009'
        """
        match = re.search(r"(\d{8}_\d{6})", filename)
        if match:
            return match.group(1)
        # Fallback to prefix
        parts = filename.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else "session_default"

    def prepare_combined_dataset(
        self,
        gloves_bootstrap_limit: int = 600,
        cctv_limit: int = 1500
    ) -> Tuple[Path, Dict[str, int]]:
        print("[DatasetManager] Preparing Session-Based Split Combined Dataset...")
        
        train_img = self.target_dir / "images" / "train"
        train_lbl = self.target_dir / "labels" / "train"
        val_img = self.target_dir / "images" / "val"
        val_lbl = self.target_dir / "labels" / "val"
        test_img = self.target_dir / "images" / "test"
        test_lbl = self.target_dir / "labels" / "test"

        for d in [train_img, train_lbl, val_img, val_lbl, test_img, test_lbl]:
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.*"):
                try:
                    old.unlink()
                except Exception:
                    pass

        # 1. Group CCTV images by session/video
        session_to_pairs: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
        
        # CCTV Negatives (Bare Hands & Workstation Hard Negatives)
        cctv_neg_img = self.cctv_dir / "images" / "negatives"
        cctv_neg_lbl = self.cctv_dir / "labels" / "negatives"
        if cctv_neg_img.exists():
            for img_path in cctv_neg_img.glob("*.jpg"):
                lbl_path = cctv_neg_lbl / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    # Create empty label file for hard negative background
                    lbl_path.write_text("", encoding="utf-8")
                sess = self.extract_session_id(img_path.name)
                session_to_pairs[sess].append((img_path, lbl_path))

        # CCTV Gloves
        cctv_glv_img = self.cctv_dir / "images" / "gloves"
        cctv_glv_lbl = self.cctv_dir / "labels" / "gloves"
        if cctv_glv_img.exists():
            for img_path in cctv_glv_img.glob("*.jpg"):
                lbl_path = cctv_glv_lbl / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    sess = self.extract_session_id(img_path.name)
                    session_to_pairs[sess].append((img_path, lbl_path))

        # 2. Stratified Session-Based Split
        # Score each session by gloved_hand count so val/test get a proportional
        # share of gloved examples — prevents the pure-random split from starving
        # the minority class in val (previously: val had 16 gloved vs 273 bare).
        def _session_glove_score(sess: str) -> int:
            lbl_files = [pair[1] for pair in session_to_pairs[sess]]
            return _count_class_in_labels(lbl_files).get(0, 0)  # class 0 = gloved_hand

        sessions = sorted(session_to_pairs.keys(), key=_session_glove_score, reverse=True)
        random.seed(42)

        n_sess = len(sessions)
        n_val_sess = max(1, int(n_sess * 0.15))
        n_test_sess = max(1, int(n_sess * 0.15))

        # Interleaved assignment: deal sessions like a deck of cards so glove-rich
        # sessions are spread across all three splits instead of piling into train.
        val_sessions: set = set()
        test_sessions: set = set()
        train_sessions: set = set()

        for i, sess in enumerate(sessions):
            if len(val_sessions) < n_val_sess and i % 3 == 0:
                val_sessions.add(sess)
            elif len(test_sessions) < n_test_sess and i % 3 == 1:
                test_sessions.add(sess)
            else:
                train_sessions.add(sess)

        # Any leftover sessions (due to modulo alignment) go to train
        for sess in sessions:
            if sess not in val_sessions and sess not in test_sessions:
                train_sessions.add(sess)

        print(f"[DatasetManager] Stratified split → train: {len(train_sessions)} sessions, "
              f"val: {len(val_sessions)} sessions, test: {len(test_sessions)} sessions")

        counts = {"train": 0, "val": 0, "test": 0, "hard_negatives": 0}

        def copy_pairs(pairs: List[Tuple[Path, Path]], dest_i: Path, dest_l: Path, split_name: str):
            for s_img, s_lbl in pairs:
                shutil.copy2(s_img, dest_i / s_img.name)
                shutil.copy2(s_lbl, dest_l / f"{s_img.stem}.txt")
                counts[split_name] += 1
                if s_lbl.stat().st_size == 0:
                    counts["hard_negatives"] += 1

        for sess, pairs in session_to_pairs.items():
            if sess in test_sessions:
                copy_pairs(pairs, test_img, test_lbl, "test")
            elif sess in val_sessions:
                copy_pairs(pairs, val_img, val_lbl, "val")
            else:
                copy_pairs(pairs, train_img, train_lbl, "train")

        # 3. Add CPPE Bootstrap Gloves to Training to Prevent Catastrophic Forgetting
        cppe_img_dir = self.cppe_dir / "yolo_dataset_augmented" / "images" / "train"
        cppe_lbl_dir = self.cppe_dir / "yolo_dataset_augmented" / "labels" / "train"
        if cppe_img_dir.exists():
            cppe_pairs = []
            for g_img in cppe_img_dir.glob("*.jpg"):
                g_lbl = cppe_lbl_dir / f"{g_img.stem}.txt"
                if g_lbl.exists():
                    cppe_pairs.append((g_img, g_lbl))
            
            if len(cppe_pairs) > gloves_bootstrap_limit:
                cppe_pairs = random.sample(cppe_pairs, gloves_bootstrap_limit)

            for s_img, s_lbl in cppe_pairs:
                shutil.copy2(s_img, train_img / s_img.name)
                shutil.copy2(s_lbl, train_lbl / f"{s_img.stem}.txt")
                counts["train"] += 1

        # Clear Ultralytics label caches
        for cache in self.target_dir.glob("labels/**/*.cache"):
            try:
                cache.unlink()
            except Exception:
                pass

        # 4. Generate Standard Dataset YAML with exact class names
        yaml_path = self.target_dir / "active_dataset.yaml"
        yaml_content = f"""path: {self.target_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: gloved_hand
  1: bare_hand
nc: 2
"""
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"[DatasetManager] Dataset ready! Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']} (Hard Negatives: {counts['hard_negatives']})")
        return yaml_path, counts
