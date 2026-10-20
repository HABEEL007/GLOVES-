import os
import shutil
import random
import argparse
from pathlib import Path

def merge_cctv_to_augmented(
    cctv_dir="cctv_dataset",
    target_dataset="CPPE-Dataset/yolo_dataset_augmented",
    val_ratio=0.15
):
    base_dir = Path(__file__).resolve().parent.parent
    cctv = base_dir / cctv_dir
    target = base_dir / target_dataset

    train_img = target / "images" / "train"
    train_lbl = target / "labels" / "train"
    val_img = target / "images" / "val"
    val_lbl = target / "labels" / "val"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔄 Merging CCTV Dataset into Augmented Dataset Pipeline")
    print(f"CCTV Source:   {cctv.resolve()}")
    print(f"Target Dataset: {target.resolve()}")
    print("=" * 60)

    # Collect all image-label pairs from gloves and negatives
    pairs = []
    for cat in ["gloves", "negatives"]:
        img_folder = cctv / "images" / cat
        lbl_folder = cctv / "labels" / cat

        if not img_folder.exists():
            continue

        for img_file in img_folder.glob("*.jpg"):
            lbl_file = lbl_folder / f"{img_file.stem}.txt"
            pairs.append((img_file, lbl_file, cat))

    if len(pairs) == 0:
        print("❌ No images found in cctv_dataset. Collect frames first using collect_cctv_dataset.py!")
        return

    random.seed(42)
    random.shuffle(pairs)

    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"Total CCTV Frames to Merge: {len(pairs)}")
    print(f"  -> Adding to Train: {len(train_pairs)}")
    print(f"  -> Adding to Val:   {len(val_pairs)}")

    for img_file, lbl_file, cat in train_pairs:
        shutil.copy2(img_file, train_img / img_file.name)
        if lbl_file.exists():
            shutil.copy2(lbl_file, train_lbl / lbl_file.name)
        else:
            with open(train_lbl / f"{img_file.stem}.txt", "w") as f:
                pass

    for img_file, lbl_file, cat in val_pairs:
        shutil.copy2(img_file, val_img / img_file.name)
        if lbl_file.exists():
            shutil.copy2(lbl_file, val_lbl / lbl_file.name)
        else:
            with open(val_lbl / f"{img_file.stem}.txt", "w") as f:
                pass

    # Remove stale YOLO caches so YOLO rescans the new data
    for cache_file in target.glob("labels/*.cache"):
        try:
            cache_file.unlink()
            print(f"Removed cache: {cache_file.name}")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print(f"New Total Train Images: {len(list(train_img.glob('*.jpg')))}")
    print(f"New Total Val Images:   {len(list(val_img.glob('*.jpg')))}")
    print("=" * 60)
    print("Ready to retrain! Run:")
    print("python train_gloves.py --epochs 30 --batch 8 --name gloves_detector_cctv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CCTV Dataset into YOLO Training Set")
    parser.add_argument("--cctv", type=str, default="cctv_dataset")
    parser.add_argument("--target", type=str, default="CPPE-Dataset/yolo_dataset_augmented")
    parser.add_argument("--val-ratio", type=float, default=0.15)

    args = parser.parse_args()
    merge_cctv_to_augmented(
        cctv_dir=args.cctv,
        target_dataset=args.target,
        val_ratio=args.val_ratio
    )
