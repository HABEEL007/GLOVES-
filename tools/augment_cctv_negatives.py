import os
import sys
import random
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

# Configure safe UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def get_transforms():
    """6 distinct albumentations for CCTV background/negative samples."""
    return [
        ("aug1_hflip", A.Compose([
            A.HorizontalFlip(p=1.0)
        ])),
        ("aug2_color", A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.4, hue=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=35, val_shift_limit=25, p=1.0)
        ])),
        ("aug3_lighting", A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0)
        ])),
        ("aug4_blur", A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.7),
                A.GaussianBlur(blur_limit=5, p=0.3)
            ], p=1.0)
        ])),
        ("aug5_affine", A.Compose([
            A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), translate_percent=(-0.05, 0.05), p=1.0)
        ])),
        ("aug6_occlusion", A.Compose([
            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.06, 0.15), hole_width_range=(0.06, 0.15), fill=0, p=1.0)
        ]))
    ]

def augment_cctv_negatives(
    cctv_src="cctv_dataset/images/negatives",
    target_dataset="CPPE-Dataset/yolo_dataset_augmented",
    val_ratio=0.15,
    preview_output="cctv_negatives_preview.jpg"
):
    base_dir = Path(__file__).resolve().parent.parent
    src_folder = base_dir / cctv_src
    target = base_dir / target_dataset

    train_img = target / "images" / "train"
    train_lbl = target / "labels" / "train"
    val_img = target / "images" / "val"
    val_lbl = target / "labels" / "val"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    cctv_images = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png"))
    if not cctv_images:
        print(f"[ERROR] No images found in {src_folder.resolve()}")
        return

    print("=" * 65)
    print("🎨 CCTV Negative Samples Augmentation Pipeline")
    print(f"Source CCTV:     {src_folder.resolve()} ({len(cctv_images)} images)")
    print(f"Target Dataset:  {target.resolve()}")
    print("Augmentations:   6 distinct transforms per image (HFlip, Color, Lighting, Blur, Affine, Dropout)")
    print("Role in YOLO:    Negative/Background Samples (Empty Labels = Zero False Positives)")
    print("=" * 65)

    transforms = get_transforms()
    random.seed(42)
    random.shuffle(cctv_images)

    val_count = max(2, int(len(cctv_images) * val_ratio))
    val_originals = set(cctv_images[:val_count])

    total_added_train = 0
    total_added_val = 0
    preview_samples = []

    for img_path in cctv_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        is_val = img_path in val_originals
        dst_img_dir = val_img if is_val else train_img
        dst_lbl_dir = val_lbl if is_val else train_lbl

        # 1. Save original image & empty negative label
        orig_dest_name = f"cctv_neg_{img_path.name}"
        cv2.imwrite(str(dst_img_dir / orig_dest_name), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(dst_lbl_dir / f"{Path(orig_dest_name).stem}.txt", "w", encoding="utf-8") as f:
            pass  # Empty file = Negative sample

        if is_val:
            total_added_val += 1
        else:
            total_added_train += 1

        # 2. Generate 6 augmentations
        for aug_name, aug_pipeline in transforms:
            try:
                augmented = aug_pipeline(image=img_bgr)
                aug_img = augmented["image"]

                aug_file_name = f"cctv_neg_{img_path.stem}_{aug_name}.jpg"
                cv2.imwrite(str(dst_img_dir / aug_file_name), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Empty label for negative sample
                with open(dst_lbl_dir / f"{Path(aug_file_name).stem}.txt", "w", encoding="utf-8") as f:
                    pass

                if is_val:
                    total_added_val += 1
                else:
                    total_added_train += 1

                # Collect sample for preview
                if len(preview_samples) < 6 and img_path == cctv_images[0]:
                    preview_samples.append((aug_name, aug_img))

            except Exception as e:
                print(f"[Warning] Failed {aug_name} on {img_path.name}: {e}")

    # Generate preview grid
    if preview_samples:
        cell_w, cell_h = 320, 240
        grid = np.zeros((cell_h * 2, cell_w * 3, 3), dtype=np.uint8)
        for idx, (name, sample_img) in enumerate(preview_samples[:6]):
            r = idx // 3
            c = idx % 3
            resized = cv2.resize(sample_img, (cell_w, cell_h))
            cv2.putText(resized, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

        preview_file = base_dir / preview_output
        cv2.imwrite(str(preview_file), grid)
        print(f"Generated Augmentation Preview Grid: {preview_file.resolve()}")

    # Clear YOLO label caches to force rescan
    for cache_file in target.glob("labels/**/*.cache"):
        try:
            cache_file.unlink()
            print(f"Cleared stale cache: {cache_file.name}")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("AUGMENTATION & MERGE COMPLETE!")
    print(f"Original CCTV Frames:          {len(cctv_images)}")
    print(f"New CCTV Negatives in Train:   +{total_added_train}")
    print(f"New CCTV Negatives in Val:     +{total_added_val}")
    print(f"Total CCTV Negatives Created:  {total_added_train + total_added_val}")
    print(f"Total Dataset Images (Train):  {len(list(train_img.glob('*.jpg')))}")
    print(f"Total Dataset Images (Val):    {len(list(val_img.glob('*.jpg')))}")
    print("=" * 65)

if __name__ == "__main__":
    augment_cctv_negatives()
