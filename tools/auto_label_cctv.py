import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

def auto_label(
    images_dir="cctv_dataset/images/gloves",
    labels_dir="cctv_dataset/labels/gloves",
    model_path="CPPE-Dataset/runs/detect/gloves_detector_aug/weights/best.pt",
    conf=0.25
):
    from ultralytics import YOLO

    base_dir = Path(__file__).resolve().parent.parent
    img_path = base_dir / images_dir
    lbl_path = base_dir / labels_dir
    lbl_path.mkdir(parents=True, exist_ok=True)

    weights = base_dir / model_path
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found at: {weights}")

    print("=" * 60)
    print("🤖 AI Auto-Labeler for CCTV Dataset")
    print(f"Model:      {weights.resolve()}")
    print(f"Images Dir: {img_path.resolve()}")
    print(f"Labels Dir: {lbl_path.resolve()}")
    print(f"Min Conf:   {conf}")
    print("=" * 60)

    model = YOLO(str(weights))
    image_files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
    print(f"Found {len(image_files)} images to pre-annotate.")

    labeled_count = 0
    total_boxes = 0

    for img_file in tqdm(image_files, desc="Auto-labeling"):
        results = model.predict(source=str(img_file), conf=conf, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
        res = results[0]

        lbl_file = lbl_path / f"{img_file.stem}.txt"
        lines = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                # Class 0: Gloves
                # normalized xywh
                xywhn = box.xywhn[0].tolist()
                xc, yc, w, h = xywhn
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                total_boxes += 1
            labeled_count += 1

        with open(lbl_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE!")
    print(f"Total Images Processed: {len(image_files)}")
    print(f"Images with Gloves detected: {labeled_count}")
    print(f"Total Glove Boxes generated: {total_boxes}")
    print(f"Labels saved in: {lbl_path.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Pre-Annotation for CCTV Gloves Dataset")
    parser.add_argument("--images", type=str, default="cctv_dataset/images/gloves")
    parser.add_argument("--labels", type=str, default="cctv_dataset/labels/gloves")
    parser.add_argument("--model", type=str, default="CPPE-Dataset/runs/detect/gloves_detector_aug/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()
    auto_label(
        images_dir=args.images,
        labels_dir=args.labels,
        model_path=args.model,
        conf=args.conf
    )
