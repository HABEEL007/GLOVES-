import os
import sys
import json
import time
import base64
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Blueprint

labeler_bp = Blueprint("labeler", __name__)
app = Flask(__name__)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Global configuration and state
BASE_DIR = Path(__file__).resolve().parent

# Prioritize Champion model, then active model, then YOLO26L base
from core.config import config
from core.active_learning.safe_daemon import safe_active_daemon

DEFAULT_MODEL_PATH = config.resolve_hand_model_path()

gloves_img_dir = BASE_DIR / "cctv_dataset" / "images" / "gloves"
if gloves_img_dir.exists() and len(list(gloves_img_dir.glob("*.jpg"))) > 0:
    default_img_dir = gloves_img_dir
    default_lbl_dir = BASE_DIR / "cctv_dataset" / "labels" / "gloves"
elif (BASE_DIR / "cctv_dataset" / "images" / "negatives").exists():
    default_img_dir = BASE_DIR / "cctv_dataset" / "images" / "negatives"
    default_lbl_dir = BASE_DIR / "cctv_dataset" / "labels" / "negatives"
else:
    default_img_dir = BASE_DIR / "cctv_dataset" / "images"
    default_lbl_dir = BASE_DIR / "cctv_dataset" / "labels"

CURRENT_CONFIG = {
    "images_dir": default_img_dir,
    "labels_dir": default_lbl_dir,
    "model_path": DEFAULT_MODEL_PATH,
    "conf_threshold": 0.25
}

# CCTV Stream Collector Integration
sys.path.insert(0, str(BASE_DIR / "tools"))
try:
    from cctv_stream_manager import cctv_collector
except Exception as e:
    print(f"[Warning] Could not load cctv_collector: {e}")
    cctv_collector = None

approved_counter = 0
AUTO_TRAIN_INTERVAL = 25

# Ensure folders exist
CURRENT_CONFIG["images_dir"].mkdir(parents=True, exist_ok=True)
CURRENT_CONFIG["labels_dir"].mkdir(parents=True, exist_ok=True)

# YOLO Model singleton
yolo_model = None
model_lock = threading.Lock()

def get_model():
    global yolo_model
    with model_lock:
        if yolo_model is None:
            m_path = CURRENT_CONFIG["model_path"]
            if m_path.exists():
                print(f"[AI] Loading YOLO weights from: {m_path}")
                yolo_model = YOLO(str(m_path))
            else:
                print(f"[AI WARNING] Model weights not found at {m_path}, attempting fallback.")
                yolo_model = YOLO("yolov8s.pt")
        return yolo_model

def get_reviewed_tracker_file():
    return CURRENT_CONFIG["labels_dir"] / ".reviewed_status.json"

def load_reviewed_tracker():
    f = get_reviewed_tracker_file()
    if f.exists():
        try:
            with open(f, "r", encoding="utf-8") as rf:
                return json.load(rf)
        except Exception:
            return {}
    return {}

def save_reviewed_tracker(data):
    f = get_reviewed_tracker_file()
    try:
        with open(f, "w", encoding="utf-8") as wf:
            json.dump(data, wf, indent=2)
    except Exception as e:
        print(f"Error saving tracker: {e}")

@labeler_bp.route("/studio")
@labeler_bp.route("/labeler")
def studio_index():
    return render_template("index.html")

@app.route("/")
def index():
    return render_template("index.html")

@labeler_bp.route("/api/folders", methods=["GET"])
def get_folders():
    """Finds available image directories across workspace for convenient selection."""
    candidates = [
        BASE_DIR / "cctv_dataset" / "images" / "gloves",
        BASE_DIR / "cctv_dataset" / "images" / "cctv_snapshots",
        BASE_DIR / "cctv_dataset" / "images" / "negatives",
        BASE_DIR / "cctv_dataset" / "images",
        BASE_DIR / "CPPE-Dataset" / "yolo_dataset_augmented" / "images" / "train",
        BASE_DIR / "CPPE-Dataset" / "yolo_dataset_augmented" / "images" / "val",
        BASE_DIR / "CPPE-Dataset" / "yolo_dataset" / "images" / "train",
        BASE_DIR / "CPPE-Dataset" / "yolo_dataset" / "images" / "val",
    ]
    folders = []
    for c in candidates:
        if c.exists():
            img_count = len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png")))
            folders.append({
                "path": str(c.relative_to(BASE_DIR)).replace("\\", "/"),
                "full_path": str(c),
                "image_count": img_count
            })
    return jsonify({
        "current_images": str(CURRENT_CONFIG["images_dir"].relative_to(BASE_DIR) if CURRENT_CONFIG["images_dir"].is_relative_to(BASE_DIR) else CURRENT_CONFIG["images_dir"]).replace("\\", "/"),
        "folders": folders
    })

@labeler_bp.route("/api/set_folder", methods=["POST"])
def set_folder():
    data = request.json or {}
    req_path = data.get("path")
    if not req_path:
        return jsonify({"error": "Path required"}), 400

    target_dir = BASE_DIR / req_path if not os.path.isabs(req_path) else Path(req_path)
    if not target_dir.exists():
        return jsonify({"error": "Directory does not exist"}), 404

    # Determine matching labels dir
    if "images" in str(target_dir):
        lbl_dir = Path(str(target_dir).replace("images", "labels"))
    else:
        lbl_dir = target_dir.parent / "labels"

    lbl_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_CONFIG["images_dir"] = target_dir
    CURRENT_CONFIG["labels_dir"] = lbl_dir

    return jsonify({
        "status": "success",
        "images_dir": str(target_dir),
        "labels_dir": str(lbl_dir)
    })

@labeler_bp.route("/api/images", methods=["GET"])
def list_images():
    img_dir = CURRENT_CONFIG["images_dir"]
    lbl_dir = CURRENT_CONFIG["labels_dir"]

    tracker = load_reviewed_tracker()
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(list(img_dir.glob(ext)))

    # Natural sorting
    image_files.sort(key=lambda x: x.name)

    items = []
    for f in image_files:
        stem = f.stem
        lbl_file = lbl_dir / f"{stem}.txt"

        has_label = lbl_file.exists()
        is_reviewed = tracker.get(f.name, {}).get("reviewed", False) or tracker.get(stem, {}).get("reviewed", False)

        box_count = 0
        if has_label:
            try:
                with open(lbl_file, "r") as lf:
                    box_count = len([line.strip() for line in lf if line.strip()])
            except Exception:
                pass

        status = "pending"
        if is_reviewed:
            status = "reviewed"
        elif has_label:
            status = "ai_draft"

        items.append({
            "name": f.name,
            "stem": stem,
            "has_label": has_label,
            "is_reviewed": is_reviewed,
            "box_count": box_count,
            "status": status
        })

    return jsonify({
        "total": len(items),
        "images": items
    })

@labeler_bp.route("/api/image/<path:filename>")
def serve_image(filename):
    file_path = CURRENT_CONFIG["images_dir"] / filename
    if not file_path.exists():
        return "Image not found", 404
    return send_file(str(file_path))

pose_model = None
def get_pose_model():
    global pose_model
    if pose_model is None:
        print("[AI] Loading YOLO Pose model for Hand/Wrist detection...")
        pose_model = YOLO("yolov8n-pose.pt")
    return pose_model

@labeler_bp.route("/api/boxes/<path:filename>")
def get_boxes(filename):
    img_path = CURRENT_CONFIG["images_dir"] / filename
    if not img_path.exists():
        return jsonify({"error": "Image not found"}), 404

    lbl_path = CURRENT_CONFIG["labels_dir"] / f"{img_path.stem}.txt"
    tracker = load_reviewed_tracker()
    is_reviewed = tracker.get(filename, {}).get("reviewed", False)

    force_ai = request.args.get("force_ai", "false").lower() == "true"
    target_class = int(request.args.get("class_id", 1)) # Default 1: Hand
    conf_thr = float(request.args.get("conf", CURRENT_CONFIG["conf_threshold"]))

    # Get image dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    img_h, img_w = img.shape[:2]

    CLASS_MAP = {0: "Gloves", 1: "Hand"}
    boxes = []

    # Case 1: Label file exists, has non-empty content, and NOT force_ai
    if lbl_path.exists() and not force_ai:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 0:
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(x) for x in parts[1:]]
                        boxes.append({
                            "class_id": cls_id,
                            "class_name": CLASS_MAP.get(cls_id, "Hand" if cls_id == 1 else "Gloves"),
                            "xc": xc,
                            "yc": yc,
                            "w": w,
                            "h": h,
                            "conf": 1.0,
                            "is_ai": False
                        })
                return jsonify({
                    "filename": filename,
                    "width": img_w,
                    "height": img_h,
                    "boxes": boxes,
                    "is_reviewed": is_reviewed,
                    "source": "existing_label"
                })

    # Case 2: Run AI Auto-Detection (either no label yet or force_ai requested)
    device = 0 if torch.cuda.is_available() else "cpu"

    if target_class == 1:
        # Check if fine-tuned model detects Hand (class 1) directly
        model = get_model()
        model_has_hands = hasattr(model, "names") and (1 in model.names or "Hand" in model.names.values())
        detected_hands = False

        if model_has_hands:
            try:
                results = model.predict(source=str(img_path), conf=conf_thr, verbose=False, device=device)
                res = results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        cid = int(box.cls[0].item()) if hasattr(box, "cls") else 0
                        if cid == 1:
                            detected_hands = True
                            xywhn = box.xywhn[0].tolist()
                            c = float(box.conf[0].item()) if hasattr(box, "conf") else 0.8
                            boxes.append({
                                "class_id": 1,
                                "class_name": "Hand",
                                "xc": xywhn[0],
                                "yc": xywhn[1],
                                "w": xywhn[2],
                                "h": xywhn[3],
                                "conf": round(c, 2),
                                "is_ai": True
                            })
            except Exception as e:
                print(f"[Warning] Error running fine-tuned hand model: {e}")

        # Fallback to Pose Wrists if no hands detected by fine-tuned model
        if not detected_hands:
            p_model = get_pose_model()
            results = p_model.predict(source=str(img_path), conf=conf_thr, verbose=False, device=device)
            res = results[0]
            if res.keypoints is not None and res.keypoints.xyn is not None:
                for kpts in res.keypoints.xyn:
                    for wrist_idx in (9, 10): # Left and Right Wrist
                        x, y = kpts[wrist_idx].tolist()
                        if x > 0.01 and y > 0.01:
                            # Estimate hand box around detected wrist
                            hw = 0.045
                            hh = 0.055
                            boxes.append({
                                "class_id": 1,
                                "class_name": "Hand",
                                "xc": max(hw/2, min(1 - hw/2, round(x, 6))),
                                "yc": max(hh/2, min(1 - hh/2, round(y, 6))),
                                "w": hw,
                                "h": hh,
                                "conf": 0.85,
                                "is_ai": True
                            })
    else:
        # Detect GLOVES using custom trained detector
        model = get_model()
        results = model.predict(source=str(img_path), conf=conf_thr, verbose=False, device=device)
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cid = int(box.cls[0].item()) if hasattr(box, "cls") else 0
                if cid == 0:
                    xywhn = box.xywhn[0].tolist()
                    c = float(box.conf[0].item()) if hasattr(box, "conf") else 0.8
                    boxes.append({
                        "class_id": 0,
                        "class_name": "Gloves",
                        "xc": xywhn[0],
                        "yc": xywhn[1],
                        "w": xywhn[2],
                        "h": xywhn[3],
                        "conf": round(c, 2),
                        "is_ai": True
                    })

    return jsonify({
        "filename": filename,
        "width": img_w,
        "height": img_h,
        "boxes": boxes,
        "is_reviewed": False,
        "source": "ai_prediction"
    })

@labeler_bp.route("/api/save", methods=["POST"])
def save_boxes():
    data = request.json or {}
    filename = data.get("filename")
    boxes = data.get("boxes", [])
    is_negative = data.get("is_negative", False)

    if not filename:
        return jsonify({"error": "Filename required"}), 400

    img_path = CURRENT_CONFIG["images_dir"] / filename
    lbl_path = CURRENT_CONFIG["labels_dir"] / f"{img_path.stem}.txt"

    lines = []
    if not is_negative and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.get("class_id", 0))
            xc = max(0.0, min(1.0, float(b["xc"])))
            yc = max(0.0, min(1.0, float(b["yc"])))
            w = max(0.001, min(1.0, float(b["w"])))
            h = max(0.001, min(1.0, float(b["h"])))
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    with open(lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Update tracker
    tracker = load_reviewed_tracker()
    was_already_reviewed = tracker.get(filename, {}).get("reviewed", False)
    tracker[filename] = {
        "reviewed": True,
        "is_negative": is_negative or len(lines) == 0,
        "box_count": len(lines)
    }
    save_reviewed_tracker(tracker)

    # Active learning auto-trigger every AUTO_TRAIN_INTERVAL newly approved frames
    global approved_counter
    if not was_already_reviewed:
        approved_counter += 1
        if approved_counter >= AUTO_TRAIN_INTERVAL:
            print(f"[SafeActiveLearning] {approved_counter} new frames approved! Auto-triggering safe candidate training...")
            safe_active_daemon.trigger_candidate_training_async(epochs=30)
            approved_counter = 0

    return jsonify({
        "status": "success",
        "saved_boxes": len(lines),
        "is_negative": is_negative or len(lines) == 0,
        "approved_counter": approved_counter
    })

@labeler_bp.route("/api/trigger_train", methods=["POST"])
def trigger_training():
    started = safe_active_daemon.trigger_candidate_training_async(epochs=30)
    if started:
        return jsonify({"status": "started", "message": "Safe Candidate Model training and validation gate started in background"})
    else:
        return jsonify({"status": "busy", "message": "Candidate training/validation is already in progress"})

@labeler_bp.route("/api/training_status")
def training_status():
    st = dict(safe_active_daemon.get_status())
    champ = st.get("champion_path")
    if champ and Path(champ).exists():
        champ_path = Path(champ)
        if CURRENT_CONFIG["model_path"] != champ_path:
            print(f"[SafeActiveLearning] Loading newly promoted Champion model: {champ_path}")
            CURRENT_CONFIG["model_path"] = champ_path
            global yolo_model
            with model_lock:
                yolo_model = None
            st["hot_reloaded"] = True

    st["approved_since_last_train"] = approved_counter
    st["auto_train_threshold"] = AUTO_TRAIN_INTERVAL
    return jsonify(st)

@labeler_bp.route("/api/cctv/start", methods=["POST"])
def start_cctv_stream():
    if not cctv_collector:
        return jsonify({"error": "CCTV Collector manager not loaded"}), 500
    data = request.json or {}
    rtsp_url = data.get("rtsp_url")
    fpm = int(data.get("frames_per_minute", 15))
    category = data.get("category", "negatives")

    success, msg = cctv_collector.start_capture(
        rtsp_url=rtsp_url,
        frames_per_minute=fpm,
        category=category
    )
    return jsonify({"success": success, "message": msg, "status": cctv_collector.get_status()})

@labeler_bp.route("/api/cctv/stop", methods=["POST"])
def stop_cctv_stream():
    if not cctv_collector:
        return jsonify({"error": "CCTV Collector manager not loaded"}), 500
    success, msg = cctv_collector.stop_capture()
    return jsonify({"success": success, "message": msg, "status": cctv_collector.get_status()})

@labeler_bp.route("/api/cctv/status")
def get_cctv_status():
    if not cctv_collector:
        return jsonify({"is_capturing": False, "error": "Not loaded"})
    return jsonify(cctv_collector.get_status())

@labeler_bp.route("/api/batch_auto_label", methods=["POST"])
def batch_auto_label():
    """Runs YOLO in background to pre-annotate all unreviewed images."""
    data = request.json or {}
    conf_thr = float(data.get("conf", CURRENT_CONFIG["conf_threshold"]))
    target_class = int(data.get("class_id", 1))

    def worker():
        img_dir = CURRENT_CONFIG["images_dir"]
        lbl_dir = CURRENT_CONFIG["labels_dir"]
        device = 0 if torch.cuda.is_available() else "cpu"

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_file in img_dir.glob(ext):
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    try:
                        lines = []
                        if target_class == 1:
                            p_model = get_pose_model()
                            res = p_model.predict(source=str(img_file), conf=conf_thr, verbose=False, device=device)[0]
                            if res.keypoints is not None and res.keypoints.xyn is not None:
                                for kpts in res.keypoints.xyn:
                                    for wrist_idx in (9, 10):
                                        x, y = kpts[wrist_idx].tolist()
                                        if x > 0.01 and y > 0.01:
                                            hw, hh = 0.045, 0.055
                                            xc = max(hw/2, min(1 - hw/2, x))
                                            yc = max(hh/2, min(1 - hh/2, y))
                                            lines.append(f"1 {xc:.6f} {yc:.6f} {hw:.6f} {hh:.6f}\n")
                        else:
                            model = get_model()
                            res = model.predict(source=str(img_file), conf=conf_thr, verbose=False, device=device)[0]
                            if res.boxes is not None:
                                for b in res.boxes:
                                    xywhn = b.xywhn[0].tolist()
                                    lines.append(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

                        with open(lbl_file, "w", encoding="utf-8") as lf:
                            lf.writelines(lines)
                    except Exception as err:
                        print(f"Error auto-labeling {img_file.name}: {err}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": f"Batch AI auto-labeling running in background as {'Hand' if target_class == 1 else 'Gloves'}"})

@labeler_bp.route("/api/stats")
def get_stats():
    img_dir = CURRENT_CONFIG["images_dir"]
    lbl_dir = CURRENT_CONFIG["labels_dir"]
    tracker = load_reviewed_tracker()

    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    total = len(all_images)
    reviewed = 0
    total_gloves = 0
    negatives = 0

    for f in all_images:
        lbl_file = lbl_dir / f"{f.stem}.txt"
        is_rev = tracker.get(f.name, {}).get("reviewed", False)
        if is_rev:
            reviewed += 1

        if lbl_file.exists():
            try:
                with open(lbl_file, "r") as lf:
                    lines = [l.strip() for l in lf if l.strip()]
                    if len(lines) > 0:
                        total_gloves += len(lines)
                    else:
                        negatives += 1
            except Exception:
                pass

    return jsonify({
        "total": total,
        "reviewed": reviewed,
        "pending": max(0, total - reviewed),
        "total_gloves": total_gloves,
        "negatives": negatives,
        "percent": int((reviewed / max(1, total)) * 100)
    })

@labeler_bp.route("/api/export")
def export_dataset():
    import io
    import zipfile
    from datetime import datetime

    img_dir = CURRENT_CONFIG["images_dir"]
    lbl_dir = CURRENT_CONFIG["labels_dir"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for img_path in images:
            # Include image
            zf.write(img_path, arcname=f"dataset/images/{img_path.name}")
            # Include label file if exists
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                zf.write(lbl_path, arcname=f"dataset/labels/{lbl_path.name}")
            else:
                zf.writestr(f"dataset/labels/{img_path.stem}.txt", "")

        # Include YOLO data.yaml configuration
        yaml_content = (
            "# YOLO Training Dataset Configuration\n"
            "path: dataset\n"
            "train: images\n"
            "val: images\n\n"
            "names:\n"
            "  0: Gloves\n"
            "  1: Hand\n"
            "nc: 2\n"
        )
        zf.writestr("dataset/data.yaml", yaml_content)

    zip_buffer.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=f"gloves_dataset_{stamp}.zip")

@labeler_bp.route("/api/detect_upload", methods=["POST"])
def detect_upload():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    conf_thr = float(request.form.get("conf", CURRENT_CONFIG["conf_threshold"]))
    device = 0 if torch.cuda.is_available() else "cpu"

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid or corrupted image"}), 400

        img_h, img_w = img.shape[:2]

        t0 = time.time()
        model = get_model()
        results = model.predict(source=img, conf=conf_thr, verbose=False, device=device)[0]
        dt = time.time() - t0

        boxes_info = []
        gloves_count = 0
        hand_count = 0
        CLASS_NAMES = {0: "Gloves", 1: "Hand"}

        annotated = img.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cid = int(b.cls[0].item()) if hasattr(b, "cls") else 0
                cname = CLASS_NAMES.get(cid, model.names.get(cid, "Gloves" if cid == 0 else "Hand"))
                c = float(b.conf[0].item()) if hasattr(b, "conf") else 0.8
                xyxy = b.xyxy[0].tolist()
                xywhn = b.xywhn[0].tolist()

                x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(img_w, int(xyxy[2])), min(img_h, int(xyxy[3]))

                if cid == 0:
                    gloves_count += 1
                    color = (34, 197, 94)  # Green for Gloves (BGR)
                else:
                    hand_count += 1
                    color = (0, 0, 240)    # Red for bare hand (BGR)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # Label tag banner
                tag = f"{cname} {c*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tag_y1 = max(0, y1 - th - 8)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                boxes_info.append({
                    "class_id": cid,
                    "class_name": cname,
                    "conf": round(c, 2),
                    "box_xyxy": [x1, y1, x2, y2],
                    "box_xywhn": [round(v, 6) for v in xywhn]
                })

        # Overall Status
        if gloves_count > 0 and hand_count == 0:
            status_text = f"COMPLIANT: {gloves_count} Glove(s) Detected (No Bare Hands)"
            badge_type = "success"
        elif gloves_count > 0 and hand_count > 0:
            status_text = f"MIXED: {gloves_count} Glove(s) and {hand_count} Bare Hand(s) Detected"
            badge_type = "warning"
        elif gloves_count == 0 and hand_count > 0:
            status_text = f"ALERT: {hand_count} Bare Hand(s) Detected (No Gloves Found!)"
            badge_type = "danger"
        else:
            status_text = "NO GLOVES OR HANDS DETECTED"
            badge_type = "neutral"

        # Encode to JPEG base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "width": img_w,
            "height": img_h,
            "latency_ms": round(dt * 1000, 1),
            "gloves_count": gloves_count,
            "hand_count": hand_count,
            "status_text": status_text,
            "badge_type": badge_type,
            "annotated_image": b64_img,
            "detections": boxes_info,
            "model_name": Path(CURRENT_CONFIG["model_path"]).name
        })

    except Exception as e:
        print(f"[Error in detect_upload] {e}")
        return jsonify({"error": str(e)}), 500

@labeler_bp.route("/api/save_detected_image", methods=["POST"])
def save_detected_image():
    """Saves a tested image & its YOLO labels into dataset for continuous learning."""
    data = request.json or {}
    category = data.get("category", "gloves")
    img_b64 = data.get("image_b64", "")
    boxes = data.get("boxes", [])
    orig_name = data.get("filename", "upload.jpg")

    if not img_b64:
        return jsonify({"error": "No image data"}), 400

    try:
        from datetime import datetime
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        target_cat = "negatives" if category == "negatives" else "gloves"
        dst_img_dir = BASE_DIR / "cctv_dataset" / "images" / target_cat
        dst_lbl_dir = BASE_DIR / "cctv_dataset" / "labels" / target_cat
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_stem = Path(orig_name).stem.replace(" ", "_")
        new_filename = f"cctv_{target_cat}_{clean_stem}_{stamp}.jpg"
        lbl_filename = f"{Path(new_filename).stem}.txt"

        cv2.imwrite(str(dst_img_dir / new_filename), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write labels
        lines = []
        if target_cat != "negatives":
            for b in boxes:
                xywhn = b.get("box_xywhn") or [b.get("xc"), b.get("yc"), b.get("w"), b.get("h")]
                cid = b.get("class_id", 0)
                if xywhn and len(xywhn) == 4:
                    lines.append(f"{cid} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

        with open(dst_lbl_dir / lbl_filename, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return jsonify({
            "success": True,
            "filename": new_filename,
            "category": target_cat,
            "boxes_saved": len(lines)
        })

    except Exception as e:
        print(f"[Error saving detected image] {e}")
        return jsonify({"error": str(e)}), 500

# Register blueprint for standalone execution
app.register_blueprint(labeler_bp)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GlovesLabeler AI Studio")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--images", type=str, default=None, help="Images directory")
    parser.add_argument("--labels", type=str, default=None, help="Labels directory")
    parser.add_argument("--conf", type=float, default=0.25, help="AI detection confidence threshold")

    args = parser.parse_args()
    if args.images:
        CURRENT_CONFIG["images_dir"] = Path(args.images).resolve()
    if args.labels:
        CURRENT_CONFIG["labels_dir"] = Path(args.labels).resolve()
    CURRENT_CONFIG["conf_threshold"] = args.conf

    print("=" * 65)
    print("🧤 Starting GlovesLabeler AI Studio")
    print(f"Local URL:   http://127.0.0.1:{args.port}")
    print(f"Images Dir:  {CURRENT_CONFIG['images_dir']}")
    print(f"Labels Dir:  {CURRENT_CONFIG['labels_dir']}")
    print("=" * 65)

    app.run(host="127.0.0.1", port=args.port, debug=False)
