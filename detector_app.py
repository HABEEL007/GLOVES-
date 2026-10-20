import os
import sys
import time
import base64
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Zero-latency FFMPEG RTSP options MUST be set before import cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0"
)

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# Safe UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.config import config
from core.capture.rtsp_stream import RTSPCaptureEngine
from core.pipeline import GloveCompliancePipeline
from core.database.db_service import db_service

class ZeroLatencyCCTVWorker:
    """
    Consumer thread receiving low-latency frames from RTSPCaptureEngine
    and executing the unified YOLO26L + ByteTrack + Temporal Compliance pipeline.
    """
    def __init__(self, rtsp_url: str = config.DEFAULT_RTSP_URL, camera_id: str = "301"):
        self.camera_id = camera_id
        self.capture_engine = RTSPCaptureEngine(rtsp_url=rtsp_url, camera_id=camera_id)
        self.pipeline = GloveCompliancePipeline(camera_id=camera_id)
        
        self.is_running = False
        self.jpeg_lock = threading.Lock()
        self.new_jpeg_condition = threading.Condition(threading.Lock())
        self.latest_jpeg_bytes: Optional[bytes] = None
        
        # Auto-Dataset Screenshot on Glove Detection (Dataset Collection for Training/Annotations)
        self.auto_capture_enabled: bool = False
        self.auto_capture_mode: str = "all_gloves"  # "all_gloves" (Green Box Compliant) or "green_colored_gloves"
        self.auto_capture_interval: float = 1.5      # Minimum cooldown in seconds between auto-captures
        self.last_auto_capture_time: float = 0.0
        self.auto_capture_count: int = 0
        self.last_auto_captured_file: Optional[str] = None
        self.recent_auto_capture_event: Optional[dict] = None

        self.stats = {
            "status": "CONNECTING",
            "badge_type": "neutral",
            "active_violations": 0,
            "total_violations_today": 0,
            "right_gloves_count": 0,
            "left_hand_bare_count": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "pose_ms": 0.0,
            "det_ms": 0.0,
            "track_ms": 0.0,
            "tracked_people": 0,
            "tracks": [],
            "model_version": self.pipeline.model_version,
            "channel": camera_id,
            "rule_mode": "Right-Hand Glove Mandatory (Left Hand Exempt)",
            "is_connected": False,
            "timestamp": ""
        }
        self.consumer_thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.capture_engine.start()
        
        self.consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="ComplianceConsumerWorker"
        )
        self.consumer_thread.start()
        print(f"[+] Zero-Latency CCTV Worker running on Camera {self.camera_id}!")

    def stop(self):
        self.is_running = False
        self.capture_engine.stop()
        with self.new_jpeg_condition:
            self.new_jpeg_condition.notify_all()

    def set_conf(self, conf: float):
        self.pipeline.set_hand_conf(conf)

    def set_rule_mode(self, right_hand_only: bool):
        self.pipeline.set_rule_mode(right_hand_only)

    def set_auto_capture(self, enabled: bool, mode: Optional[str] = None, interval: Optional[float] = None):
        self.auto_capture_enabled = bool(enabled)
        if mode in ("all_gloves", "green_colored_gloves"):
            self.auto_capture_mode = mode
        if interval is not None and float(interval) >= 0.3:
            self.auto_capture_interval = float(interval)
        print(f"[Worker] Auto-Capture: enabled={self.auto_capture_enabled}, mode={self.auto_capture_mode}, interval={self.auto_capture_interval}s")

    def _check_auto_capture(self, raw_frame: np.ndarray, frame_stats: dict):
        if not self.auto_capture_enabled or raw_frame is None or raw_frame.size == 0:
            return

        now = time.time()
        if (now - self.last_auto_capture_time) < self.auto_capture_interval:
            return

        # Check glove detection trigger
        has_gloves = frame_stats.get("has_gloves", False)
        has_green_gloves = frame_stats.get("has_green_gloves", False)

        trigger = False
        if self.auto_capture_mode == "green_colored_gloves":
            trigger = has_green_gloves
        else:  # "all_gloves" (any glove detected, appearing as green bounding box)
            trigger = has_gloves

        if not trigger:
            return

        # Perform Auto-Capture: Save full-res raw frame and YOLO label to dataset
        try:
            target_img_dir = config.BASE_DIR / "cctv_dataset" / "images" / "gloves"
            target_lbl_dir = config.BASE_DIR / "cctv_dataset" / "labels" / "gloves"
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)

            self.auto_capture_count += 1
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            src_tag = "vid" if getattr(self.capture_engine, "source_type", "video") == "video" else "cctv"
            filename = f"auto_{src_tag}_glove_{stamp}_{self.auto_capture_count:04d}.jpg"
            lbl_filename = f"auto_{src_tag}_glove_{stamp}_{self.auto_capture_count:04d}.txt"

            img_path = target_img_dir / filename
            lbl_path = target_lbl_dir / lbl_filename

            # Save clean full-res raw frame (ideal for annotations & YOLO retraining)
            cv2.imwrite(str(img_path), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write YOLO format labels (Class 0 = gloved_hand)
            # Format: <class_id> <x_center> <y_center> <width> <height>
            h, w = raw_frame.shape[:2]
            yolo_lines = []
            for g in frame_stats.get("detected_gloves", []):
                bx = g["box"]
                x1, y1, x2, y2 = bx[0], bx[1], bx[2], bx[3]
                xc = max(0.0, min(1.0, ((x1 + x2) / 2.0) / w))
                yc = max(0.0, min(1.0, ((y1 + y2) / 2.0) / h))
                bw = max(0.0, min(1.0, (x2 - x1) / w))
                bh = max(0.0, min(1.0, (y2 - y1) / h))
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            with open(lbl_path, "w", encoding="utf-8") as lf:
                lf.writelines(yolo_lines)

            self.last_auto_capture_time = now
            self.last_auto_captured_file = filename
            self.recent_auto_capture_event = {
                "filename": filename,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "count": self.auto_capture_count,
                "gloves_in_frame": len(frame_stats.get("detected_gloves", [])),
                "time": now
            }
            print(f"[AutoCapture #{self.auto_capture_count}] Auto-saved screenshot to dataset: {filename} ({len(yolo_lines)} glove box annotations)")

        except Exception as e:
            print(f"[AutoCapture Error] {e}")

    def switch_source(self, new_source: str, camera_id: Optional[str] = None, loop: bool = True):
        self.capture_engine.switch_source(new_source, camera_id=camera_id, loop_video=loop)
        self.camera_id = self.capture_engine.camera_id
        self.pipeline.camera_id = self.camera_id
        self.stats["channel"] = self.camera_id
        self.stats["source_type"] = self.capture_engine.source_type
        print(f"[Worker] Switched source to: {self.camera_id} ({self.capture_engine.source_type})")

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.jpeg_lock:
            return self.latest_jpeg_bytes

    def get_raw_frame(self) -> Optional[np.ndarray]:
        frame, _, _ = self.capture_engine.get_latest_frame()
        return frame

    def get_stats(self) -> dict:
        s = dict(self.stats)
        s["source_type"] = getattr(self.capture_engine, "source_type", "rtsp")
        s["source_name"] = self.camera_id
        s["is_paused"] = getattr(self.capture_engine, "is_paused", False)
        s["loop_video"] = getattr(self.capture_engine, "loop_video", True)
        s["auto_capture_enabled"] = self.auto_capture_enabled
        s["auto_capture_count"] = self.auto_capture_count
        s["auto_capture_mode"] = self.auto_capture_mode
        s["auto_capture_interval"] = self.auto_capture_interval
        s["last_auto_capture_file"] = self.last_auto_captured_file
        s["auto_capture_dir"] = "cctv_dataset/images/gloves"

        if self.recent_auto_capture_event and (time.time() - self.recent_auto_capture_event.get("time", 0)) < 2.0:
            s["just_captured"] = self.recent_auto_capture_event
        else:
            s["just_captured"] = None

        return s

    def _consumer_loop(self):
        while self.is_running:
            got_frame = self.capture_engine.wait_for_frame(timeout=0.05)
            if not got_frame:
                if not self.capture_engine.is_connected:
                    self.stats["status"] = "CAMERA OFFLINE / RECONNECTING"
                    self.stats["badge_type"] = "danger"
                    self.stats["is_connected"] = False
                continue

            frame, frame_time, frame_num = self.capture_engine.get_latest_frame()
            if frame is None or frame.size == 0:
                continue

            # Always process the most recent frame for all source types.
            # If a newer frame arrived while we were fetching (pipeline slower than FPS),
            # grab that instead so playback never falls behind real-time.
            if self.capture_engine.has_new_frame():
                frame, frame_time, frame_num = self.capture_engine.get_latest_frame()
                if frame is None or frame.size == 0:
                    continue

            # Run through unified pipeline (wrapped to prevent thread death on any single frame error)
            try:
                annotated_frame, frame_stats = self.pipeline.process_frame(frame, frame_num)
                frame_stats["is_connected"] = True

                # Automatically save screenshot into dataset when gloves are detected!
                self._check_auto_capture(frame, frame_stats)

                # Compress to JPEG for Web Stream (Quality 82 — good sharpness, still fast)
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                jpeg_bytes = buf.tobytes()

                with self.jpeg_lock:
                    self.latest_jpeg_bytes = jpeg_bytes

                with self.new_jpeg_condition:
                    self.new_jpeg_condition.notify_all()

                self.stats = frame_stats

            except Exception as pipeline_err:
                print(f"[ConsumerLoop] Pipeline error on frame {frame_num}: {pipeline_err}")
                # Keep running — bad frame silently skipped, stream stays alive

# Initialize Worker
worker = ZeroLatencyCCTVWorker(rtsp_url=config.DEFAULT_RTSP_URL, camera_id="301")
worker.start()

# Flask Web Server
app = Flask(__name__, template_folder="templates", static_folder="static")

# Mount Annotation & Labeler Studio directly into unified detector platform
from labeler_studio import labeler_bp
app.register_blueprint(labeler_bp)

@app.route("/")
def index():
    return render_template("detector.html")

@app.route("/api/video_feed")
def video_feed():
    def generate():
        last_frame = None
        while True:
            with worker.new_jpeg_condition:
                worker.new_jpeg_condition.wait(timeout=0.08)
                frame = worker.get_latest_jpeg()

            if frame is not None and frame != last_frame:
                last_frame = frame
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                       frame + b"\r\n")

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@app.route("/api/live_stats")
def live_stats():
    resp = jsonify(worker.get_stats())
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api/stream/set_conf", methods=["POST"])
def set_conf():
    data = request.get_json(silent=True) or request.form
    conf = float(data.get("conf", 0.25))
    worker.set_conf(conf)
    return jsonify({"success": True, "conf": worker.pipeline.conf_threshold})

@app.route("/api/stream/set_rule", methods=["POST"])
def set_rule():
    data = request.get_json(silent=True) or request.form
    right_only = bool(data.get("right_hand_only", True))
    worker.set_rule_mode(right_only)
    return jsonify({"success": True, "right_hand_only": worker.pipeline.right_hand_only_rule})

@app.route("/api/stream/snapshot", methods=["POST", "GET"])
def stream_snapshot():
    raw = worker.get_raw_frame()
    if raw is None:
        return jsonify({"success": False, "error": "No frame available from CCTV stream"}), 404

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = config.SNAPSHOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"cctv301_snap_{stamp}.jpg"
    filepath = target_dir / filename
    cv2.imwrite(str(filepath), raw)

    return jsonify({
        "success": True,
        "message": f"Full-res snapshot saved: {filename}",
        "filepath": str(filepath),
        "filename": filename
    })

@app.route("/api/stream/auto_capture", methods=["GET", "POST"])
def stream_auto_capture():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        enabled = data.get("enabled")
        mode = data.get("mode")
        interval = data.get("interval")
        reset_cnt = data.get("reset_count")

        if enabled is not None:
            worker.auto_capture_enabled = bool(enabled)
        if mode in ("all_gloves", "green_colored_gloves"):
            worker.auto_capture_mode = mode
        if interval is not None:
            try:
                worker.auto_capture_interval = max(0.4, float(interval))
            except (ValueError, TypeError):
                pass
        if reset_cnt:
            worker.auto_capture_count = 0

    return jsonify({
        "success": True,
        "enabled": worker.auto_capture_enabled,
        "mode": worker.auto_capture_mode,
        "interval": worker.auto_capture_interval,
        "count": worker.auto_capture_count,
        "last_file": worker.last_auto_captured_file,
        "target_folder": "cctv_dataset/images/gloves"
    })

@app.route("/api/stream/auto_capture_history", methods=["GET"])
def stream_auto_capture_history():
    target_img_dir = config.BASE_DIR / "cctv_dataset" / "images" / "gloves"
    files = []
    if target_img_dir.exists():
        for f in sorted(target_img_dir.glob("auto_*.jpg"), key=os.path.getmtime, reverse=True)[:40]:
            files.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "time": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
    return jsonify({
        "success": True,
        "count": len(files),
        "total_captured": worker.auto_capture_count,
        "files": files,
        "target_folder": str(target_img_dir)
    })

@app.route("/api/dataset_image/<category>/<path:filename>")
def serve_dataset_image(category, filename):
    folder = config.BASE_DIR / "cctv_dataset" / "images" / category
    return send_from_directory(str(folder), filename)

@app.route("/api/stream/sources", methods=["GET"])
def get_sources():
    videos = []
    if config.VIDEOS_DIR.exists():
        for f in config.VIDEOS_DIR.glob("*.*"):
            if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    return jsonify({
        "current_source": worker.camera_id,
        "source_type": getattr(worker.capture_engine, "source_type", "rtsp"),
        "is_paused": getattr(worker.capture_engine, "is_paused", False),
        "loop_video": getattr(worker.capture_engine, "loop_video", True),
        "available_videos": videos
    })

@app.route("/api/stream/set_source", methods=["POST"])
def set_source():
    data = request.get_json(silent=True) or request.form
    source_type = data.get("type", "video").lower()
    loop = bool(data.get("loop", True))

    if source_type == "rtsp":
        rtsp_url = data.get("url", config.DEFAULT_RTSP_URL)
        worker.switch_source(rtsp_url, camera_id="301", loop=False)
        return jsonify({"success": True, "source": "CCTV Channel 301", "type": "rtsp"})
    elif source_type == "webcam":
        cam_idx = str(data.get("index", "0"))
        worker.switch_source(cam_idx, camera_id=f"Webcam {cam_idx}", loop=False)
        return jsonify({"success": True, "source": f"Webcam {cam_idx}", "type": "webcam"})
    elif source_type == "video":
        video_name = data.get("video_name")
        video_path = config.VIDEOS_DIR / video_name if video_name else None
        if not video_path or not video_path.exists():
            raw_path = Path(data.get("path", ""))
            if raw_path.exists():
                video_path = raw_path
            else:
                return jsonify({"success": False, "error": f"Video file not found: {video_name}"}), 404
        worker.switch_source(str(video_path), camera_id=video_path.name, loop=loop)
        return jsonify({"success": True, "source": video_path.name, "type": "video"})
    else:
        return jsonify({"success": False, "error": f"Unknown source type: {source_type}"}), 400

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        return jsonify({"success": False, "error": f"Unsupported video format: {ext}. Allowed formats: MP4, AVI, MOV, MKV, WEBM"}), 400

    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = config.VIDEOS_DIR / safe_name
    file.save(str(target_path))

    # Switch stream to this video immediately
    worker.switch_source(str(target_path), camera_id=safe_name, loop=True)

    return jsonify({
        "success": True,
        "filename": safe_name,
        "message": f"Video '{safe_name}' uploaded. Real-time detection active!",
        "path": str(target_path)
    })

@app.route("/api/stream/control", methods=["POST"])
def stream_control():
    data = request.get_json(silent=True) or request.form
    action = data.get("action", "")

    if action == "pause":
        is_paused = worker.capture_engine.toggle_pause()
        return jsonify({"success": True, "action": "pause", "is_paused": is_paused})
    elif action == "restart":
        worker.capture_engine.restart_playback()
        return jsonify({"success": True, "action": "restart"})
    elif action == "toggle_loop":
        worker.capture_engine.loop_video = not worker.capture_engine.loop_video
        return jsonify({"success": True, "loop_video": worker.capture_engine.loop_video})
    else:
        return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

@app.route("/api/violations", methods=["GET"])
def get_violations():
    limit = int(request.args.get("limit", 15))
    violations = db_service.get_recent_violations(limit=limit)
    return jsonify({
        "success": True,
        "count": len(violations),
        "violations": violations
    })

@app.route("/api/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory(str(config.EVIDENCE_DIR), filename)

@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "hand_model": str(worker.pipeline.hand_detector.model_path),
        "pose_model": str(worker.pipeline.pose_detector.model_path),
        "tracker": "ByteTrack (Kalman + SciPy)",
        "model_version": worker.pipeline.model_version,
        "device": str(config.DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "channel": worker.camera_id,
        "rule": "Right-Hand Glove Mandatory (Left Hand Exempt)",
        "classes": config.CLASS_NAMES
    })

@app.route("/api/detect", methods=["POST"])
def detect_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    try:
        conf_thr = float(request.form.get("conf", worker.pipeline.conf_threshold))
    except (ValueError, TypeError):
        conf_thr = 0.25

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400

        # Run upload through pipeline
        worker.set_conf(conf_thr)
        annotated_frame, frame_stats = worker.pipeline.process_frame(img, 1)

        _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "width": img.shape[1],
            "height": img.shape[0],
            "latency_ms": frame_stats["latency_ms"],
            "status_text": frame_stats["status"],
            "badge_type": frame_stats["badge_type"],
            "active_violations": frame_stats["active_violations"],
            "right_gloves_count": frame_stats["right_gloves_count"],
            "left_hand_bare_count": frame_stats["left_hand_bare_count"],
            "tracked_people": frame_stats["tracked_people"],
            "tracks": frame_stats["tracks"],
            "annotated_image": b64_img
        })

    except Exception as e:
        print(f"[Error in /api/detect] {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Latency Live CCTV AI Detector Server with Right-Hand Rule")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    print(f"================================================================")
    print(f"  AI ZERO-LATENCY Live CCTV Detector (YOLO26L + ByteTrack)     ")
    print(f"  Rule          : Right Hand Glove Mandatory (Left Hand Exempt)")
    print(f"  Tracker       : ByteTrack Persistent Track IDs               ")
    print(f"  Live Stream   : CCTV Channel 301 (RTSP 1080p)                ")
    print(f"  Local URL     : http://127.0.0.1:{args.port}                 ")
    print(f"  Device        : {config.DEVICE} (CUDA GPU Active)            ")
    print(f"================================================================")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        worker.stop()
