import os
import sys
import time
import urllib.parse
import threading
from datetime import datetime
from pathlib import Path
import cv2

# Configure safe UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Robust FFMPEG RTSP parameters for OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;500000|probesize;32768|analyzeduration;100000"
)

def sanitize_rtsp_url(url: str) -> str:
    """Encodes special characters (like '@') in RTSP credentials."""
    prefix = "rtsp://"
    if not url.lower().startswith(prefix):
        return url
    rest = url[len(prefix):]
    if rest.count("@") > 1:
        last_at = rest.rfind("@")
        creds = rest[:last_at]
        host_path = rest[last_at + 1:]
        if ":" in creds:
            colon_idx = creds.find(":")
            user = creds[:colon_idx]
            pwd = creds[colon_idx + 1:]
            encoded_pwd = urllib.parse.quote(pwd)
            return f"rtsp://{user}:{encoded_pwd}@{host_path}"
    return url

class CCTVStreamCollector:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "cctv_dataset"
        self.is_capturing = False
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Configurable settings
        self.rtsp_url = "rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/301"
        self.frames_per_minute = 4
        self.category = "negatives"
        self.captured_count = 0
        self.error_msg = None
        self.last_capture_time = None
        self.start_time = None

    def get_status(self):
        with self.lock:
            elapsed = int(time.time() - self.start_time) if self.start_time and self.is_capturing else 0
            return {
                "is_capturing": self.is_capturing,
                "captured_count": self.captured_count,
                "rtsp_url": self.rtsp_url,
                "frames_per_minute": self.frames_per_minute,
                "category": self.category,
                "interval_seconds": round(60.0 / max(1, self.frames_per_minute), 1),
                "error": self.error_msg,
                "last_capture_time": self.last_capture_time,
                "elapsed_seconds": elapsed
            }

    def start_capture(self, rtsp_url=None, frames_per_minute=15, category="negatives"):
        with self.lock:
            if self.is_capturing:
                return False, "Capture already running"

            if rtsp_url:
                self.rtsp_url = rtsp_url.strip()
            self.frames_per_minute = max(1, min(120, int(frames_per_minute)))
            self.category = category if category in ["negatives", "gloves"] else "negatives"
            self.captured_count = 0
            self.error_msg = None
            self.is_capturing = True
            self.stop_event.clear()
            self.start_time = time.time()

        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        print(f"[CCTV Manager] Started stream capture: {self.rtsp_url} ({self.frames_per_minute} frames/min -> {self.category})")
        return True, "Capture started"

    def stop_capture(self):
        with self.lock:
            if not self.is_capturing:
                return False, "Not capturing"
            self.stop_event.set()
            self.is_capturing = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        print(f"[CCTV Manager] Capture stopped. Total frames captured: {self.captured_count}")
        return True, f"Capture stopped. Saved {self.captured_count} frames."

    def _capture_worker(self):
        sanitized_url = sanitize_rtsp_url(self.rtsp_url)
        cap = cv2.VideoCapture(sanitized_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            with self.lock:
                self.is_capturing = False
                self.error_msg = f"Failed to connect to RTSP stream: {self.rtsp_url}"
            print(f"[CCTV Manager ERROR] {self.error_msg}")
            return

        interval = 60.0 / max(1, self.frames_per_minute) # 15 fpm = 4.0 sec
        last_save_time = 0.0
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        img_dir = self.dataset_dir / "images" / self.category
        lbl_dir = self.dataset_dir / "labels" / self.category
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

                now = time.time()
                if now - last_save_time >= interval:
                    last_save_time = now
                    with self.lock:
                        self.captured_count += 1
                        cnt = self.captured_count
                        self.last_capture_time = datetime.now().strftime("%H:%M:%S")

                    img_filename = f"cctv_{self.category}_{session_id}_{cnt:05d}.jpg"
                    lbl_filename = f"cctv_{self.category}_{session_id}_{cnt:05d}.txt"

                    img_path = img_dir / img_filename
                    lbl_path = lbl_dir / lbl_filename

                    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    if self.category == "negatives":
                        with open(lbl_path, "w", encoding="utf-8") as f:
                            pass

                    print(f"[CCTV Stream Grabber #{cnt:04d}] Saved: {img_filename} at {self.last_capture_time} (Rate: {self.frames_per_minute}/min)")

                # Sleep slightly to yield CPU while reading stream
                time.sleep(0.05)

        except Exception as e:
            with self.lock:
                self.error_msg = f"Capture stream error: {str(e)}"
            print(f"[CCTV Manager Exception] {e}")
        finally:
            cap.release()
            with self.lock:
                self.is_capturing = False

cctv_collector = CCTVStreamCollector(base_dir=Path(__file__).resolve().parent.parent)
