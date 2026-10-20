import os
import time
import urllib.parse
import threading
from datetime import datetime
from typing import Optional, Tuple, Any
from pathlib import Path

# Zero-latency FFMPEG RTSP options MUST be set before import cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0"
)

import cv2
import numpy as np

from ..config import config

def sanitize_rtsp_url(url: str) -> str:
    """Safe URL encoder for special characters in RTSP passwords."""
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

class RTSPCaptureEngine:
    """
    High-performance frame producer supporting:
    - Zero-latency RTSP camera streams
    - Video files (.mp4, .avi, .mov, .mkv) with natural FPS pacing & looping
    - Local USB / webcams (0, 1)
    - Dynamic live source switching without server restarts
    """
    def __init__(self, rtsp_url: str = config.DEFAULT_RTSP_URL, camera_id: str = "301", loop_video: bool = True):
        self.raw_source = str(rtsp_url)
        self.camera_id = camera_id
        self.loop_video = loop_video
        self.is_paused = False
        
        self.source, self.source_type = self._parse_source(self.raw_source)
        
        self.is_running = False
        self.is_connected = False
        self.error_msg: Optional[str] = None
        self.frame_counter = 0
        
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_frame_time: float = 0.0
        self._last_consumed_frame: int = -1  # For stale-frame skipping

        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.restart_source_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def _parse_source(self, src: str) -> Tuple[Any, str]:
        src_str = str(src).strip()
        if src_str.lower().startswith("rtsp://"):
            return sanitize_rtsp_url(src_str), "rtsp"
        elif src_str.isdigit():
            return int(src_str), "webcam"
        else:
            return str(src_str), "video"

    def switch_source(self, new_source: str, camera_id: Optional[str] = None, loop_video: bool = True):
        """Dynamically switches stream source to a video file, webcam, or RTSP URL."""
        with self.frame_lock:
            self.raw_source = str(new_source)
            self.source, self.source_type = self._parse_source(self.raw_source)
            if camera_id:
                self.camera_id = str(camera_id)
            elif self.source_type == "video":
                self.camera_id = Path(self.raw_source).name
            elif self.source_type == "webcam":
                self.camera_id = f"Webcam-{self.source}"
            else:
                self.camera_id = "301"
            self.loop_video = loop_video
            self.is_paused = False
            self.frame_counter = 0
            self.restart_source_event.set()
        print(f"[RTSPCaptureEngine] Switching source to [{self.source_type}]: {self.camera_id}")

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def restart_playback(self):
        self.restart_source_event.set()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True, name="RTSPCaptureWorker")
        self.worker_thread.start()
        print(f"[RTSPCaptureEngine] Started capture engine on [{self.source_type}]: {self.camera_id}")

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()
        self.restart_source_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print(f"[RTSPCaptureEngine] Stopped capture on {self.camera_id}")

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Returns (frame_copy, timestamp_sec, frame_number). Always returns the freshest frame."""
        with self.frame_lock:
            if self.latest_raw_frame is None:
                return None, 0.0, self.frame_counter
            self._last_consumed_frame = self.frame_counter
            return self.latest_raw_frame.copy(), self.latest_frame_time, self.frame_counter

    def has_new_frame(self) -> bool:
        """True if a frame newer than the last consumed one is available."""
        return self.frame_counter > self._last_consumed_frame

    def wait_for_frame(self, timeout: float = 0.05) -> bool:
        """Blocks until a new frame arrives or timeout (50ms default for responsive pipeline)."""
        flag = self.new_frame_event.wait(timeout=timeout)
        self.new_frame_event.clear()
        return flag

    def _capture_loop(self):
        while self.is_running:
            self.restart_source_event.clear()
            curr_src = self.source
            curr_type = self.source_type
            
            # Open VideoCapture based on source type
            if curr_type == "rtsp":
                cap = cv2.VideoCapture(curr_src, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif curr_type == "webcam":
                cap = cv2.VideoCapture(int(curr_src))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else: # Video file
                cap = cv2.VideoCapture(str(curr_src))

            if not cap.isOpened():
                self.is_connected = False
                self.error_msg = f"Failed to open {curr_type} source: {self.camera_id}"
                time.sleep(config.RTSP_RECONNECT_INTERVAL_SEC)
                continue

            self.is_connected = True
            self.error_msg = None
            consecutive_failures = 0

            # Determine frame interval for smooth video playback
            fps = cap.get(cv2.CAP_PROP_FPS) if curr_type == "video" else 0
            if fps <= 0 or fps > 120:
                fps = 25.0
            frame_interval = 1.0 / fps

            while self.is_running and cap.isOpened():
                if self.restart_source_event.is_set():
                    break

                if self.is_paused:
                    time.sleep(0.05)
                    continue

                t_frame_start = time.time()

                if curr_type == "rtsp":
                    if not cap.grab():
                        consecutive_failures += 1
                        if consecutive_failures > config.RTSP_MAX_CONSECUTIVE_FAILURES:
                            self.error_msg = "Too many frame grab failures, reconnecting stream"
                            break
                        time.sleep(0.005)
                        continue
                    consecutive_failures = 0
                    ret, frame = cap.retrieve()
                else:
                    ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    if curr_type == "video":
                        if self.loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            time.sleep(0.02)
                            continue
                        else:
                            time.sleep(0.1)
                            continue
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            break
                        time.sleep(0.01)
                        continue

                # Validate frame dimensions
                if frame.shape[0] < 80 or frame.shape[1] < 80:
                    continue

                self.frame_counter += 1
                with self.frame_lock:
                    self.latest_raw_frame = frame
                    self.latest_frame_time = time.time()
                self.new_frame_event.set()

                # For video files: pace playback at the video's native FPS.
                # Do NOT wait for the consumer — if pipeline is slower than FPS,
                # consumer will pick the freshest available frame (skip detection
                # on missed frames) so video never lags behind real-time.
                if curr_type == "video":
                    elapsed = time.time() - t_frame_start
                    delay = max(0.001, frame_interval - elapsed)
                    time.sleep(delay)

            cap.release()
            self.is_connected = False
            if not self.restart_source_event.is_set():
                time.sleep(config.RTSP_RECONNECT_INTERVAL_SEC)
