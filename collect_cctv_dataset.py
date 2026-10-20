import os
import sys
import cv2
import time
import argparse
import urllib.parse
from pathlib import Path
from datetime import datetime

# Configure safe UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Robust low-latency single-threaded FFMPEG RTSP parameters for OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|threads;1|fflags;nobuffer|flags;low_delay|max_delay;0"
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

def run_collector(
    source=None,
    channel="301",
    base_dir_name="cctv_dataset",
    category="negatives",
    target_fps=None,
    fpm=4.0,
    interval=None,
    auto_start=True,
    max_frames=0,
    headless=False
):
    base_dir = Path(__file__).resolve().parent
    root_dataset = base_dir / base_dir_name

    img_dir = root_dataset / "images" / category
    lbl_dir = root_dataset / "labels" / category
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Determine source URL
    if source is None:
        ch = channel if channel else "301"
        source = f"rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/{ch}"
    elif str(source).isdigit():
        source = f"rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/{source}"

    # Sanitize URL if RTSP
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        source = sanitize_rtsp_url(source)

    # Calculate capture interval (seconds per frame)
    if interval is not None and interval > 0:
        capture_interval = float(interval)
        rate_str = f"Every {capture_interval:.1f}s ({60.0 / capture_interval:.1f} frames/min)"
    elif fpm is not None and fpm > 0:
        capture_interval = 60.0 / float(fpm)
        rate_str = f"{fpm:.1f} frame(s)/minute (every {capture_interval:.1f}s)"
    elif target_fps is not None and target_fps > 0:
        capture_interval = 1.0 / float(target_fps)
        rate_str = f"{target_fps:.2f} frame(s)/second (every {capture_interval:.2f}s)"
    else:
        capture_interval = 15.0  # default: 4 frames per minute
        rate_str = "4.0 frames/minute (every 15.0s)"

    print("=" * 65)
    print("[CCTV] CCTV Dataset Collector & Frame Grabber")
    print(f"Connecting to: {source}")
    print(f"Category:      {category.upper()}")
    print(f"Image Folder:  {img_dir.resolve()}")
    print(f"Capture Rate:  {rate_str}")
    print(f"Mode:          {'HEADLESS (Background)' if headless else 'GUI (Interactive)'}")
    print("=" * 65)

    if not headless:
        print("\nCONTROLS on Preview Window:")
        print("  [SPACEBAR] : Start / Pause continuous capture")
        print("  [S] key    : Take a single snapshot")
        print("  [B] key    : Burst capture (10 snapshots)")
        print("  [C] key    : Toggle Category between GLOVES <-> NEGATIVES")
        print("  [Q] or ESC : Stop and finish\n")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open stream from: {source}")
        return

    is_recording = auto_start or headless
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    captured_count = 0
    burst_remaining = 0
    current_category = category
    last_capture_time = 0.0

    fps_display_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    window_name = f"CCTV Frame Collector - Ch {channel or '301'} (SPACE: Rec | Q: Quit)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Waiting for stream packet...")
                time.sleep(0.5)
                continue

            now = time.time()
            fps_counter += 1
            if now - fps_display_time >= 1.0:
                current_fps = fps_counter / (now - fps_display_time)
                fps_counter = 0
                fps_display_time = now

            # Save snapshot logic
            should_save = False

            actual_interval = 0.0
            if burst_remaining > 0:
                if now - last_capture_time >= capture_interval:
                    should_save = True
                    burst_remaining -= 1
                    actual_interval = now - last_capture_time if last_capture_time > 0 else 0.0
                    last_capture_time = now

            elif is_recording:
                if now - last_capture_time >= capture_interval:
                    should_save = True
                    actual_interval = now - last_capture_time if last_capture_time > 0 else 0.0
                    last_capture_time = now

            if should_save:
                captured_count += 1
                cat_img_dir = root_dataset / "images" / current_category
                cat_lbl_dir = root_dataset / "labels" / current_category
                cat_img_dir.mkdir(parents=True, exist_ok=True)
                cat_lbl_dir.mkdir(parents=True, exist_ok=True)

                img_filename = f"cctv_{current_category}_{session_id}_{captured_count:05d}.jpg"
                lbl_filename = f"cctv_{current_category}_{session_id}_{captured_count:05d}.txt"

                img_file_path = cat_img_dir / img_filename
                lbl_file_path = cat_lbl_dir / lbl_filename

                cv2.imwrite(str(img_file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # If negatives, auto-create empty label file
                if current_category == "negatives":
                    with open(lbl_file_path, "w", encoding="utf-8") as lf:
                        pass

                int_str = f"{actual_interval:.1f}s interval" if actual_interval > 0 else "initial snapshot"
                print(f"[CAPTURE #{captured_count:04d}] Saved: {img_filename} ({int_str} | next in {capture_interval:.1f}s)")

                if max_frames > 0 and captured_count >= max_frames:
                    print(f"Target frame limit reached ({max_frames} frames).")
                    break

            if headless:
                # In headless background mode, briefly sleep to reduce CPU usage
                time.sleep(0.02)
                continue

            # Draw HUD in GUI mode
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.70, display_frame, 0.30, 0)

            cat_badge = f"CAT: [{current_category.upper()}] (Press 'C' to switch)"
            cat_color = (0, 200, 0) if current_category == "gloves" else (0, 165, 255)
            cv2.putText(display_frame, cat_badge, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cat_color, 2, cv2.LINE_AA)

            if is_recording or burst_remaining > 0:
                rec_status = f"REC: {rate_str}" if is_recording else f"BURST ({burst_remaining})"
                rec_color = (0, 0, 255)
            else:
                rec_status = "PAUSED"
                rec_color = (0, 220, 220)

            time_since_last = now - last_capture_time if last_capture_time > 0 else 0
            time_to_next = max(0.0, capture_interval - time_since_last) if is_recording else 0.0

            cv2.putText(display_frame, rec_status, (w - 420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, rec_color, 2, cv2.LINE_AA)
            if is_recording and burst_remaining == 0:
                cv2.putText(display_frame, f"Next grab in: {time_to_next:.1f}s", (w - 420, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Saved: {captured_count} frames", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "SPACE: Rec/Pause | C: Switch Cat | S: Snapshot | B: Burst | Q: Quit", (w - 600, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            if is_recording and int(now * 2) % 2 == 0:
                cv2.circle(display_frame, (w - 20, 25), 9, (0, 0, 255), -1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # SPACEBAR
                is_recording = not is_recording
                state_str = "STARTED" if is_recording else "PAUSED"
                print(f"[INFO] Recording {state_str} (Captured: {captured_count} in '{current_category}')")
            elif key == ord('c') or key == ord('C'):
                current_category = "negatives" if current_category == "gloves" else "gloves"
                print(f"[CATEGORY SWITCH] Now saving as: {current_category.upper()}")
            elif key == ord('s') or key == ord('S'):
                captured_count += 1
                cat_img_dir = root_dataset / "images" / current_category
                cat_lbl_dir = root_dataset / "labels" / current_category
                cat_img_dir.mkdir(parents=True, exist_ok=True)
                cat_lbl_dir.mkdir(parents=True, exist_ok=True)

                img_filename = f"cctv_{current_category}_{session_id}_{captured_count:05d}.jpg"
                lbl_filename = f"cctv_{current_category}_{session_id}_{captured_count:05d}.txt"

                cv2.imwrite(str(cat_img_dir / img_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if current_category == "negatives":
                    with open(cat_lbl_dir / lbl_filename, "w", encoding="utf-8") as lf:
                        pass
                print(f"[SNAPSHOT] Saved: {img_filename}")
            elif key == ord('b') or key == ord('B'):
                burst_remaining = 10
                print(f"[BURST] Capturing 10 frames in '{current_category}'...")

    finally:
        cap.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("\n" + "=" * 65)
        print("CAPTURE SESSION FINISHED!")
        print(f"Total Frames Saved: {captured_count}")
        print(f"Dataset Folder:     {root_dataset.resolve()}")
        print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CCTV Dataset Frames with Auto-Categorization")
    parser.add_argument("--channel", type=str, default="301", help="Channel number (default: 301)")
    parser.add_argument("--source", type=str, default=None, help="RTSP URL, video file, or camera index")
    parser.add_argument("--output", type=str, default="cctv_dataset", help="Root folder for dataset")
    parser.add_argument("--category", type=str, choices=["gloves", "negatives"], default="negatives", help="gloves or negatives (default: negatives)")
    parser.add_argument("--fps", type=float, default=None, help="Snapshots per second (e.g. 1.0)")
    parser.add_argument("--fpm", type=float, default=4.0, help="Frames per minute (default: 4.0 fpm / 1 frame every 15s)")
    parser.add_argument("--interval", type=float, default=None, help="Seconds between frames (e.g. 15.0)")
    parser.add_argument("--auto", action="store_true", default=True, help="Start recording immediately upon launch")
    parser.add_argument("--pause", action="store_true", help="Start in paused mode (press SPACE to begin)")
    parser.add_argument("--max", type=int, default=0, help="Stop automatically after saving N frames (0 = unlimited)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window in background")

    args = parser.parse_args()
    run_collector(
        source=args.source,
        channel=args.channel,
        base_dir_name=args.output,
        category=args.category,
        target_fps=args.fps,
        fpm=args.fpm,
        interval=args.interval,
        auto_start=(not args.pause),
        max_frames=args.max,
        headless=args.headless
    )
