import os
import cv2
import time
import argparse
import threading
import urllib.parse
from pathlib import Path
import torch

# Robust low-latency FFMPEG RTSP parameters for OpenCV (zero buffer, no delay)
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
            safe_url = f"rtsp://{user}:{encoded_pwd}@{host_path}"
            return safe_url
    return url





def run_cctv_detector(
    source="0",
    model_path="CPPE-Dataset/runs/detect/gloves_detector/weights/best.pt",
    conf_threshold=0.20,
    imgsz=1024,
    save_output=False,
    output_path="cctv_gloves_output.mp4",
    no_show=False
):
    from ultralytics import YOLO

    base_dir = Path(__file__).resolve().parent
    model_file = Path(model_path)
    
    # Path resolution - prioritize latest active_model, then cctv_50ep weights
    active_weights = base_dir / "runs" / "detect" / "active_model" / "weights" / "best.pt"
    cctv_50ep_weights = base_dir / "CPPE-Dataset" / "runs" / "detect" / "gloves_detector_cctv_50ep" / "weights" / "best.pt"
    aug_weights = Path("D:/Gloves/CPPE-Dataset/runs/detect/gloves_detector_aug/weights/best.pt")
    base_weights = base_dir / "CPPE-Dataset" / "runs" / "detect" / "gloves_detector" / "weights" / "best.pt"

    if not model_file.exists():
        candidates = [
            active_weights,
            cctv_50ep_weights,
            aug_weights,
            base_weights,
            base_dir / "runs" / "detect" / "gloves_detector" / "weights" / "best.pt",
            base_dir / "yolov8s.pt",
            base_dir / "CPPE-Dataset" / "yolov8s.pt",
            Path("D:/Gloves/CPPE-Dataset/yolov8s.pt")
        ]
        for candidate in candidates:
            if candidate.exists():
                model_file = candidate
                break

    print("=" * 60)
    print("Launching ZERO-LATENCY CCTV Gloves Detection Engine")
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model: {model_file}")

    # Check RTSP
    is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
    if is_rtsp:
        clean_url = sanitize_rtsp_url(source)
        source = clean_url
        print(f"Connecting to RTSP Stream (Zero-Latency Threaded Mode)...")
    else:
        print(f"Source: {source}")

    print(f"Confidence Threshold: {conf_threshold:.2f} (Use '+' or '-' in window to adjust)")
    print("Press 'q' or 'ESC' on the video window to stop.")
    print("=" * 60)

    # Load YOLO Model
    model = YOLO(str(model_file))

    # Single Image Mode
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if isinstance(source, str) and source.lower().endswith(image_exts):
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: Could not read image: {source}")
            return
        t0 = time.time()
        results = model.predict(source=frame, device=device, half=False, conf=conf_threshold, imgsz=imgsz, verbose=False)
        dt = time.time() - t0
        try:
            annotated_frame = results[0].plot()
        except Exception:
            annotated_frame = frame.copy()
        gloves_count = len(results[0].boxes)
        fps = 1.0 / max(dt, 1e-5)

        # Overlay HUD
        cv2.rectangle(annotated_frame, (10, 10), (380, 110), (20, 20, 20), -1)
        cv2.rectangle(annotated_frame, (10, 10), (380, 110), (100, 100, 100), 1)
        cv2.putText(annotated_frame, f"GPU FPS: {fps:.1f} | Conf: {conf_threshold:.2f}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        status_text = f"GLOVES DETECTED: {gloves_count}" if gloves_count > 0 else "NO GLOVES DETECTED"
        badge_color = (0, 200, 0) if gloves_count > 0 else (0, 0, 230)
        cv2.putText(annotated_frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2, cv2.LINE_AA)

        out_img_path = output_path if output_path.lower().endswith(image_exts) else "cctv_gloves_result.jpg"
        cv2.imwrite(out_img_path, annotated_frame)
        print(f"Single Image Processed: {gloves_count} gloves detected in {dt*1000:.1f}ms ({fps:.1f} FPS)!")
        if not no_show:
            try:
                cv2.imshow("CCTV Gloves Monitoring", annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        return

    # Video / Stream Capture
    if source.isdigit():
        video_src = int(source)
    else:
        video_src = source

    if is_rtsp:
        cap = cv2.VideoCapture(video_src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    print("Connected to stream. Real-time inference running...")

    writer = None
    if save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"Recording output to: {output_path}")

    prev_time = time.time()
    fps_history = []

    try:
        while cap.isOpened():
            # Flush queued backlog on RTSP to maintain real-time LIVE zero latency!
            if is_rtsp:
                cap.grab()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Run GPU inference (half=False prevents Turing TU116 NaN bbox underflow)
            results = model.predict(
                source=frame,
                device=device,
                half=False,
                conf=conf_threshold,
                imgsz=imgsz,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes
            gloves_boxes = [b for b in boxes if int(b.cls.item()) == 0] if len(boxes) > 0 else []
            hand_boxes = [b for b in boxes if int(b.cls.item()) == 1] if len(boxes) > 0 else []
            gloves_count = len(gloves_boxes)
            hand_count = len(hand_boxes)

            # Draw bounding boxes (safe from NaN/inf)
            try:
                annotated_frame = result.plot()
            except Exception:
                annotated_frame = frame.copy()

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # HUD Display
            cv2.rectangle(annotated_frame, (10, 10), (420, 115), (20, 20, 20), -1)
            cv2.rectangle(annotated_frame, (10, 10), (420, 115), (100, 100, 100), 1)

            cv2.putText(
                annotated_frame,
                f"GPU FPS: {avg_fps:.1f} | Conf: {conf_threshold:.2f} (+/-)",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if gloves_count > 0:
                max_conf = max([b.conf.item() for b in gloves_boxes])
                status_text = f"GLOVES: {gloves_count} ({max_conf*100:.0f}%)"
                badge_color = (0, 220, 0)
            elif hand_count > 0:
                max_conf = max([b.conf.item() for b in hand_boxes])
                status_text = f"NO GLOVES (HANDS: {hand_count})"
                badge_color = (0, 0, 230)
            else:
                status_text = "NO GLOVES DETECTED"
                badge_color = (0, 0, 230)

            cv2.putText(
                annotated_frame,
                status_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                badge_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated_frame,
                "Press +/- to tune threshold | 'q' to quit",
                (20, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            if writer:
                writer.write(annotated_frame)

            # Interactive display
            if not no_show:
                try:
                    cv2.imshow("CCTV Gloves Monitoring (Zero-Latency)", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('+') or key == ord('='):
                        conf_threshold = min(0.95, conf_threshold + 0.05)
                        print(f"Confidence threshold set to: {conf_threshold:.2f}")
                    elif key == ord('-') or key == ord('_'):
                        conf_threshold = max(0.05, conf_threshold - 0.05)
                        print(f"Confidence threshold set to: {conf_threshold:.2f}")
                except cv2.error:
                    pass

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("CCTV monitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Latency CCTV Gloves Detection via GPU")
    parser.add_argument("--source", type=str, default="rtsp://admin:Syn@vos@456@192.168.1.128:554/Streaming/Channels/301", help="Video source: RTSP URL, video path (.mp4), webcam index ('0'), or image file")
    parser.add_argument("--weights", type=str, default="CPPE-Dataset/runs/detect/gloves_detector/weights/best.pt", help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution scale (1024 for real-time 1080p CCTV)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="cctv_gloves_output.mp4", help="Output video path")
    parser.add_argument("--no-show", action="store_true", help="Do not display GUI window")

    args = parser.parse_args()
    run_cctv_detector(
        source=args.source,
        model_path=args.weights,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        save_output=args.save,
        output_path=args.output,
        no_show=args.no_show
    )
