"""
app.py — Road Safety Monitoring System with Object Detection.

Phone camera setup (Android IP Webcam):
  1. Install 'IP Webcam' from Play Store
  2. Open app → tap 'Start server'
  3. Set PHONE_IP below to your phone's IP
  4. Phone and PC must be on the same WiFi
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
from camera import Camera
from road_segmentation import RoadSegmentation
from speed_monitor import SpeedMonitor
from review_analysis import ReviewAnalysis
from alert_system import AlertSystem
from safety_score import SafetyScore
from object_detection import ObjectDetection
from utils.helper import log_event, ensure_dirs

# ── Configuration ─────────────────────────────────────────────────────────────
PHONE_IP       = "192.168.0.108"         # <-- Your phone's IP here
PHONE_PORT     = 8080
VIDEO_SOURCE   = f"http://{PHONE_IP}:{PHONE_PORT}/video"

MODEL_PATH     = "models/road_model.tflite"
DATA_DIR       = "data"
DISPLAY        = True
TARGET_FPS     = 30
WARMUP_SECONDS = 5.0


def check_connection():
    print(f"\n[INFO] Connecting to phone at: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            print("[INFO] Phone camera connected!\n")
            return True
    cap.release()
    print(f"""
[ERROR] Cannot connect to phone camera.
  1. Is IP Webcam running? (tap 'Start server')
  2. Is PHONE_IP correct? Currently: "{PHONE_IP}"
  3. Are phone and PC on the same WiFi?
  4. Test in browser: {VIDEO_SOURCE}
""")
    return False


def draw_hud(frame, speed_result, seg_result, det_result, score, scorer, phone_ip):
    """Draw all overlays onto the frame."""
    h, w = frame.shape[:2]

    # ── Road/non-road banner ──────────────────────────────────────────────────
    is_road = det_result.get("is_road_scene", False)
    if not is_road:
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
        cv2.putText(frame, "NOT A ROAD SCENE — Scores paused",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Left panel: speed + score ─────────────────────────────────────────────
    spd = speed_result.get("speed_kmh", 0)
    lim = speed_result.get("speed_limit", 60)

    spd_color = (0, 80, 255) if spd > lim else (0, 255, 100)
    cv2.putText(frame, f"Speed: {spd:.1f} km/h",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, spd_color, 2)
    cv2.putText(frame, f"Limit: {lim:.0f} km/h",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
    cv2.putText(frame, f"Safety: {score:.1f}/100  [{scorer.grade}]",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

    # Road coverage
    road_pct = seg_result.get("road_ratio", 0) * 100
    cv2.putText(frame, f"Road coverage: {road_pct:.0f}%",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 1)

    # Source IP (top right)
    cv2.putText(frame, f"{phone_ip}:{PHONE_PORT}",
                (w - 210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    # ── Warnings (right side) ─────────────────────────────────────────────────
    warnings = det_result.get("warnings", [])
    for i, warn in enumerate(warnings):
        color = (0, 60, 255) if "⚠" in warn else (0, 200, 255)
        cv2.putText(frame, warn,
                    (w - 380, 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Lane departure (bottom center) ───────────────────────────────────────
    if seg_result.get("departure") and is_road:
        cv2.putText(frame, "!! LANE DEPARTURE !!",
                    (w // 2 - 160, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    # ── Speeding alert (bottom) ───────────────────────────────────────────────
    if speed_result.get("speeding") and is_road:
        cv2.putText(frame, f"!! SPEEDING — {spd:.1f} km/h !!",
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # ── Object counts (bottom right) ─────────────────────────────────────────
    counts = det_result.get("counts", {})
    count_str = "  ".join(
        f"{g[0].upper()}:{n}" for g, n in counts.items() if n > 0)
    if count_str:
        cv2.putText(frame, count_str,
                    (w - 300, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Quit hint
    cv2.putText(frame, "Q / ESC to quit",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)

    return frame


def main():
    ensure_dirs(DATA_DIR)
    log_event("System starting up.")

    if not check_connection():
        return

    # ── Init all modules ──────────────────────────────────────────────────────
    camera    = Camera(source=VIDEO_SOURCE, fps=TARGET_FPS)
    segmentor = RoadSegmentation(model_path=MODEL_PATH)
    detector  = ObjectDetection(model_size="n")      # YOLOv8 nano = fastest
    speed_mon = SpeedMonitor(fps=TARGET_FPS)
    reviewer  = ReviewAnalysis(data_dir=DATA_DIR)
    alerter   = AlertSystem(data_dir=DATA_DIR)
    scorer    = SafetyScore()

    session_events = []

    try:
        camera.start()
        log_event("Camera started. Warming up...")

        # Wait for first frame
        warmup_start = time.time()
        while True:
            if camera.read_frame() is not None:
                break
            if time.time() - warmup_start > WARMUP_SECONDS:
                print("[ERROR] No frames received. Check IP Webcam is running.")
                return
            time.sleep(0.05)

        log_event("Stream ready. Press Q to quit.")

        if DISPLAY:
            cv2.namedWindow("Road Safety Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Road Safety Monitor", 1280, 720)

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Perception ───────────────────────────────────────────────────
            seg_result = segmentor.process(frame)
            det_result = detector.detect(frame)

            # Only run speed / alerts when we're actually on a road
            is_road = det_result.get("is_road_scene", True)

            speed_result = speed_mon.update(frame, seg_result)

            # ── Analysis & Alerting ──────────────────────────────────────────
            if is_road:
                review_event = reviewer.analyse(seg_result, speed_result)
                alert        = alerter.check(speed_result, seg_result)
                score        = scorer.update(speed_result, seg_result, alert)

                if review_event:
                    session_events.append(review_event)
                    log_event(f"Event: {review_event['type']} @ {review_event['speed_kmh']} km/h")
                if alert:
                    log_event(f"ALERT: {alert}")
            else:
                alert = None
                score = scorer.current_score    # Freeze score when not on road

            # ── Display ──────────────────────────────────────────────────────
            if DISPLAY:
                annotated = segmentor.draw_overlay(frame.copy(), seg_result)
                annotated = detector.draw_detections(annotated, det_result)
                annotated = draw_hud(annotated, speed_result, seg_result,
                                     det_result, score, scorer, PHONE_IP)
                cv2.imshow("Road Safety Monitor", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                log_event("User quit.")
                break
            if DISPLAY and cv2.getWindowProperty(
                    "Road Safety Monitor", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        log_event("Interrupted by user.")
    except Exception as e:
        log_event(f"Error: {e}", level="error")
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()
        final_score = scorer.final_score()
        reviewer.save_session(session_events)
        log_event(f"Session ended. Score: {final_score:.1f}")
        print(f"\n{'='*45}")
        print(f"  Final Safety Score : {final_score:.1f} / 100")
        print(f"  Grade              : {scorer.grade}")
        report = scorer.report()
        print(f"  Offences           : {report['offence_tally']}")
        print(f"  {report['interpretation']}")
        print(f"{'='*45}\n")


if __name__ == "__main__":
    main()