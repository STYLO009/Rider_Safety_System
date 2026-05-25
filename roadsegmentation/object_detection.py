"""
object_detection.py — Real-time object detection using YOLOv8.

Detects all road-relevant objects:
  Vehicles   : car, truck, bus, motorcycle, bicycle
  People     : person
  Animals    : dog, cat, horse, cow, sheep, bird
  Road signs : stop sign, traffic light
  Others     : bench, backpack, umbrella

Also determines if the current scene LOOKS like a road
by checking what objects are present.
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ObjectDetection] ultralytics not installed. Run: pip install ultralytics")


# ── COCO class groups ─────────────────────────────────────────────────────────
VEHICLE_CLASSES    = {2: "car", 3: "motorcycle", 5: "bus",
                      7: "truck", 1: "bicycle"}
PERSON_CLASSES     = {0: "person"}
ANIMAL_CLASSES     = {16: "dog", 15: "cat", 17: "horse",
                      19: "cow", 18: "sheep", 14: "bird"}
SIGN_CLASSES       = {9: "traffic light", 11: "stop sign"}
ALL_TRACKED        = {**VEHICLE_CLASSES, **PERSON_CLASSES,
                      **ANIMAL_CLASSES, **SIGN_CLASSES}

# Colors per group (BGR)
GROUP_COLORS = {
    "vehicle":  (0,   200, 255),   # yellow
    "person":   (0,   80,  255),   # red-orange
    "animal":   (255, 150,  0),    # blue-ish
    "sign":     (0,   255, 180),   # green
    "unknown":  (180, 180, 180),   # gray
}

CONFIDENCE_THRESHOLD = 0.40


def _group(class_id: int) -> str:
    if class_id in VEHICLE_CLASSES: return "vehicle"
    if class_id in PERSON_CLASSES:  return "person"
    if class_id in ANIMAL_CLASSES:  return "animal"
    if class_id in SIGN_CLASSES:    return "sign"
    return "unknown"


class ObjectDetection:
    """
    Wraps YOLOv8n for real-time road object detection.

    Parameters
    ----------
    model_size : str
        YOLOv8 variant: 'n' (nano, fastest), 's', 'm', 'l', 'x' (most accurate).
        Use 'n' for real-time on CPU.
    confidence : float
        Minimum detection confidence (0–1).
    """

    def __init__(self, model_size: str = "n", confidence: float = CONFIDENCE_THRESHOLD):
        self.confidence  = confidence
        self._mock_mode  = False
        self._model      = None

        if not YOLO_AVAILABLE:
            print("[ObjectDetection] Running in MOCK mode (ultralytics not installed).")
            self._mock_mode = True
            return

        try:
            model_name = f"yolov8{model_size}.pt"
            print(f"[ObjectDetection] Loading {model_name} (downloads on first run ~6MB)...")
            self._model = YOLO(model_name)
            print("[ObjectDetection] Model ready.")
        except Exception as e:
            print(f"[ObjectDetection] Failed to load YOLO ({e}). Using MOCK mode.")
            self._mock_mode = True

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a BGR frame.

        Returns
        -------
        dict:
            detections  : list of dicts — each has keys:
                            label, class_id, group, confidence, box (x1,y1,x2,y2)
            is_road_scene : bool — True if scene looks like a road
            warnings    : list[str] — e.g. "Pedestrian ahead", "Vehicle close"
            counts      : dict — {group: count}
        """
        if self._mock_mode:
            return self._empty_result()

        try:
            results = self._model(frame, conf=self.confidence,
                                  classes=list(ALL_TRACKED.keys()),
                                  verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_TRACKED:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "label":      ALL_TRACKED[cls_id],
                    "class_id":   cls_id,
                    "group":      _group(cls_id),
                    "confidence": round(float(box.conf[0]), 2),
                    "box":        (x1, y1, x2, y2),
                })

            counts       = self._count_groups(detections)
            is_road      = self._is_road_scene(detections, frame)
            warnings     = self._generate_warnings(detections, frame)

            return {
                "detections":   detections,
                "is_road_scene": is_road,
                "warnings":     warnings,
                "counts":       counts,
            }

        except Exception as e:
            print(f"[ObjectDetection] Warning: {e}")
            return self._empty_result()

    def draw_detections(self, frame: np.ndarray, det_result: dict) -> np.ndarray:
        """Draw bounding boxes and labels onto the frame. Returns annotated frame."""
        for det in det_result.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            color = GROUP_COLORS.get(det["group"], GROUP_COLORS["unknown"])
            label = f"{det['label']} {det['confidence']:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

            # Label text
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Scene type indicator (top right corner)
        h, w = frame.shape[:2]
        scene_text  = "ROAD SCENE" if det_result.get("is_road_scene") else "NON-ROAD"
        scene_color = (0, 200, 100) if det_result.get("is_road_scene") else (0, 60, 200)
        cv2.putText(frame, scene_text,
                    (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scene_color, 2)

        # Object count summary
        counts = det_result.get("counts", {})
        y_off  = 55
        for group, count in counts.items():
            if count > 0:
                color = GROUP_COLORS.get(group, GROUP_COLORS["unknown"])
                cv2.putText(frame, f"{group.capitalize()}: {count}",
                            (w - 200, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y_off += 22

        return frame

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_road_scene(self, detections: list, frame: np.ndarray) -> bool:
        """
        Determine if the scene is a road by:
        1. Presence of vehicles, traffic lights, or stop signs
        2. Dominant color analysis (gray/black tones = asphalt)
        """
        # Check for road objects
        for det in detections:
            if det["group"] in ("vehicle", "sign"):
                return True

        # Fallback: check if bottom half is mostly gray (asphalt)
        h, w   = frame.shape[:2]
        bottom = frame[h // 2:, :]
        hsv    = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        # Low saturation + medium value = gray road surface
        gray_mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 40) & (hsv[:, :, 2] < 180)
        gray_ratio = np.sum(gray_mask) / gray_mask.size
        return gray_ratio > 0.30

    def _generate_warnings(self, detections: list, frame: np.ndarray) -> list:
        """Generate contextual warnings based on detected objects and their size."""
        warnings = []
        h, w = frame.shape[:2]
        frame_area = h * w

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            obj_area  = (x2 - x1) * (y2 - y1)
            size_ratio = obj_area / frame_area
            label      = det["label"]

            if det["group"] == "person":
                if size_ratio > 0.05:
                    warnings.append(f"⚠ Pedestrian close ahead!")
                else:
                    warnings.append(f"Pedestrian detected")

            elif det["group"] == "vehicle":
                if size_ratio > 0.15:
                    warnings.append(f"⚠ {label.capitalize()} very close!")
                elif size_ratio > 0.05:
                    warnings.append(f"{label.capitalize()} ahead")

            elif det["group"] == "animal":
                warnings.append(f"⚠ {label.capitalize()} on road!")

            elif label == "traffic light":
                warnings.append("Traffic light detected")

            elif label == "stop sign":
                warnings.append("⚠ Stop sign ahead!")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w_str in warnings:
            key = w_str.replace("⚠ ", "").split(" ")[0]
            if key not in seen:
                seen.add(key)
                unique.append(w_str)

        return unique[:4]   # Show max 4 warnings at once

    def _count_groups(self, detections: list) -> dict:
        counts = {"vehicle": 0, "person": 0, "animal": 0, "sign": 0}
        for det in detections:
            g = det["group"]
            if g in counts:
                counts[g] += 1
        return counts

    def _empty_result(self) -> dict:
        return {
            "detections":    [],
            "is_road_scene": False,
            "warnings":      [],
            "counts":        {"vehicle": 0, "person": 0, "animal": 0, "sign": 0},
        }
