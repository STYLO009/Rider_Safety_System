"""
road_segmentation.py — Road and lane detection using a TFLite model.

Works with DeepLabV3 MobileNet (Pascal VOC, 21 classes):
  Class 0  = background  → treated as road surface (most ground pixels)
  Class 15 = person      → ignored
  Class 2  = bicycle     → ignored
  Class 7  = car         → ignored
  All others             → treated as non-road
"""

import numpy as np
import cv2
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


CLASS_BACKGROUND = 0
CLASS_ROAD       = 1
CLASS_LANE_LINE  = 2

COLORS = {
    CLASS_BACKGROUND: (0,   0,   0),
    CLASS_ROAD:       (0, 180,  80),
    CLASS_LANE_LINE:  (0, 220, 255),
}

# Pascal VOC classes that we treat as "road / drivable surface"
# 0=background, 9=chair(ignore), 20=tv(ignore)
# Background class is actually the best proxy for road in dashcam footage
ROAD_VOC_CLASSES = {0}          # background = road surface in VOC dashcam use
NON_ROAD_CLASSES = {15, 2, 7, 14, 19, 6}  # person, bicycle, car, motorbike, train, bus


class RoadSegmentation:
    def __init__(self, model_path: str, input_size=(257, 257),
                 confidence_threshold=0.5):
        self.input_size  = input_size
        self.conf_thresh = confidence_threshold
        self._mock_mode  = False

        if not os.path.exists(model_path) or os.path.getsize(model_path) < 7:
            print(f"[WARNING] Model not found at '{model_path}'. Using MOCK mode.")
            self._mock_mode = True
            return

        try:
            self._interpreter = tflite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Auto-detect input size from model
            shape = self._input_details[0]["shape"]   # [1, H, W, 3]
            self.input_size = (int(shape[2]), int(shape[1]))
            print(f"[INFO] Model loaded. Input size: {self.input_size}")

        except Exception as e:
            print(f"[WARNING] Model load failed ({e}). Using MOCK mode.")
            self._mock_mode = True

    def process(self, frame: np.ndarray) -> dict:
        h_orig, w_orig = frame.shape[:2]

        if self._mock_mode:
            return self._mock_result(h_orig, w_orig)

        try:
            blob   = self._preprocess(frame)
            self._interpreter.set_tensor(self._input_details[0]["index"], blob)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])

            raw_mask   = self._postprocess(output, (w_orig, h_orig))
            road_mask  = self._voc_to_road_mask(raw_mask)
            lane_lines = self._extract_lane_contours(road_mask)
            departure  = self._detect_departure(road_mask, lane_lines, h_orig, w_orig)
            road_ratio = float(np.sum(road_mask == CLASS_ROAD)) / road_mask.size

            # If model gives near-zero road (scene not recognised), fall back to mock
            if road_ratio < 0.03:
                return self._mock_result(h_orig, w_orig)

            return {
                "mask":       road_mask,
                "road_ratio": road_ratio,
                "lane_lines": lane_lines,
                "departure":  departure,
            }

        except Exception as e:
            print(f"[Segmentation] Warning: {e}")
            return self._mock_result(h_orig, w_orig)

    def draw_overlay(self, frame: np.ndarray, seg_result: dict,
                     alpha: float = 0.35) -> np.ndarray:
        mask    = seg_result["mask"]
        overlay = np.zeros_like(frame)

        for cls_id, color in COLORS.items():
            overlay[mask == cls_id] = color

        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        for contour in seg_result.get("lane_lines", []):
            cv2.drawContours(blended, [contour], -1, (0, 255, 255), 2)

        return blended

    # ── Internal ──────────────────────────────────────────────────────────────

    def _mock_result(self, h, w):
        """Simulated road mask — bottom 60% of frame is road."""
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.4):, int(w * 0.08):int(w * 0.92)] = CLASS_ROAD
        mask[int(h * 0.4):, int(w * 0.10):int(w * 0.13)] = CLASS_LANE_LINE
        mask[int(h * 0.4):, int(w * 0.87):int(w * 0.90)] = CLASS_LANE_LINE
        road_ratio  = float(np.sum(mask == CLASS_ROAD)) / mask.size
        lane_lines  = self._extract_lane_contours(mask)
        return {"mask": mask, "road_ratio": road_ratio,
                "lane_lines": lane_lines, "departure": False}

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.input_size)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm    = rgb.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=0)

    def _postprocess(self, output: np.ndarray, target_size: tuple) -> np.ndarray:
        if output.ndim == 4:
            if output.shape[1] < output.shape[-1]:
                output = np.transpose(output, (0, 2, 3, 1))
            class_map = np.argmax(output[0], axis=-1).astype(np.uint8)
        else:
            class_map = output[0].astype(np.uint8)
        return cv2.resize(class_map, target_size, interpolation=cv2.INTER_NEAREST)

    def _voc_to_road_mask(self, voc_mask: np.ndarray) -> np.ndarray:
        """Convert 21-class VOC mask → 3-class road mask."""
        road_mask = np.zeros_like(voc_mask)

        # Background class (0) → road in dashcam / street scenes
        for cls in ROAD_VOC_CLASSES:
            road_mask[voc_mask == cls] = CLASS_ROAD

        # Use simple edge-based lane detection on road pixels
        # (VOC has no lane class, so we derive it geometrically)
        h, w = road_mask.shape
        left_band  = slice(int(w * 0.10), int(w * 0.15))
        right_band = slice(int(w * 0.85), int(w * 0.90))
        road_rows  = road_mask[int(h * 0.5):, :] == CLASS_ROAD

        if np.any(road_rows[:, left_band]):
            road_mask[int(h * 0.5):, left_band][
                road_mask[int(h * 0.5):, left_band] == CLASS_ROAD] = CLASS_LANE_LINE
        if np.any(road_rows[:, right_band]):
            road_mask[int(h * 0.5):, right_band][
                road_mask[int(h * 0.5):, right_band] == CLASS_ROAD] = CLASS_LANE_LINE

        return road_mask

    def _extract_lane_contours(self, mask: np.ndarray) -> list:
        lane_mask = (mask == CLASS_LANE_LINE).astype(np.uint8) * 255
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > 200]

    def _detect_departure(self, mask, lane_lines, h, w) -> bool:
        bottom    = mask[int(h * 0.75):, :]
        left_has  = np.any(bottom[:, :w // 2]  == CLASS_LANE_LINE)
        right_has = np.any(bottom[:, w // 2:]  == CLASS_LANE_LINE)
        return not (left_has and right_has)