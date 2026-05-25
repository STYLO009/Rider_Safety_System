"""
speed_monitor.py — Vehicle speed estimation from frame-to-frame optical flow.
Uses Lucas-Kanade sparse optical flow on road-region feature points.
"""

import cv2
import numpy as np
from collections import deque


# ── Constants ─────────────────────────────────────────────────────────────────
SPEED_LIMIT_KMH   = 60.0
PIXELS_PER_METER  = 8.0
SMOOTHING_WINDOW  = 10


class SpeedMonitor:
    def __init__(self, fps: float = 30.0,
                 speed_limit_kmh: float = SPEED_LIMIT_KMH,
                 pixels_per_meter: float = PIXELS_PER_METER):
        self.fps              = fps
        self.speed_limit      = speed_limit_kmh
        self.pixels_per_meter = pixels_per_meter

        self._prev_gray  = None
        self._prev_pts   = None
        self._speed_buf  = deque(maxlen=SMOOTHING_WINDOW)
        self._frame_idx  = 0

        self._lk_params = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self._feature_params = dict(
            maxCorners   = 100,
            qualityLevel = 0.3,
            minDistance  = 7,
            blockSize    = 7,
        )

    def update(self, frame: np.ndarray, seg_result: dict) -> dict:
        speed_kmh    = 0.0
        raw_speed    = 0.0
        flow_vectors = []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            road_mask = self._make_road_mask(seg_result, frame.shape[:2])

            if self._prev_gray is not None and self._prev_pts is not None \
                    and len(self._prev_pts) > 0:

                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._prev_pts, None, **self._lk_params)

                if new_pts is not None and status is not None:
                    # Flatten from (N,1,2) → (N,2) safely
                    status_flat = status.ravel()
                    good_new  = new_pts[status_flat == 1].reshape(-1, 2)
                    good_prev = self._prev_pts[status_flat == 1].reshape(-1, 2)

                    if len(good_new) > 2:
                        displacements = np.linalg.norm(
                            good_new - good_prev, axis=1)

                        # Remove outliers (top 10%) to avoid jumpy readings
                        threshold = np.percentile(displacements, 90)
                        filtered = displacements[displacements < threshold]

                        if len(filtered) > 0:
                            median_px = float(np.median(filtered))
                            meters_per_frame = median_px / self.pixels_per_meter
                            raw_speed = float(meters_per_frame * self.fps * 3.6)

                            self._speed_buf.append(raw_speed)
                            speed_kmh = float(np.mean(list(self._speed_buf)))

                        flow_vectors = [
                            (int(p[0]), int(p[1]), int(n[0]), int(n[1]))
                            for p, n in zip(good_prev.tolist(), good_new.tolist())
                        ]

            # Re-detect feature points every 5 frames or when too few remain
            if self._frame_idx % 5 == 0 or self._prev_pts is None \
                    or len(self._prev_pts) < 10:
                pts = cv2.goodFeaturesToTrack(
                    gray, mask=road_mask, **self._feature_params)
                # Always store as (N,1,2) — what LK expects
                self._prev_pts = pts

            self._prev_gray = gray
            self._frame_idx += 1

        except Exception as e:
            # Never crash the main loop due to a speed calculation error
            print(f"[SpeedMonitor] Warning: {e}")

        return {
            "speed_kmh":     round(speed_kmh, 2),
            "raw_speed_kmh": round(raw_speed, 2),
            "speeding":      speed_kmh > self.speed_limit,
            "speed_limit":   self.speed_limit,
            "flow_vectors":  flow_vectors,
        }

    def reset(self):
        self._prev_gray = None
        self._prev_pts  = None
        self._speed_buf.clear()
        self._frame_idx = 0

    def _make_road_mask(self, seg_result: dict, shape: tuple) -> np.ndarray:
        """Use bottom 60% of frame as road region when model gives 0% road."""
        if seg_result and "mask" in seg_result:
            road_mask = (seg_result["mask"] == 1).astype(np.uint8) * 255
            # If model detects very little road, fall back to bottom portion
            if np.sum(road_mask) < (shape[0] * shape[1] * 0.05 * 255):
                road_mask = np.zeros(shape, dtype=np.uint8)
                road_mask[int(shape[0] * 0.4):, :] = 255
        else:
            road_mask = np.zeros(shape, dtype=np.uint8)
            road_mask[int(shape[0] * 0.4):, :] = 255
        return road_mask