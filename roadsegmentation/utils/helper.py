"""
utils/helper.py — Shared utility functions used across the project.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path


# ── Logging setup ─────────────────────────────────────────────────────────────

_logger = None

def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger("road_safety")
    _logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    _logger.addHandler(ch)

    # File handler
    os.makedirs("data", exist_ok=True)
    fh = logging.FileHandler(
        f"data/system_{datetime.now().strftime('%Y%m%d')}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    _logger.addHandler(fh)

    return _logger


# ── Public helpers ────────────────────────────────────────────────────────────

def log_event(message: str, level: str = "info"):
    """
    Log a message at the given level ('debug', 'info', 'warning', 'error').
    Mirrors to both console and the daily log file in data/.
    """
    logger = _get_logger()
    getattr(logger, level.lower(), logger.info)(message)


def ensure_dirs(*paths: str):
    """Create one or more directories (and parents) if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_str() -> str:
    """Return the current datetime as a compact string, e.g. '20240523_142035'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def moving_average(values: list, window: int) -> float:
    """Return the average of the last `window` elements in `values`."""
    if not values:
        return 0.0
    subset = values[-window:]
    return sum(subset) / len(subset)


class FPSCounter:
    """
    Lightweight FPS counter based on a sliding time window.

    Usage::

        fps_counter = FPSCounter(window=30)
        while True:
            fps_counter.tick()
            print(f"FPS: {fps_counter.fps:.1f}")
    """

    def __init__(self, window: int = 30):
        self._window     = window
        self._timestamps = []

    def tick(self):
        """Record a frame tick."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def frame_to_jpeg_bytes(frame, quality: int = 85) -> bytes:
    """
    Encode a BGR NumPy frame to JPEG bytes.
    Useful for streaming over HTTP or saving thumbnails.
    """
    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


def list_session_logs(data_dir: str = "data") -> list:
    """Return a sorted list of session JSON log paths in data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted(data_path.glob("session_*.json"))


def load_session_log(path: str) -> dict:
    """Load and return a session JSON log as a Python dict."""
    import json
    with open(path) as f:
        return json.load(f)
