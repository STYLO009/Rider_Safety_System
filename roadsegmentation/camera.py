"""
camera.py — Camera interface for capturing video frames.
Supports webcam, IP cameras, and local video files.
"""

import cv2
import threading
import time


class Camera:
    """
    Thread-safe camera wrapper around OpenCV VideoCapture.

    Parameters
    ----------
    source : int or str
        0 for default webcam, or a path/URL to a video file / RTSP stream.
    fps : int
        Target capture frame rate. Ignored for file sources (uses native FPS).
    width : int
        Requested frame width (may be ignored by some cameras).
    height : int
        Requested frame height (may be ignored by some cameras).
    """

    def __init__(self, source=0, fps=30, width=1280, height=720):
        self.source  = source
        self.fps     = fps
        self.width   = width
        self.height  = height

        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Open the capture device and start the background reader thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def read_frame(self):
        """
        Return the most recent frame, or None if no frame is available yet
        or the stream has ended.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        """Stop the reader thread and release the capture device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    @property
    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    def get_native_fps(self):
        """Return the native FPS reported by the capture device."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return self.fps

    def get_resolution(self):
        """Return (width, height) as reported by the capture device."""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reader(self):
        """Background thread: continuously grab the latest frame."""
        delay = 1.0 / self.fps
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._running = False
                break
            with self._lock:
                self._frame = frame
            time.sleep(delay * 0.1)   # Minimal sleep — don't busy-spin at 100%
