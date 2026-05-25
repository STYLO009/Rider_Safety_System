"""
alert_system.py — Real-time safety alert generation and logging.
Produces human-readable alert strings and persists alert logs to disk.
"""

import os
import csv
import time
from datetime import datetime


# ── Alert severity levels ─────────────────────────────────────────────────────
SEVERITY_INFO     = "INFO"
SEVERITY_WARNING  = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"


class AlertSystem:
    """
    Evaluates speed and segmentation results every frame and emits alerts
    when safety thresholds are exceeded.

    Parameters
    ----------
    data_dir : str
        Directory where the alert log CSV is written.
    speed_warning_kmh : float
        Speed that triggers a WARNING alert.
    speed_critical_kmh : float
        Speed that triggers a CRITICAL alert (default: 1.3 × warning).
    cooldown_seconds : float
        Minimum seconds between repeated alerts of the same type.
    """

    ALERT_DEFS = {
        "speeding_warning":  (SEVERITY_WARNING,  "Speed exceeding limit"),
        "speeding_critical": (SEVERITY_CRITICAL, "Dangerous speed detected"),
        "lane_departure":    (SEVERITY_WARNING,  "Lane departure detected"),
        "low_road_coverage": (SEVERITY_INFO,     "Low road visibility"),
    }

    def __init__(self, data_dir: str = "data",
                 speed_warning_kmh: float = 60.0,
                 speed_critical_kmh: float = None,
                 cooldown_seconds: float = 3.0):
        self.data_dir           = data_dir
        self.speed_warning      = speed_warning_kmh
        self.speed_critical     = speed_critical_kmh or speed_warning_kmh * 1.3
        self.cooldown           = cooldown_seconds

        self._last_alert_time: dict = {}   # alert_key → last timestamp
        self._alert_log:       list = []

        os.makedirs(self.data_dir, exist_ok=True)
        self._log_path = os.path.join(
            self.data_dir,
            f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_csv()

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, speed_result: dict, seg_result: dict) -> str | None:
        """
        Check current frame results and return the highest-priority alert
        string, or None if no alert should fire this frame.

        Alerts respect the cooldown window to avoid spamming.
        """
        speed   = speed_result.get("speed_kmh", 0.0)
        depart  = seg_result.get("departure", False)
        road_r  = seg_result.get("road_ratio", 1.0)

        candidates = []

        # Speed checks (critical takes priority over warning)
        if speed >= self.speed_critical:
            candidates.append(
                ("speeding_critical",
                 f"CRITICAL: Speed {speed:.1f} km/h — far above limit"))
        elif speed >= self.speed_warning:
            candidates.append(
                ("speeding_warning",
                 f"WARNING: Speed {speed:.1f} km/h — above limit"))

        if depart:
            candidates.append(
                ("lane_departure", "WARNING: Lane departure detected"))

        if road_r < 0.10:
            candidates.append(
                ("low_road_coverage",
                 f"INFO: Low road visibility ({road_r:.0%} road detected)"))

        # Return first candidate that is not on cooldown
        for alert_key, message in candidates:
            if self._can_fire(alert_key):
                self._record(alert_key, message, speed, road_r)
                return message

        return None

    def get_alert_log(self) -> list:
        """Return the list of all alert records this session."""
        return list(self._alert_log)

    def recent_alerts(self, n: int = 5) -> list:
        """Return the N most recent alert strings."""
        return [a["message"] for a in self._alert_log[-n:]]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _can_fire(self, alert_key: str) -> bool:
        last = self._last_alert_time.get(alert_key, 0)
        return (time.time() - last) >= self.cooldown

    def _record(self, alert_key: str, message: str,
                speed: float, road_ratio: float):
        now = time.time()
        self._last_alert_time[alert_key] = now
        severity = self.ALERT_DEFS.get(alert_key, (SEVERITY_INFO, ""))[0]

        record = {
            "timestamp":  now,
            "datetime":   datetime.now().isoformat(),
            "alert_key":  alert_key,
            "severity":   severity,
            "message":    message,
            "speed_kmh":  round(speed, 2),
            "road_ratio": round(road_ratio, 3),
        }
        self._alert_log.append(record)
        self._append_csv(record)

    def _init_csv(self):
        with open(self._log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "datetime", "alert_key", "severity",
                "message", "speed_kmh", "road_ratio"])
            writer.writeheader()

    def _append_csv(self, record: dict):
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
