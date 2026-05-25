"""
review_analysis.py — Post-hoc and real-time analysis of driving events.
Aggregates segmentation and speed data into structured event records,
and persists session logs to the data/ directory.
"""

import json
import os
import time
from datetime import datetime
from collections import deque


# ── Event types ───────────────────────────────────────────────────────────────
EVENT_SPEEDING         = "speeding"
EVENT_LANE_DEPARTURE   = "lane_departure"
EVENT_LOW_ROAD_COVERAGE = "low_road_coverage"


class ReviewAnalysis:
    """
    Analyses per-frame results from the segmentation and speed modules,
    debounces noisy signals into discrete events, and saves session summaries.

    Parameters
    ----------
    data_dir : str
        Directory where session logs are saved (JSON files).
    debounce_frames : int
        A condition must persist for this many consecutive frames before
        it is recorded as an event (reduces false positives).
    road_coverage_min : float
        Minimum road-area ratio (0–1) below which a low-visibility event fires.
    """

    def __init__(self, data_dir: str = "data",
                 debounce_frames: int = 10,
                 road_coverage_min: float = 0.15):
        self.data_dir          = data_dir
        self.debounce_frames   = debounce_frames
        self.road_coverage_min = road_coverage_min

        self._counters   = {k: 0 for k in
                            [EVENT_SPEEDING, EVENT_LANE_DEPARTURE,
                             EVENT_LOW_ROAD_COVERAGE]}
        self._active     = {k: False for k in self._counters}
        self._event_log  = []                    # All events this session

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, seg_result: dict, speed_result: dict):
        """
        Evaluate the current frame's results and emit an event dict if a
        new driving infraction is detected (with debouncing).

        Returns an event dict, or None if no new event this frame.
        """
        conditions = {
            EVENT_SPEEDING:          speed_result.get("speeding", False),
            EVENT_LANE_DEPARTURE:    seg_result.get("departure", False),
            EVENT_LOW_ROAD_COVERAGE: seg_result.get("road_ratio", 1.0)
                                     < self.road_coverage_min,
        }

        new_event = None

        for event_type, triggered in conditions.items():
            if triggered:
                self._counters[event_type] += 1
            else:
                self._counters[event_type] = 0
                self._active[event_type]   = False

            # Fire event on rising edge after debounce window
            if (self._counters[event_type] >= self.debounce_frames
                    and not self._active[event_type]):
                self._active[event_type] = True
                event = self._build_event(event_type, seg_result, speed_result)
                self._event_log.append(event)
                new_event = event     # Return the most recent new event

        return new_event

    def get_summary(self) -> dict:
        """Return a summary dict of the current session's events."""
        counts = {}
        for e in self._event_log:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        return {
            "total_events": len(self._event_log),
            "event_counts": counts,
            "events":       self._event_log,
        }

    def save_session(self, extra_events: list = None) -> str:
        """
        Save the session event log as a JSON file in data_dir.
        Returns the path of the saved file.
        """
        if extra_events:
            self._event_log.extend(extra_events)

        os.makedirs(self.data_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.data_dir, f"session_{timestamp}.json")

        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        return path

    def reset(self):
        """Clear all state for a new session."""
        self._counters  = {k: 0 for k in self._counters}
        self._active    = {k: False for k in self._active}
        self._event_log = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_event(self, event_type: str,
                     seg_result: dict, speed_result: dict) -> dict:
        """Build a structured event record."""
        return {
            "type":       event_type,
            "timestamp":  time.time(),
            "datetime":   datetime.now().isoformat(),
            "speed_kmh":  round(speed_result.get("speed_kmh", 0.0), 2),
            "road_ratio": round(seg_result.get("road_ratio", 0.0), 3),
            "departure":  seg_result.get("departure", False),
        }
