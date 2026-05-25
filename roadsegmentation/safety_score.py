"""
safety_score.py — Real-time and session-final safety score calculation.

Scoring model
─────────────
Base score: 100.0 (perfect drive)

Per-frame deductions:
  • Speeding (minor):    –0.05 / frame
  • Speeding (critical): –0.15 / frame
  • Lane departure:      –0.10 / frame
  • Low road coverage:   –0.02 / frame

Final score is clamped to [0, 100] and rounded to one decimal place.

A letter grade is also provided:
  A  ≥ 90   B  ≥ 80   C  ≥ 70   D  ≥ 60   F  < 60
"""


class SafetyScore:
    """
    Accumulates per-frame safety deductions and exposes a rolling score
    as well as a final session score.

    Parameters
    ----------
    speed_warning_kmh  : float  Speed limit for minor speeding deduction.
    speed_critical_kmh : float  Speed threshold for major deduction.
    """

    GRADE_THRESHOLDS = [(90, "A"), (80, "B"), (70, "C"), (60, "D")]

    # Deduction weights (points lost per offending frame)
    DEDUCTIONS = {
        "speeding_minor":    0.05,
        "speeding_critical": 0.15,
        "lane_departure":    0.10,
        "low_road":          0.02,
    }

    def __init__(self, speed_warning_kmh: float = 60.0,
                 speed_critical_kmh: float = None):
        self.speed_warning  = speed_warning_kmh
        self.speed_critical = speed_critical_kmh or speed_warning_kmh * 1.3

        self._score         = 100.0
        self._frame_count   = 0
        self._offence_tally = {k: 0 for k in self.DEDUCTIONS}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, speed_result: dict, seg_result: dict,
               alert: str | None = None) -> float:
        """
        Apply per-frame deductions and return the current rolling score.

        Parameters
        ----------
        speed_result : dict from SpeedMonitor.update()
        seg_result   : dict from RoadSegmentation.process()
        alert        : alert string from AlertSystem.check() (unused for now,
                       kept for future weight adjustments)

        Returns
        -------
        float — current score in [0, 100]
        """
        self._frame_count += 1
        speed   = speed_result.get("speed_kmh", 0.0)
        depart  = seg_result.get("departure", False)
        road_r  = seg_result.get("road_ratio", 1.0)

        # Speed deductions
        if speed >= self.speed_critical:
            self._deduct("speeding_critical")
        elif speed >= self.speed_warning:
            self._deduct("speeding_minor")

        # Lane departure
        if depart:
            self._deduct("lane_departure")

        # Low road visibility
        if road_r < 0.10:
            self._deduct("low_road")

        return self.current_score

    def final_score(self) -> float:
        """Return the final clamped score for the session."""
        return self.current_score

    @property
    def current_score(self) -> float:
        return round(max(0.0, min(100.0, self._score)), 1)

    @property
    def grade(self) -> str:
        """Letter grade for the current score."""
        for threshold, letter in self.GRADE_THRESHOLDS:
            if self.current_score >= threshold:
                return letter
        return "F"

    def report(self) -> dict:
        """Return a full report dict for logging or display."""
        return {
            "score":         self.current_score,
            "grade":         self.grade,
            "frames":        self._frame_count,
            "offence_tally": dict(self._offence_tally),
            "interpretation": self._interpretation(),
        }

    def reset(self):
        """Reset for a new session."""
        self._score         = 100.0
        self._frame_count   = 0
        self._offence_tally = {k: 0 for k in self.DEDUCTIONS}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _deduct(self, offence: str):
        self._score -= self.DEDUCTIONS[offence]
        self._offence_tally[offence] += 1

    def _interpretation(self) -> str:
        s = self.current_score
        if s >= 90: return "Excellent — very safe driving."
        if s >= 80: return "Good — minor infractions detected."
        if s >= 70: return "Fair — several safety issues observed."
        if s >= 60: return "Poor — significant safety concerns."
        return "Dangerous — immediate attention required."
