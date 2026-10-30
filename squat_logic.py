"""
squat_logic.py - Accurate squat form analysis, depth tracking, and rep counting.
"""

import time
import numpy as np
from angle_calculator import get_knee_angle, get_torso_lean


class SquatAnalyzer:
    """
    Super-accurate, intuitive squat analyzer with real-time depth calculation
    and clear visual feedback.
    """

    # ──────────────────────────────────────────────
    # Biomechanical Thresholds (Degrees)
    # ──────────────────────────────────────────────
    KNEE_STAND_ANGLE = 158.0      # Above this → Standing straight (0% depth)
    KNEE_SQUAT_START = 142.0      # Below this → Squat descent begins
    KNEE_TARGET_DEPTH = 100.0     # Target parallel depth (100% depth)
    KNEE_DEPTH_TOLERANCE = 106.0  # Max angle to qualify as a valid rep
    MAX_TORSO_LEAN = 45.0         # Above this → Excessive forward lean (chest dropped)

    # ──────────────────────────────────────────────
    # Color Palette (BGR)
    # ──────────────────────────────────────────────
    COLOR_READY = (255, 230, 80)      # Clean Cyan
    COLOR_GOOD = (0, 235, 120)        # Emerald Green
    COLOR_WARNING = (0, 215, 255)     # Warm Yellow/Gold
    COLOR_BAD = (60, 76, 235)         # Vibrant Red
    COLOR_NEUTRAL = (180, 180, 180)   # Clean Silver Gray

    def __init__(self):
        self.rep_count = 0
        self.state = "WAITING"        # WAITING, READY, SQUATTING, REP_COMPLETE
        self.feedback = "Step back until full body is in view"
        self.feedback_color = self.COLOR_NEUTRAL
        self.depth_progress = 0.0     # 0.0 to 1.0 (0% to 100%)
        self.knee_angle = None
        self.torso_lean = None
        self.active_side = None

        self._min_knee_angle = 180.0
        self._rep_completion_time = 0.0
        self._min_rep_interval = 0.8  # Prevent double-counts within 0.8s

    def reset(self):
        """Reset rep count and state."""
        self.rep_count = 0
        self.state = "WAITING"
        self.feedback = "Step back until full body is in view"
        self.feedback_color = self.COLOR_NEUTRAL
        self.depth_progress = 0.0
        self.knee_angle = None
        self.torso_lean = None
        self._min_knee_angle = 180.0
        self._rep_completion_time = 0.0

    def update(self, side_data):
        """
        Process detected pose joints and update squat metrics.

        Args:
            side_data: Dictionary returned by PoseDetector.get_side_data(...)

        Returns:
            Dictionary with feedback, color, rep_count, depth_progress, knee_angle, etc.
        """
        best_side = side_data.get("best_side")
        self.active_side = best_side

        # If user is not far enough for full body tracking
        if best_side is None:
            self.state = "WAITING"
            self.feedback = "Step back so legs are visible"
            self.feedback_color = self.COLOR_WARNING
            self.depth_progress = 0.0
            self.knee_angle = None
            self.torso_lean = None
            self._min_knee_angle = 180.0
            return self._build_result(None)

        # Extract joints for the best/active side
        joints = side_data["left"] if best_side == "LEFT" else side_data["right"]
        shoulder, hip, knee, ankle = joints

        # Compute angles
        knee_angle = get_knee_angle(hip, knee, ankle)
        torso_lean = get_torso_lean(shoulder, hip)
        self.knee_angle = knee_angle
        self.torso_lean = torso_lean

        # Compute continuous depth progress (0.0 = standing at >=158°, 1.0 = target depth at <=100°)
        depth_raw = (self.KNEE_STAND_ANGLE - knee_angle) / (self.KNEE_STAND_ANGLE - self.KNEE_TARGET_DEPTH)
        self.depth_progress = float(np.clip(depth_raw, 0.0, 1.0))

        now = time.time()

        # Handle momentary celebratory state after rep completion
        if self.state == "REP_COMPLETE":
            if now - self._rep_completion_time > 1.2:
                self.state = "READY"

        # ── STATE: WAITING -> READY ──
        if self.state == "WAITING":
            if knee_angle > self.KNEE_STAND_ANGLE - 5.0:
                self.state = "READY"
                self.feedback = "Ready — Squat down"
                self.feedback_color = self.COLOR_READY
            else:
                self.feedback = "Stand tall to begin"
                self.feedback_color = self.COLOR_READY

        # ── STATE: READY (Standing) ──
        elif self.state == "READY":
            if knee_angle < self.KNEE_SQUAT_START:
                self.state = "SQUATTING"
                self._min_knee_angle = knee_angle
                self.feedback = "Squat down..."
                self.feedback_color = self.COLOR_WARNING
            else:
                self.feedback = "Ready — Squat down"
                self.feedback_color = self.COLOR_READY

        # ── STATE: SQUATTING (Descent / Ascent) ──
        elif self.state == "SQUATTING":
            self._min_knee_angle = min(self._min_knee_angle, knee_angle)

            # Live feedback during movement
            if torso_lean > self.MAX_TORSO_LEAN:
                self.feedback = "Chest Up — Straighten Back!"
                self.feedback_color = self.COLOR_BAD
            elif self._min_knee_angle <= self.KNEE_TARGET_DEPTH:
                self.feedback = "Good Depth — Stand Up!"
                self.feedback_color = self.COLOR_GOOD
            else:
                self.feedback = "Go Lower..."
                self.feedback_color = self.COLOR_WARNING

            # Returning to standing
            if knee_angle >= self.KNEE_STAND_ANGLE - 5.0:
                # Check if depth was valid
                if self._min_knee_angle <= self.KNEE_DEPTH_TOLERANCE and (now - self._rep_completion_time) > self._min_rep_interval:
                    self.rep_count += 1
                    self._rep_completion_time = now
                    self.state = "REP_COMPLETE"
                    self.feedback = "Good Rep! (+1)"
                    self.feedback_color = self.COLOR_GOOD
                else:
                    self.state = "READY"
                    self.feedback = "Go Lower Next Time"
                    self.feedback_color = self.COLOR_WARNING

                self._min_knee_angle = 180.0

        return self._build_result(knee)

    def _build_result(self, knee_pt):
        return {
            "feedback": self.feedback,
            "color": self.feedback_color,
            "rep_count": self.rep_count,
            "depth_progress": self.depth_progress,
            "knee_angle": self.knee_angle,
            "torso_lean": self.torso_lean,
            "active_side": self.active_side,
            "knee_pt": knee_pt,
        }
