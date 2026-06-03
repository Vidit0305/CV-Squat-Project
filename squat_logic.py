"""
squat_logic.py - Squat validation rules, state machine, and rep counting.

The squat cycle state machine:
    IDLE      →  (standing posture confirmed for N frames)  →  STANDING
    STANDING  →  (knee angle drops below threshold)         →  SQUATTING
    SQUATTING →  (knee angle rises above threshold)         →  STANDING  (+1 rep)

Key improvement over naive angle-only detection:
    • Requires confirmed upright standing posture before counting begins.
    • Checks that ankles are roughly below hips (rejects sitting in a chair
      where legs extend forward).
    • Uses averaged left+right angles for robustness.

Feedback is generated based on:
    • Knee angle (depth check)
    • Hip angle (back straightness)
    • Ankle-under-hip alignment (sitting rejection)
"""


class SquatAnalyzer:
    """
    Analyzes squat form and counts repetitions.

    Attributes:
        rep_count: Number of completed squat reps.
        state: Current phase — "IDLE", "STANDING", or "SQUATTING".
        feedback: Latest feedback string.
        feedback_color: BGR color matching the feedback severity.
    """

    # ──────────────────────────────────────────────
    # Tunable thresholds (degrees)
    # ──────────────────────────────────────────────
    KNEE_ANGLE_SQUAT = 120     # Below this → transition to SQUATTING
    KNEE_ANGLE_STAND = 155     # Above this → transition back to STANDING
    KNEE_ANGLE_DEEP = 100      # Below this → "good depth" achieved

    HIP_ANGLE_MIN = 55         # Below this → back way too bent forward
    HIP_ANGLE_GOOD_MIN = 65    # Above this → back is acceptably straight

    # Standing confirmation: must see this many consecutive frames with
    # knee angle > KNEE_ANGLE_STAND before we start tracking squats.
    STANDING_CONFIRM_FRAMES = 10

    # Maximum horizontal distance (as fraction of frame width) between
    # ankle and hip x-coordinates. If ankles are too far forward from
    # hips, the person is likely sitting in a chair, not squatting.
    MAX_ANKLE_HIP_X_RATIO = 0.20  # 20% of frame width

    # ──────────────────────────────────────────────
    # Feedback color palette (BGR)
    # ──────────────────────────────────────────────
    COLOR_GOOD = (0, 230, 118)       # Green
    COLOR_WARNING = (0, 215, 255)    # Yellow
    COLOR_BAD = (60, 76, 231)        # Red
    COLOR_NEUTRAL = (200, 200, 200)  # Gray

    def __init__(self):
        self.rep_count = 0
        self.state = "IDLE"         # "IDLE", "STANDING", or "SQUATTING"
        self.feedback = "Stand up straight"
        self.feedback_color = self.COLOR_NEUTRAL

        # Track the minimum knee angle reached during a squat
        self._min_knee_angle = 180.0
        # Counter for standing confirmation
        self._standing_frames = 0

    def reset(self):
        """Reset rep counter and state."""
        self.rep_count = 0
        self.state = "IDLE"
        self.feedback = "Stand up straight"
        self.feedback_color = self.COLOR_NEUTRAL
        self._min_knee_angle = 180.0
        self._standing_frames = 0

    def update(self, knee_angle, hip_angle, hip_pos, knee_pos, ankle_pos,
               frame_width):
        """
        Process one frame's worth of joint data and update feedback/state.

        Args:
            knee_angle: Angle at the knee (hip–knee–ankle) in degrees.
            hip_angle: Angle at the hip (shoulder–hip–knee) in degrees.
            hip_pos: (x, y) pixel coordinates of the hip.
            knee_pos: (x, y) pixel coordinates of the knee.
            ankle_pos: (x, y) pixel coordinates of the ankle.
            frame_width: Width of the video frame in pixels.

        Returns:
            Tuple of (feedback_text, feedback_color_bgr).
        """
        hip_x, hip_y = hip_pos
        knee_x, knee_y = knee_pos
        ankle_x, ankle_y = ankle_pos

        # ── Sitting rejection ──
        # In a chair, ankles are typically far forward of the hips horizontally.
        # In a squat, ankles stay roughly under or slightly ahead of the hips.
        ankle_hip_x_dist = abs(ankle_x - hip_x) / max(frame_width, 1)
        is_sitting_posture = (
            ankle_hip_x_dist > self.MAX_ANKLE_HIP_X_RATIO
            and ankle_y < hip_y  # ankles higher than hips on screen = legs forward
        )

        # ── Back straightness check ──
        back_ok = hip_angle >= self.HIP_ANGLE_GOOD_MIN
        back_very_bad = hip_angle < self.HIP_ANGLE_MIN

        # ────────────────────────────────────────
        # State: IDLE — waiting for the person to stand upright
        # ────────────────────────────────────────
        if self.state == "IDLE":
            if knee_angle > self.KNEE_ANGLE_STAND and not is_sitting_posture:
                self._standing_frames += 1
            else:
                self._standing_frames = 0

            if self._standing_frames >= self.STANDING_CONFIRM_FRAMES:
                self.state = "STANDING"
                self._standing_frames = 0
                self.feedback = "Ready — begin squat"
                self.feedback_color = self.COLOR_NEUTRAL
            else:
                self.feedback = "Stand up straight"
                self.feedback_color = self.COLOR_NEUTRAL

            return self.feedback, self.feedback_color

        # ────────────────────────────────────────
        # State: STANDING — waiting for squat to begin
        # ────────────────────────────────────────
        if self.state == "STANDING":
            # If we detect a sitting posture, go back to IDLE
            if is_sitting_posture:
                self.state = "IDLE"
                self._standing_frames = 0
                self.feedback = "Stand up straight"
                self.feedback_color = self.COLOR_NEUTRAL
                return self.feedback, self.feedback_color

            if knee_angle < self.KNEE_ANGLE_SQUAT:
                # Transition → SQUATTING
                self.state = "SQUATTING"
                self._min_knee_angle = knee_angle
            else:
                self.feedback = "Ready — begin squat"
                self.feedback_color = self.COLOR_NEUTRAL
                return self.feedback, self.feedback_color

        # ────────────────────────────────────────
        # State: SQUATTING — tracking the squat movement
        # ────────────────────────────────────────
        if self.state == "SQUATTING":
            # If sitting posture detected mid-squat, reject and reset
            if is_sitting_posture:
                self.state = "IDLE"
                self._standing_frames = 0
                self._min_knee_angle = 180.0
                self.feedback = "Stand up — not a squat"
                self.feedback_color = self.COLOR_BAD
                return self.feedback, self.feedback_color

            # Track lowest angle
            self._min_knee_angle = min(self._min_knee_angle, knee_angle)

            if knee_angle > self.KNEE_ANGLE_STAND:
                # Transition → STANDING (rep completed)
                self.state = "STANDING"

                # Evaluate the completed rep
                if self._min_knee_angle <= self.KNEE_ANGLE_DEEP and back_ok:
                    self.rep_count += 1
                    self.feedback = "Good Squat!"
                    self.feedback_color = self.COLOR_GOOD
                elif not back_ok:
                    self.rep_count += 1
                    self.feedback = "Keep Back Straight"
                    self.feedback_color = self.COLOR_BAD
                else:
                    self.rep_count += 1
                    self.feedback = "Go Lower"
                    self.feedback_color = self.COLOR_WARNING

                self._min_knee_angle = 180.0
                return self.feedback, self.feedback_color

            # ── Live feedback while in the squat ──
            if back_very_bad:
                self.feedback = "Keep Back Straight!"
                self.feedback_color = self.COLOR_BAD
            elif not back_ok:
                self.feedback = "Straighten Back"
                self.feedback_color = self.COLOR_WARNING
            elif knee_angle > self.KNEE_ANGLE_DEEP:
                self.feedback = "Go Lower"
                self.feedback_color = self.COLOR_WARNING
            else:
                self.feedback = "Good Depth — Hold!"
                self.feedback_color = self.COLOR_GOOD

        return self.feedback, self.feedback_color
