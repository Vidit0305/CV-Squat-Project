"""
pose_detector.py - Wrapper around MediaPipe PoseLandmarker (Tasks API)
       for body landmark detection with EMA smoothing.

Provides a clean interface to:
    1. Initialize the PoseLandmarker model.
    2. Detect landmarks from a video frame.
    3. Smooth landmark positions with exponential moving average (reduces jitter).
    4. Extract specific landmark coordinates in pixel space.
    5. Draw the skeleton overlay on the frame.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
RunningMode = mp.tasks.vision.RunningMode


class PoseDetector:
    """
    Wraps MediaPipe PoseLandmarker to detect and draw body landmarks,
    with built-in EMA smoothing for stable skeleton rendering.

    Usage:
        detector = PoseDetector()
        landmarks = detector.detect(frame)
        if landmarks:
            detector.draw_skeleton(frame, landmarks)
            hip = detector.get_landmark_xy(landmarks, "LEFT_HIP", frame)
    """

    # Landmark name → PoseLandmark enum mapping (commonly used joints)
    LANDMARK_NAMES = {
        "LEFT_SHOULDER":  PoseLandmark.LEFT_SHOULDER,
        "RIGHT_SHOULDER": PoseLandmark.RIGHT_SHOULDER,
        "LEFT_HIP":       PoseLandmark.LEFT_HIP,
        "RIGHT_HIP":      PoseLandmark.RIGHT_HIP,
        "LEFT_KNEE":      PoseLandmark.LEFT_KNEE,
        "RIGHT_KNEE":     PoseLandmark.RIGHT_KNEE,
        "LEFT_ANKLE":     PoseLandmark.LEFT_ANKLE,
        "RIGHT_ANKLE":    PoseLandmark.RIGHT_ANKLE,
    }

    # Skeleton connections to draw (pairs of landmark names)
    SKELETON_CONNECTIONS = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_HIP", "LEFT_KNEE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]

    # Try full model first, fall back to lite
    DEFAULT_MODEL_PATHS = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "pose_landmarker_full.task"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "pose_landmarker_lite.task"),
    ]

    def __init__(self, model_path=None, min_detection_confidence=0.7,
                 min_tracking_confidence=0.7, smoothing_factor=0.4):
        """
        Initialize the MediaPipe PoseLandmarker model.

        Args:
            model_path: Path to the .task model file. If None, auto-detects
                        full model first, then falls back to lite.
            min_detection_confidence: Minimum confidence for person detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            smoothing_factor: EMA alpha (0–1). Lower = smoother but laggier.
                              0.4 is a good balance for real-time pose.
        """
        # Auto-detect model
        if model_path is None:
            for path in self.DEFAULT_MODEL_PATHS:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Pose model not found. Download it with:\n"
                "  Invoke-WebRequest -Uri "
                "\"https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/"
                "pose_landmarker_full.task\" -OutFile "
                "\"pose_landmarker_full.task\""
            )

        print(f"[INFO] Using pose model: {os.path.basename(model_path)}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)

        # Frame counter for video-mode timestamps (must be monotonically increasing)
        self._frame_timestamp_ms = 0

        # ── EMA Smoothing ──
        self._smoothing_factor = smoothing_factor  # alpha: higher = more responsive
        self._smoothed_landmarks = {}  # name → (x_smooth, y_smooth) in normalized coords
        self._min_visibility = 0.45    # Reliable tracking even in normal indoor lighting

    def detect(self, frame):
        """
        Run pose detection on a BGR frame.

        Args:
            frame: A BGR image (NumPy array) from OpenCV.

        Returns:
            A list of normalized landmarks for the first detected person,
            or None if no person was detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33  # ~30 FPS equivalent
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]
            self._update_smoothing(raw_landmarks)
            return raw_landmarks
        return None

    def _update_smoothing(self, landmarks):
        """Update EMA-smoothed positions for tracked landmarks."""
        alpha = self._smoothing_factor

        for name, enum_val in self.LANDMARK_NAMES.items():
            idx = enum_val.value
            lm = landmarks[idx]

            if lm.visibility < self._min_visibility:
                continue

            raw_x, raw_y = lm.x, lm.y

            if name in self._smoothed_landmarks:
                prev_x, prev_y = self._smoothed_landmarks[name]
                smooth_x = alpha * raw_x + (1 - alpha) * prev_x
                smooth_y = alpha * raw_y + (1 - alpha) * prev_y
            else:
                smooth_x, smooth_y = raw_x, raw_y

            self._smoothed_landmarks[name] = (smooth_x, smooth_y)

    def get_landmark_xy(self, landmarks, name, frame):
        """
        Get smoothed (x, y) pixel coordinates of a landmark.
        Returns None if below visibility threshold.
        """
        if name not in self.LANDMARK_NAMES:
            return None

        idx = self.LANDMARK_NAMES[name].value
        lm = landmarks[idx]

        if lm.visibility < self._min_visibility:
            return None

        h, w = frame.shape[:2]
        if name in self._smoothed_landmarks:
            sx, sy = self._smoothed_landmarks[name]
            x = int(sx * w)
            y = int(sy * h)
        else:
            x = int(lm.x * w)
            y = int(lm.y * h)

        return (x, y)

    def get_landmark_visibility(self, landmarks, name):
        """Get visibility score (0.0–1.0) of a named landmark."""
        if name not in self.LANDMARK_NAMES:
            return 0.0
        idx = self.LANDMARK_NAMES[name].value
        return float(landmarks[idx].visibility)

    def get_side_data(self, landmarks, frame):
        """
        Extract joints and confidence for left and right body sides.
        Returns:
            dict containing left joints, right joints, visibilities, and best side.
        """
        left_shoulder = self.get_landmark_xy(landmarks, "LEFT_SHOULDER", frame)
        right_shoulder = self.get_landmark_xy(landmarks, "RIGHT_SHOULDER", frame)
        left_hip = self.get_landmark_xy(landmarks, "LEFT_HIP", frame)
        right_hip = self.get_landmark_xy(landmarks, "RIGHT_HIP", frame)
        left_knee = self.get_landmark_xy(landmarks, "LEFT_KNEE", frame)
        right_knee = self.get_landmark_xy(landmarks, "RIGHT_KNEE", frame)
        left_ankle = self.get_landmark_xy(landmarks, "LEFT_ANKLE", frame)
        right_ankle = self.get_landmark_xy(landmarks, "RIGHT_ANKLE", frame)

        v_left = (self.get_landmark_visibility(landmarks, "LEFT_HIP") +
                  self.get_landmark_visibility(landmarks, "LEFT_KNEE") +
                  self.get_landmark_visibility(landmarks, "LEFT_ANKLE")) / 3.0

        v_right = (self.get_landmark_visibility(landmarks, "RIGHT_HIP") +
                   self.get_landmark_visibility(landmarks, "RIGHT_KNEE") +
                   self.get_landmark_visibility(landmarks, "RIGHT_ANKLE")) / 3.0

        left_valid = all(p is not None for p in [left_shoulder, left_hip, left_knee, left_ankle])
        right_valid = all(p is not None for p in [right_shoulder, right_hip, right_knee, right_ankle])

        best_side = None
        if left_valid and right_valid:
            best_side = "LEFT" if v_left >= v_right else "RIGHT"
        elif left_valid:
            best_side = "LEFT"
        elif right_valid:
            best_side = "RIGHT"

        return {
            "left": (left_shoulder, left_hip, left_knee, left_ankle),
            "right": (right_shoulder, right_hip, right_knee, right_ankle),
            "v_left": v_left,
            "v_right": v_right,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "best_side": best_side,
        }

    def draw_skeleton(self, frame, landmarks, active_side=None):
        """
        Draw sleek, high-tech glowing skeleton overlay.
        """
        positions = {}
        for name in self.LANDMARK_NAMES:
            pos = self.get_landmark_xy(landmarks, name, frame)
            if pos is not None:
                positions[name] = pos

        # Base line color (slate blue / subtle)
        base_line_color = (220, 160, 40)
        active_line_color = (0, 240, 255)  # Cyan/Gold for active side

        # Draw skeleton connections
        for start_name, end_name in self.SKELETON_CONNECTIONS:
            if start_name in positions and end_name in positions:
                is_active = (active_side and active_side in start_name and active_side in end_name)
                color = active_line_color if is_active else base_line_color
                thickness = 4 if is_active else 2
                cv2.line(frame, positions[start_name], positions[end_name],
                         color, thickness, cv2.LINE_AA)

        # Draw glowing joint nodes
        for name, pos in positions.items():
            is_active_node = (active_side and active_side in name)
            inner_color = (0, 255, 180) if is_active_node else (0, 220, 255)
            # Outer soft glow ring
            cv2.circle(frame, pos, 8, (255, 255, 255), 1, cv2.LINE_AA)
            # Inner solid circle
            cv2.circle(frame, pos, 5, inner_color, -1, cv2.LINE_AA)

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

