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
        self._min_visibility = 0.65    # Only use landmarks above this visibility

    def detect(self, frame):
        """
        Run pose detection on a BGR frame.

        Args:
            frame: A BGR image (NumPy array) from OpenCV.

        Returns:
            A list of normalized landmarks for the first detected person,
            or None if no person was detected.
            Each landmark has .x, .y, .z (normalized 0-1) and .visibility.
        """
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Increment timestamp (must be monotonically increasing in VIDEO mode)
        self._frame_timestamp_ms += 33  # ~30 FPS equivalent

        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]
            self._update_smoothing(raw_landmarks)
            return raw_landmarks
        return None

    def _update_smoothing(self, landmarks):
        """
        Update the EMA-smoothed positions for all tracked landmarks.

        Args:
            landmarks: Raw landmark list from MediaPipe.
        """
        alpha = self._smoothing_factor

        for name, enum_val in self.LANDMARK_NAMES.items():
            idx = enum_val.value
            lm = landmarks[idx]

            # Skip low-visibility landmarks (don't pollute the smoother)
            if lm.visibility < self._min_visibility:
                continue

            raw_x, raw_y = lm.x, lm.y

            if name in self._smoothed_landmarks:
                prev_x, prev_y = self._smoothed_landmarks[name]
                smooth_x = alpha * raw_x + (1 - alpha) * prev_x
                smooth_y = alpha * raw_y + (1 - alpha) * prev_y
            else:
                # First frame: initialize directly
                smooth_x, smooth_y = raw_x, raw_y

            self._smoothed_landmarks[name] = (smooth_x, smooth_y)

    def get_landmark_xy(self, landmarks, name, frame):
        """
        Get the SMOOTHED (x, y) pixel coordinates of a named landmark.

        Uses EMA-filtered positions for stable rendering. Falls back to raw
        coordinates if no smoothed data is available.

        Args:
            landmarks: The list of NormalizedLandmark from detect().
            name: One of the keys in LANDMARK_NAMES (e.g. "LEFT_KNEE").
            frame: The video frame (used to convert normalized → pixel coords).

        Returns:
            (x, y) tuple in pixel coordinates, or None if landmark not visible.
        """
        if name not in self.LANDMARK_NAMES:
            return None

        idx = self.LANDMARK_NAMES[name].value
        lm = landmarks[idx]

        # Reject low-visibility landmarks entirely
        if lm.visibility < self._min_visibility:
            return None

        h, w = frame.shape[:2]

        # Use smoothed coordinates if available
        if name in self._smoothed_landmarks:
            sx, sy = self._smoothed_landmarks[name]
            x = int(sx * w)
            y = int(sy * h)
        else:
            x = int(lm.x * w)
            y = int(lm.y * h)

        return (x, y)

    def get_landmark_visibility(self, landmarks, name):
        """
        Get the visibility confidence of a named landmark.

        Args:
            landmarks: The landmark list.
            name: Landmark name string.

        Returns:
            Visibility score (0.0–1.0), or 0.0 if not found.
        """
        if name not in self.LANDMARK_NAMES:
            return 0.0

        idx = self.LANDMARK_NAMES[name].value
        return landmarks[idx].visibility

    def draw_skeleton(self, frame, landmarks, line_color=(255, 200, 50),
                      joint_color=(0, 255, 200), line_thickness=3,
                      joint_radius=6):
        """
        Draw custom skeleton lines and joint circles on the frame
        using smoothed landmark positions.

        Args:
            frame: The video frame (modified in-place).
            landmarks: The landmark list from detect().
            line_color: BGR color for skeleton lines.
            joint_color: BGR color for joint circles.
            line_thickness: Thickness of skeleton lines.
            joint_radius: Radius of joint circles.
        """
        # Collect smoothed pixel positions for all tracked landmarks
        positions = {}
        for name in self.LANDMARK_NAMES:
            pos = self.get_landmark_xy(landmarks, name, frame)
            if pos is not None:
                positions[name] = pos

        # Draw skeleton lines
        for start_name, end_name in self.SKELETON_CONNECTIONS:
            if start_name in positions and end_name in positions:
                cv2.line(frame, positions[start_name], positions[end_name],
                         line_color, line_thickness, cv2.LINE_AA)

        # Draw joint circles on top
        for name, pos in positions.items():
            cv2.circle(frame, pos, joint_radius, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, joint_radius + 2, line_color, 1, cv2.LINE_AA)

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
