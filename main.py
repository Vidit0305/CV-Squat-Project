"""
main.py - Squat Form Analyzer
==============================
Real-time squat form detection using your webcam.

Detects body landmarks via MediaPipe Pose, calculates knee and hip angles,
validates squat form, counts reps, and displays live visual feedback.

Controls:
    Q  →  Quit
    R  →  Reset rep counter

Usage:
    python main.py
"""

import cv2
import time

from pose_detector import PoseDetector
from angle_calculator import get_knee_angle, get_hip_angle
from squat_logic import SquatAnalyzer
from utils import (
    draw_angle_arc,
    draw_feedback_panel,
    draw_instructions,
    COLOR_GREEN,
    COLOR_WHITE,
    COLOR_CYAN,
)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
WINDOW_NAME = "Squat Form Analyzer"
CAMERA_INDEX = 0          # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def _avg_point(p1, p2):
    """Average two (x, y) points. Returns None if either is None."""
    if p1 is None or p2 is None:
        return p1 or p2  # Use whichever is available
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def main():
    """Main application loop."""

    # ── Initialize components ──
    print("[INFO] Starting Squat Form Analyzer...")
    print("[INFO] Press 'Q' to quit, 'R' to reset reps.\n")

    detector = PoseDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smoothing_factor=0.4,
    )
    analyzer = SquatAnalyzer()

    # ── Open webcam ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        return

    print(f"[INFO] Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # For FPS calculation
    prev_time = time.time()

    # ──────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Mirror the frame so it feels natural
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # ── Detect pose landmarks ──
        landmarks = detector.detect(frame)

        knee_angle = None
        hip_angle = None

        if landmarks is not None:
            # Draw skeleton overlay
            detector.draw_skeleton(frame, landmarks)

            # ── Extract key joint positions (both sides) ──
            left_shoulder = detector.get_landmark_xy(landmarks, "LEFT_SHOULDER", frame)
            right_shoulder = detector.get_landmark_xy(landmarks, "RIGHT_SHOULDER", frame)
            left_hip = detector.get_landmark_xy(landmarks, "LEFT_HIP", frame)
            right_hip = detector.get_landmark_xy(landmarks, "RIGHT_HIP", frame)
            left_knee = detector.get_landmark_xy(landmarks, "LEFT_KNEE", frame)
            right_knee = detector.get_landmark_xy(landmarks, "RIGHT_KNEE", frame)
            left_ankle = detector.get_landmark_xy(landmarks, "LEFT_ANKLE", frame)
            right_ankle = detector.get_landmark_xy(landmarks, "RIGHT_ANKLE", frame)

            # Average left + right for more stable readings
            shoulder = _avg_point(left_shoulder, right_shoulder)
            hip = _avg_point(left_hip, right_hip)
            knee = _avg_point(left_knee, right_knee)
            ankle = _avg_point(left_ankle, right_ankle)

            # Check that all critical landmarks are available
            if all(p is not None for p in [shoulder, hip, knee, ankle]):

                # ── Calculate angles ──
                knee_angle = get_knee_angle(hip, knee, ankle)
                hip_angle = get_hip_angle(shoulder, hip, knee)

                # ── Update squat analyzer ──
                feedback, feedback_color = analyzer.update(
                    knee_angle=knee_angle,
                    hip_angle=hip_angle,
                    hip_pos=hip,
                    knee_pos=knee,
                    ankle_pos=ankle,
                    frame_width=frame_w,
                )

                # ── Draw angle arcs at the averaged joint positions ──
                draw_angle_arc(frame, knee, knee_angle, radius=30,
                               color=COLOR_CYAN)
                draw_angle_arc(frame, hip, hip_angle, radius=30,
                               color=COLOR_GREEN)

        # ── Draw HUD panel ──
        draw_feedback_panel(
            frame,
            feedback_text=analyzer.feedback,
            feedback_color=analyzer.feedback_color,
            rep_count=analyzer.rep_count,
            knee_angle=knee_angle,
            hip_angle=hip_angle,
        )

        # ── FPS counter ──
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (frame_w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)

        # ── Keyboard hints ──
        draw_instructions(frame)

        # ── Show the frame ──
        cv2.imshow(WINDOW_NAME, frame)

        # ── Handle keyboard input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Quitting...")
            break
        elif key == ord('r') or key == ord('R'):
            analyzer.reset()
            print("[INFO] Rep counter reset.")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[INFO] Application closed. Great workout! 💪")


if __name__ == "__main__":
    main()
