"""
main.py - Fullscreen AI Squat Form Analyzer
===========================================
Real-time, edge-to-edge fullscreen squat tracking using MediaPipe Pose.
Calculates knee angles, depth progress, torso posture, counts reps,
and displays a sleek fitness coach HUD with zero borders or clutter.

Controls:
    Q / ESC  →  Quit
    R        →  Reset Rep Counter
    F        →  Toggle Fullscreen
"""

import os
import cv2
import time
import subprocess

from pose_detector import PoseDetector
from squat_logic import SquatAnalyzer
from utils import draw_hud, draw_angle_arc, draw_minimal_controls


WINDOW_NAME = "Squat Form Analyzer"
CAMERA_INDEX = 0


def get_screen_resolution():
    """Detect primary screen resolution using xrandr."""
    try:
        output = subprocess.check_output(['xrandr'], stderr=subprocess.DEVNULL).decode()
        for line in output.splitlines():
            if '*' in line:
                res = line.split()[0]
                w, h = map(int, res.split('x'))
                return w, h
    except Exception:
        pass
    return 1920, 1080


def fit_to_screen(frame, target_w, target_h):
    """
    Seamlessly fit frame to screen dimensions.
    Performs center-crop if aspect ratio differs to guarantee
    zero black or white bars, then scales cleanly.
    """
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if abs(current_ratio - target_ratio) > 0.02:
        if current_ratio > target_ratio:
            # Frame is wider than target -> center crop width
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            frame = frame[:, start_x:start_x + new_w]
        else:
            # Frame is taller than target -> center crop height
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            frame = frame[start_y:start_y + new_h, :]

    if frame.shape[1] != target_w or frame.shape[0] != target_h:
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return frame


def main():
    """Main application loop."""
    print("[INFO] Starting Squat Form Analyzer...")
    print("[INFO] Controls: [Q/ESC] Quit  |  [R] Reset Reps  |  [F] Toggle Fullscreen\n")

    screen_w, screen_h = get_screen_resolution()
    print(f"[INFO] Screen Resolution: {screen_w}x{screen_h}")

    # ── Initialize detector and squat analyzer ──
    detector = PoseDetector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smoothing_factor=0.45,
    )
    analyzer = SquatAnalyzer()

    # ── Open webcam with MJPG 16:9 widescreen capture ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera connection.")
        return

    actual_cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera capturing at: {actual_cam_w}x{actual_cam_h}")

    # ── Setup Fullscreen Window (Clean, No Qt toolbar) ──
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    # ── Main processing loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Mirror frame naturally
        frame = cv2.flip(frame, 1)

        # Scale and fit frame to fill screen with zero letterboxing / borders
        frame = fit_to_screen(frame, screen_w, screen_h)

        # ── Pose Detection ──
        landmarks = detector.detect(frame)

        if landmarks is not None:
            # Extract side joints & confidence
            side_data = detector.get_side_data(landmarks, frame)

            # Update squat metrics & state machine
            hud_data = analyzer.update(side_data)

            # Draw sleek glowing skeleton overlay
            detector.draw_skeleton(frame, landmarks, active_side=hud_data.get("active_side"))

            # Draw angle indicator arc at active knee
            knee_pt = hud_data.get("knee_pt")
            knee_angle = hud_data.get("knee_angle")
            if knee_pt is not None and knee_angle is not None:
                draw_angle_arc(frame, knee_pt, knee_angle, radius=42, color=hud_data.get("color"))
        else:
            # No person detected at all
            hud_data = analyzer.update({"best_side": None})

        # ── Render HUD Card & Minimal Controls ──
        draw_hud(frame, hud_data)
        draw_minimal_controls(frame)

        # ── Display Frame ──
        cv2.imshow(WINDOW_NAME, frame)

        # ── Handle Keyboard Input ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # 'Q' or ESC
            print("\n[INFO] Quitting...")
            break
        elif key in (ord('r'), ord('R')):     # 'R'
            analyzer.reset()
            print("[INFO] Rep counter reset.")
        elif key in (ord('f'), ord('F')):     # 'F'
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[INFO] Application closed cleanly. Great workout! 💪")


if __name__ == "__main__":
    main()
