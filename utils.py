"""
utils.py - Helper functions for drawing overlays and text on the video feed.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Color palette (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_GREEN = (0, 230, 118)      # Good feedback
COLOR_RED = (60, 76, 231)        # Bad feedback
COLOR_YELLOW = (0, 215, 255)     # Warning / go lower
COLOR_ORANGE = (0, 140, 255)     # Incorrect form
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_SKELETON = (255, 200, 50)  # Skeleton lines
COLOR_JOINT = (0, 255, 200)      # Joint dots
COLOR_PANEL_BG = (30, 30, 30)    # Dark panel background


def draw_text_with_bg(frame, text, position, font_scale=0.7,
                      text_color=COLOR_WHITE, bg_color=COLOR_BLACK,
                      thickness=2, padding=8, alpha=0.7):
    """
    Draw text with a semi-transparent background rectangle.
    
    Args:
        frame: The video frame (modified in-place).
        text: The string to display.
        position: (x, y) top-left corner of the text.
        font_scale: Font size multiplier.
        text_color: BGR color for the text.
        bg_color: BGR color for the background rectangle.
        thickness: Text thickness.
        padding: Padding around the text inside the rectangle.
        alpha: Opacity of the background (0.0 to 1.0).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    # Measure the text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Rectangle coordinates
    x1 = x - padding
    y1 = y - text_h - padding
    x2 = x + text_w + padding
    y2 = y + baseline + padding

    # Clamp to frame boundaries
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness,
                cv2.LINE_AA)


def draw_angle_arc(frame, center, angle_deg, radius=40, color=COLOR_CYAN,
                   thickness=2):
    """
    Draw a small arc near a joint to visualize the measured angle.
    
    Args:
        frame: The video frame.
        center: (x, y) joint position in pixel coordinates.
        angle_deg: The angle value to display.
        radius: Arc radius in pixels.
        color: BGR color.
        thickness: Line thickness.
    """
    # Draw a circle outline at the joint
    cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)

    # Put the angle value near the joint
    text = f"{int(angle_deg)} deg"
    text_pos = (center[0] + radius + 5, center[1] - 5)
    draw_text_with_bg(frame, text, text_pos, font_scale=0.5,
                      text_color=color, bg_color=COLOR_BLACK,
                      thickness=1, padding=4, alpha=0.6)


def draw_feedback_panel(frame, feedback_text, feedback_color, rep_count,
                        knee_angle, hip_angle):
    """
    Draw a heads-up display panel with feedback, rep count, and angles.
    
    Args:
        frame: The video frame.
        feedback_text: The main feedback string (e.g. "Good Squat").
        feedback_color: BGR color matching the feedback severity.
        rep_count: Current repetition count.
        knee_angle: Current knee angle in degrees (or None).
        hip_angle: Current hip angle in degrees (or None).
    """
    h, w = frame.shape[:2]
    panel_h = 160
    panel_w = 320

    # Draw semi-transparent panel in the top-left corner
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                  feedback_color, 2, cv2.LINE_AA)

    # Title
    draw_text_with_bg(frame, "SQUAT ANALYZER", (25, 40),
                      font_scale=0.65, text_color=COLOR_WHITE,
                      bg_color=(0, 0, 0), thickness=2, padding=4, alpha=0.0)

    # Feedback line
    draw_text_with_bg(frame, feedback_text, (25, 72),
                      font_scale=0.7, text_color=feedback_color,
                      bg_color=(0, 0, 0), thickness=2, padding=4, alpha=0.0)

    # Rep counter
    draw_text_with_bg(frame, f"Reps: {rep_count}", (25, 102),
                      font_scale=0.6, text_color=COLOR_WHITE,
                      bg_color=(0, 0, 0), thickness=2, padding=4, alpha=0.0)

    # Angles
    knee_str = f"Knee: {int(knee_angle)} deg" if knee_angle is not None else "Knee: --"
    hip_str = f"Hip:  {int(hip_angle)} deg" if hip_angle is not None else "Hip:  --"

    draw_text_with_bg(frame, knee_str, (25, 130),
                      font_scale=0.5, text_color=COLOR_CYAN,
                      bg_color=(0, 0, 0), thickness=1, padding=4, alpha=0.0)
    draw_text_with_bg(frame, hip_str, (25, 155),
                      font_scale=0.5, text_color=COLOR_CYAN,
                      bg_color=(0, 0, 0), thickness=1, padding=4, alpha=0.0)


def draw_instructions(frame):
    """
    Draw keyboard shortcut hints at the bottom of the frame.
    """
    h, w = frame.shape[:2]
    instructions = "Press 'Q' to Quit  |  'R' to Reset Reps"
    draw_text_with_bg(frame, instructions, (15, h - 15),
                      font_scale=0.45, text_color=COLOR_WHITE,
                      bg_color=COLOR_PANEL_BG, thickness=1, padding=6,
                      alpha=0.7)
