"""
utils.py - High-precision rendering of HUD panels, depth gauges, and angles.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Curated Modern Color Palette (BGR format)
# ──────────────────────────────────────────────
COLOR_EMERALD = (80, 230, 110)    # Good / Valid / Target reached
COLOR_CYAN = (255, 220, 40)       # Active / Ready / Tracking
COLOR_GOLD = (30, 210, 255)       # Warning / Descending
COLOR_CORAL = (60, 70, 240)       # Alert / Bad posture
COLOR_WHITE = (255, 255, 255)
COLOR_MUTED = (160, 160, 160)
COLOR_PANEL_BG = (18, 20, 24)     # Deep slate background
COLOR_SLOT = (45, 50, 58)         # Progress bar slot background


def draw_hud(frame, data):
    """
    Draw a clean, modern, high-contrast workout HUD.
    Displays only the essential metrics: Rep Count, Form Feedback,
    Depth Progress Bar, and Knee Angle.

    Args:
        frame: The video frame to draw on (modified in-place).
        data: Dict containing 'feedback', 'color', 'rep_count',
              'depth_progress', 'knee_angle', 'torso_lean', etc.
    """
    h, w = frame.shape[:2]

    # Panel dimensions & position (top-left)
    panel_x = 30
    panel_y = 30
    panel_w = 460
    panel_h = 210

    # 1. Glassmorphism dark background panel
    sub_img = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
    bg_rect = np.full(sub_img.shape, COLOR_PANEL_BG, dtype=np.uint8)
    blended = cv2.addWeighted(bg_rect, 0.82, sub_img, 0.18, 0)
    frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = blended

    # 2. Glowing Accent Border matching current status color
    border_color = data.get("color", COLOR_CYAN)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  border_color, 2, cv2.LINE_AA)

    # 3. Header: App title + Large Rep Counter Badge
    cv2.putText(frame, "SQUAT AI COACH", (panel_x + 20, panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MUTED, 1, cv2.LINE_AA)

    # Rep count pill badge (top-right of panel)
    rep_text = f"REPS: {data.get('rep_count', 0)}"
    (rw, rh), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    pill_x = panel_x + panel_w - rw - 35
    pill_y = panel_y + 15
    cv2.rectangle(frame, (pill_x - 10, pill_y), (pill_x + rw + 10, pill_y + rh + 14),
                  (35, 42, 52), -1)
    cv2.rectangle(frame, (pill_x - 10, pill_y), (pill_x + rw + 10, pill_y + rh + 14),
                  COLOR_EMERALD, 2, cv2.LINE_AA)
    cv2.putText(frame, rep_text, (pill_x, pill_y + rh + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_WHITE, 2, cv2.LINE_AA)

    # 4. Large Form Feedback Headline
    feedback = data.get("feedback", "Ready")
    cv2.putText(frame, feedback, (panel_x + 20, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, border_color, 2, cv2.LINE_AA)

    # 5. Squat Depth Progress Bar
    bar_x = panel_x + 20
    bar_y = panel_y + 125
    bar_w = panel_w - 40
    bar_h = 22

    # Background slot
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  COLOR_SLOT, -1)

    # Filled progress
    progress = data.get("depth_progress", 0.0)
    fill_w = int(bar_w * progress)

    if progress >= 0.95:
        bar_color = COLOR_EMERALD
    elif progress >= 0.5:
        bar_color = COLOR_GOLD
    else:
        bar_color = COLOR_CYAN

    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1)

    # Target depth mark at 100%
    cv2.rectangle(frame, (bar_x + bar_w - 3, bar_y - 3),
                  (bar_x + bar_w, bar_y + bar_h + 3), COLOR_WHITE, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 85, 95), 1, cv2.LINE_AA)

    # Depth labels
    pct = int(progress * 100)
    cv2.putText(frame, f"Squat Depth: {pct}%", (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Target Depth", (bar_x + bar_w - 95, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)

    # 6. Knee Angle Metric
    knee_angle = data.get("knee_angle")
    if knee_angle is not None:
        angle_str = f"Knee Angle: {int(knee_angle)} deg  (Target: <=100 deg)"
    else:
        angle_str = "Knee Angle: --"
    cv2.putText(frame, angle_str, (panel_x + 20, panel_y + 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_angle_arc(frame, center, angle_deg, radius=42, color=COLOR_CYAN):
    """
    Draw a clean circular indicator and degree readout near the knee joint.
    """
    if center is None or angle_deg is None:
        return

    x, y = center
    # Outer ring
    cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)

    # Degree label tag
    deg_str = f"{int(angle_deg)} deg"
    (tw, th), _ = cv2.getTextSize(deg_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tag_x = x + radius + 8
    tag_y = y - 5

    # Tag background
    cv2.rectangle(frame, (tag_x - 4, tag_y - th - 4),
                  (tag_x + tw + 4, tag_y + 4), (20, 20, 20), -1)
    cv2.rectangle(frame, (tag_x - 4, tag_y - th - 4),
                  (tag_x + tw + 4, tag_y + 4), color, 1, cv2.LINE_AA)
    cv2.putText(frame, deg_str, (tag_x, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_minimal_controls(frame):
    """
    Subtle keyboard hint in the bottom-right corner.
    """
    h, w = frame.shape[:2]
    hint = "[Q] Quit   |   [R] Reset Reps"
    (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = w - tw - 25
    y = h - 20

    # Soft dark background
    cv2.rectangle(frame, (x - 8, y - th - 6), (x + tw + 8, y + 6),
                  (20, 20, 24), -1)
    cv2.putText(frame, hint, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
