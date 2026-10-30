"""
angle_calculator.py - Functions to calculate joint angles from landmark coordinates.

All angles are computed using 3-point geometry:
    Given points A, B, C  → the angle at vertex B (i.e. ∠ABC).
"""

import numpy as np


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle (in degrees) at point B formed by the vectors
    BA and BC.

    Args:
        point_a: (x, y) or (x, y, z) coordinates of point A.
        point_b: (x, y) or (x, y, z) coordinates of point B (vertex).
        point_c: (x, y) or (x, y, z) coordinates of point C.

    Returns:
        Angle in degrees (0–180).
    """
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)

    # Vectors from the vertex
    ba = a - b
    bc = c - b

    # Dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    # Clamp cosine to [-1, 1] to handle floating-point errors
    cosine_angle = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)

    # Convert radians → degrees
    angle_deg = np.degrees(np.arccos(cosine_angle))

    return angle_deg


def get_knee_angle(hip, knee, ankle):
    """
    Compute the knee angle: ∠(hip–knee–ankle).

    A fully extended leg is ~170-180°.
    A deep squat brings this down to ~60-90°.

    Args:
        hip: (x, y) coordinates of the hip landmark.
        knee: (x, y) coordinates of the knee landmark.
        ankle: (x, y) coordinates of the ankle landmark.

    Returns:
        Knee angle in degrees.
    """
    return calculate_angle(hip, knee, ankle)


def get_hip_angle(shoulder, hip, knee):
    """
    Compute the hip angle: ∠(shoulder–hip–knee).

    When standing upright this is ~170-180°.
    Leaning forward excessively reduces this angle.

    Args:
        shoulder: (x, y) coordinates of the shoulder landmark.
        hip: (x, y) coordinates of the hip landmark.
        knee: (x, y) coordinates of the knee landmark.

    Returns:
        Hip angle in degrees.
    """
    return calculate_angle(shoulder, hip, knee)


def get_torso_lean(shoulder, hip):
    """
    Calculate torso forward lean angle relative to vertical in degrees.

    0° = perfectly upright torso.
    15°–35° = standard healthy squat hinge.
    >45° = excessive forward lean (chest collapsing).

    Args:
        shoulder: (x, y) coordinates of shoulder.
        hip: (x, y) coordinates of hip.

    Returns:
        Lean angle in degrees.
    """
    dx = abs(float(shoulder[0]) - float(hip[0]))
    dy = max(float(hip[1]) - float(shoulder[1]), 1e-5)
    return float(np.degrees(np.arctan2(dx, dy)))

