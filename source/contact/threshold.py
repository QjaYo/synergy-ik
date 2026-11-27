"""Binary ground contact from world-space positions in meters and elapsed seconds."""

import numpy as np


def detect_ground_contact(
    current: np.ndarray,
    previous: np.ndarray,
    dt: float,
    height_threshold: float,
    speed_threshold: float,
    ground_y: float = 0.0,
) -> np.ndarray:
    """Return (..., sites) bool values; both comparisons are strictly below threshold."""
    if dt <= 0:
        raise ValueError("Contact requires positive elapsed physical time")
    speed = np.linalg.norm(current - previous, axis=-1) / dt
    return ((current[..., 1] - ground_y < height_threshold)
            & (speed < speed_threshold))
