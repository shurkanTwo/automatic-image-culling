"""Orientation helpers for PIL images and numpy arrays."""

from typing import Any

import numpy as np


def apply_pil_orientation(img: Any, orientation: int) -> Any:
    """Rotate a PIL-like image based on EXIF orientation codes."""
    if orientation == 3:
        return img.rotate(180, expand=True)
    if orientation == 6:
        return img.rotate(-90, expand=True)
    if orientation == 8:
        return img.rotate(90, expand=True)
    return img


def rotate_array(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate a numpy array based on EXIF orientation codes."""
    if orientation == 3:
        return np.rot90(arr, 2)
    if orientation == 6:
        return np.rot90(arr, -1)
    if orientation == 8:
        return np.rot90(arr, 1)
    return arr
