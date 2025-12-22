"""Image metric helpers used by the analyzer pipeline."""

from functools import lru_cache
import pathlib
from typing import Any, Dict, Optional, TypedDict

import numpy as np


class BrightnessStats(TypedDict):
    """Summarized brightness statistics for an image."""

    mean: float
    shadows: float
    highlights: float


class StructureTensorStats(TypedDict):
    """Eigenvalues and ratio derived from the structure tensor."""

    lambda_max: float
    lambda_min: float
    ratio: float


def variance_of_laplacian(arr: np.ndarray) -> float:
    """Return a simple focus measure using a Laplacian kernel."""
    core = -4 * arr[1:-1, 1:-1]
    core += arr[:-2, 1:-1]
    core += arr[2:, 1:-1]
    core += arr[1:-1, :-2]
    core += arr[1:-1, 2:]
    return float(core.var())


def tenengrad(arr: np.ndarray) -> float:
    """Compute the Tenengrad focus measure using Sobel operators."""
    padded = np.pad(arr, 1, mode="reflect")
    gx = (
        padded[0:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - (padded[0:-2, 0:-2] + 2 * padded[1:-1, 0:-2] + padded[2:, 0:-2])
    )
    gy = (
        padded[2:, 0:-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    )
    return float((gx * gx + gy * gy).mean())


def structure_tensor_ratio(arr: np.ndarray) -> StructureTensorStats:
    """Return structure tensor eigenvalues and their ratio for motion estimation."""
    gx, gy = np.gradient(arr)
    gxx = float((gx * gx).mean())
    gyy = float((gy * gy).mean())
    gxy = float((gx * gy).mean())
    trace = gxx + gyy
    tmp = ((gxx - gyy) ** 2 + 4 * (gxy**2)) ** 0.5
    lam1 = 0.5 * (trace + tmp)
    lam2 = 0.5 * (trace - tmp)
    ratio = lam2 / lam1 if lam1 > 1e-9 else 0.0
    return {"lambda_max": lam1, "lambda_min": lam2, "ratio": ratio}


def noise_estimate(arr: np.ndarray, sample_step: int = 2) -> float:
    """Estimate noise via residual variance after a simple box blur.

    Downsamples by ``sample_step`` to reduce work; set to 1 to use full resolution.
    """
    if sample_step > 1:
        arr = arr[::sample_step, ::sample_step]
    if arr.size == 0:
        return 0.0
    padded = np.pad(arr, 1, mode="reflect")
    blur = (
        padded[:-2, :-2]
        + padded[1:-1, :-2]
        + padded[2:, :-2]
        + padded[:-2, 1:-1]
        + padded[1:-1, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, 2:]
        + padded[2:, 2:]
    ) / 9.0
    residual = arr - blur
    gx, gy = np.gradient(arr)
    grad_mag = np.hypot(gx, gy)
    flat_mask = grad_mag < np.percentile(grad_mag, 30)
    if flat_mask.any():
        return float(residual[flat_mask].std())
    return float(residual.std())


def brightness_stats(arr: np.ndarray) -> BrightnessStats:
    """Summarize brightness, shadows, and highlights for an array."""
    norm = arr / 255.0
    shadow_cut = 0.2
    highlight_cut = 0.7
    return {
        "mean": float(norm.mean()),
        "shadows": float((norm < shadow_cut).mean()),
        "highlights": float((norm > highlight_cut).mean()),
    }


def composition_score(arr: np.ndarray) -> float:
    """Score how close the weighted center is to rule-of-thirds intersections."""
    h, w = arr.shape
    weight = arr.astype(np.float64) + 1e-6
    total = weight.sum()
    if total <= 0.0:
        return 0.0
    x_weight = weight.sum(axis=0)
    y_weight = weight.sum(axis=1)
    x_coords = np.arange(w, dtype=np.float64)
    y_coords = np.arange(h, dtype=np.float64)
    cx = float((x_weight * x_coords).sum() / (total * w))
    cy = float((y_weight * y_coords).sum() / (total * h))
    thirds = np.array([1 / 3, 2 / 3])
    dx = float(np.min(np.abs(thirds - cx)))
    dy = float(np.min(np.abs(thirds - cy)))
    score = 1.0 - min(1.0, (dx + dy))
    return score


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    """Return an orthonormal DCT-II transform matrix of shape (size, size)."""
    if size <= 0:
        raise ValueError("DCT size must be positive.")
    idx = np.arange(size, dtype=np.float32)
    matrix = np.empty((size, size), dtype=np.float32)
    factor = np.pi / float(size)
    for k in range(size):
        scale = np.sqrt(1.0 / size) if k == 0 else np.sqrt(2.0 / size)
        matrix[k, :] = scale * np.cos((idx + 0.5) * k * factor)
    return matrix


def _downsample_mean(gray: np.ndarray, size: int) -> Optional[np.ndarray]:
    """Downsample a 2D array to a square using block averaging."""
    if size < 1 or gray.ndim != 2:
        return None
    h, w = gray.shape
    if h < 1 or w < 1:
        return None
    if h == size and w == size:
        return gray.astype(np.float32)
    y_edges = np.linspace(0, h, num=size + 1, dtype=int)
    x_edges = np.linspace(0, w, num=size + 1, dtype=int)
    block_sums = np.add.reduceat(
        np.add.reduceat(gray, y_edges[:-1], axis=0), x_edges[:-1], axis=1
    )
    heights = np.diff(y_edges).astype(np.float32)
    widths = np.diff(x_edges).astype(np.float32)
    area = np.maximum(heights[:, None] * widths[None, :], 1.0)
    return (block_sums / area).astype(np.float32)


def phash(
    preview_path: pathlib.Path,
    image_module: Optional[Any] = None,
    image: Optional[Any] = None,
    gray_array: Optional[np.ndarray] = None,
) -> Optional[int]:
    """Compute a DCT-based perceptual hash over a 32x32 luminance thumbnail.

    Accepts either a pre-opened PIL image via ``image`` or a precomputed grayscale
    array via ``gray_array``. Falls back to opening from ``preview_path`` when
    needed.
    """
    hash_size = 8
    dct_size = 32
    small: Optional[np.ndarray] = None
    if gray_array is not None:
        gray = np.asarray(gray_array, dtype=np.float32)
        if gray.ndim != 2 or gray.size == 0:
            return None
        small = _downsample_mean(gray, dct_size)
        if small is None or small.size == 0:
            return None
    else:
        if image_module is None:
            return None
        if image is None:
            with image_module.open(preview_path) as img_handle:
                img = img_handle.convert("L").resize(
                    (dct_size, dct_size), image_module.LANCZOS
                )
                small = np.array(img, dtype=np.float32)
        else:
            img = image.convert("L").resize((dct_size, dct_size), image_module.LANCZOS)
            small = np.array(img, dtype=np.float32)
    if small is None or small.size == 0:
        return None
    dct_mat = _dct_matrix(dct_size)
    coeffs = dct_mat @ small @ dct_mat.T
    top_left = coeffs[:hash_size, :hash_size]
    flat = top_left.ravel()
    if flat.size == 0:
        return None
    median = (
        float(np.median(flat[1:])) if flat.size > 1 else float(flat[0])
    )
    bits = 0
    for i, coeff in enumerate(flat):
        if coeff > median:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    """Return the Hamming distance between two hash integers."""
    return (a ^ b).bit_count()
