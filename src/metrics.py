"""Image metric helpers used by the analyzer pipeline."""

from typing import Dict, Optional

import numpy as np

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


def structure_tensor_ratio(arr: np.ndarray) -> Dict[str, float]:
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


def noise_estimate(arr: np.ndarray) -> float:
    """Estimate noise via residual variance after a simple box blur."""
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


def brightness_stats(arr: np.ndarray) -> Dict[str, float]:
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
    y, x = np.indices(arr.shape)
    weight = arr + 1e-6
    cx = (x * weight).sum() / (weight.sum() * w)
    cy = (y * weight).sum() / (weight.sum() * h)
    thirds = np.array([1 / 3, 2 / 3])
    dx = float(np.min(np.abs(thirds - cx)))
    dy = float(np.min(np.abs(thirds - cy)))
    score = 1.0 - min(1.0, (dx + dy))
    return score


def phash(preview_path, Image=None) -> Optional[int]:
    """Compute a perceptual hash over an 8x8 luminance thumbnail."""
    if Image is None:
        return None
    img = Image.open(preview_path).convert("L").resize((8, 8), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    """Return the Hamming distance between two hash integers."""
    return (a ^ b).bit_count()
