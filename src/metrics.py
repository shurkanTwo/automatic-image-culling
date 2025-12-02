import numpy as np
from typing import Dict, Optional


def variance_of_laplacian(arr: np.ndarray) -> float:
    core = -4 * arr[1:-1, 1:-1]
    core += arr[:-2, 1:-1]
    core += arr[2:, 1:-1]
    core += arr[1:-1, :-2]
    core += arr[1:-1, 2:]
    return float(core.var())


def tenengrad(arr: np.ndarray) -> float:
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
    norm = arr / 255.0
    return {
        "mean": float(norm.mean()),
        "shadows": float((norm < 0.1).mean()),
        "highlights": float((norm > 0.9).mean()),
    }


def composition_score(arr: np.ndarray) -> float:
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
    return (a ^ b).bit_count()
