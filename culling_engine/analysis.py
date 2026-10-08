"""Small, explainable technical estimates; no subject or aesthetic claims."""

from __future__ import annotations

from functools import lru_cache
from math import log1p
from typing import Any

import numpy as np
from PIL import Image

from .imaging import resize_to_edge


@lru_cache(maxsize=1)
def _dct_basis() -> np.ndarray:
    coordinates = np.arange(32, dtype=np.float64)
    frequencies = np.arange(8, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (coordinates + 0.5) * frequencies / 32)
    basis[0] /= np.sqrt(2)
    return basis * np.sqrt(2 / 32)


def perceptual_hash(image: Image.Image) -> str:
    sample = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    basis = _dct_basis()
    frequencies = (basis @ sample @ basis.T).ravel()
    median = np.median(frequencies[1:])
    bits = frequencies > median
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def analyze_image(image: Image.Image) -> dict[str, Any]:
    """Compare images at the same scale using float RGB luminance."""
    sample = resize_to_edge(image, 640)
    rgb = np.asarray(sample.convert("RGB"), dtype=np.float32) / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if min(luminance.shape) < 3:
        raise ValueError("Image is too small for technical analysis")
    gradient_x = np.diff(luminance, axis=1)
    gradient_y = np.diff(luminance, axis=0)
    gradient = float((np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))) / 2)
    laplacian = (
        -4 * luminance[1:-1, 1:-1]
        + luminance[:-2, 1:-1]
        + luminance[2:, 1:-1]
        + luminance[1:-1, :-2]
        + luminance[1:-1, 2:]
    )
    detail = float(np.sqrt(np.mean(laplacian**2)))
    contrast = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
    highlight_clipping = float(np.mean(np.all(rgb >= 0.99, axis=2)))
    shadow_clipping = float(np.mean(np.all(rgb <= 0.01, axis=2)))
    # Broad, deliberately unsaturated range. Sharpness on an untextured scene
    # is uncertain, so low detail is a review hint, never a discard decision.
    detail_estimate = log1p(detail * 35) / log1p(35)
    contrast_estimate = min(1.0, contrast / 0.75)
    clipping_penalty = min(20.0, highlight_clipping * 65 + shadow_clipping * 25)
    score = float(
        np.clip(
            45 + 35 * detail_estimate + 12 * contrast_estimate - clipping_penalty,
            0,
            100,
        )
    )
    hints = []
    if detail < 0.018 and gradient < 0.014:
        hints.append(
            "Low visible detail; check focus at 100% (smooth scenes may be intentional)"
        )
    if highlight_clipping > 0.015:
        hints.append(
            f"Possible clipped highlights in {highlight_clipping:.0%} of the preview"
        )
    if shadow_clipping > 0.10:
        hints.append(
            f"Possible clipped shadows in {shadow_clipping:.0%} of the preview"
        )
    if contrast < 0.16:
        hints.append("Low tonal contrast; may reflect the scene or lighting")
    signature = (
        np.asarray(sample.resize((4, 4), Image.Resampling.BOX), dtype=np.float32)
        / 255.0
    )
    return {
        "qualityScore": round(score, 1),
        "hints": hints,
        "phash": perceptual_hash(sample),
        "visualSignature": np.round(signature.ravel(), 4).tolist(),
    }
