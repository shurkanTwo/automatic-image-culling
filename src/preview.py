"""Preview generation and loading utilities."""

import io
import pathlib
from typing import Any, Dict, Optional

import numpy as np

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from .discovery import exif_orientation, read_exif


def resize_image(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize an image while keeping aspect ratio and limiting the longest edge."""
    width, height = img.size
    if max(width, height) <= long_edge:
        return img
    if width >= height:
        new_width = long_edge
        new_height = int(height * (long_edge / width))
    else:
        new_height = long_edge
        new_width = int(width * (long_edge / height))
    return img.resize((new_width, new_height), Image.LANCZOS)


def _apply_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Rotate the image based on EXIF orientation codes."""
    if orientation == 3:
        return img.rotate(180, expand=True)
    if orientation == 6:
        return img.rotate(-90, expand=True)
    if orientation == 8:
        return img.rotate(90, expand=True)
    return img


def _preview_format(cfg: Dict[str, Any]) -> str:
    """Return the desired preview format with a safe default."""
    return cfg.get("format", "webp")


def generate_preview(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict[str, Any]
) -> Optional[pathlib.Path]:
    """Write a resized preview for a RAW file if missing and return its path."""
    if not rawpy or not Image:
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    target = preview_path_for(path, preview_dir, cfg)
    if target.exists():
        return target
    orientation = exif_orientation(read_exif(path))
    long_edge = int(cfg.get("long_edge", 2048))
    fmt = _preview_format(cfg)
    quality = int(cfg.get("quality", 85))
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
            else:
                raise RuntimeError("Non-JPEG thumbnail; falling back to demosaic")
        except Exception:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            img = Image.fromarray(rgb)
    img = _apply_orientation(img, orientation)
    img = resize_image(img, long_edge)
    img.save(target, fmt.upper(), quality=quality)
    return target


def ensure_preview(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict[str, Any]
) -> Optional[pathlib.Path]:
    """Guarantee that a preview file exists for the given RAW file."""
    preview = preview_path_for(path, preview_dir, cfg)
    if preview.exists():
        return preview
    return generate_preview(path, preview_dir, cfg)


def open_preview_gray(preview_path: pathlib.Path) -> Optional[np.ndarray]:
    """Open a preview image as a grayscale float array."""
    if not Image:
        return None
    img = Image.open(preview_path).convert("L")
    return np.array(img, dtype=np.float32)


def open_preview_rgb(
    preview_path: pathlib.Path, size: Optional[int] = 256
) -> Optional[np.ndarray]:
    """Open a preview image as RGB and optionally downscale to a target size."""
    if not Image:
        return None
    img = Image.open(preview_path).convert("RGB")
    if size:
        width, height = img.size
        scale = size / float(max(width, height))
        if scale < 1.0:
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def preview_path_for(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict[str, Any]
) -> pathlib.Path:
    """Return the expected preview path for a RAW file and preview config."""
    return preview_dir / f"{path.stem}.{_preview_format(cfg)}"
