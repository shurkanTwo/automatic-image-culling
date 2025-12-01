import io
import pathlib
from typing import Dict, Optional

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

import numpy as np


def resize_image(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= long_edge:
        return img
    if w >= h:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    else:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def generate_preview(path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict) -> Optional[pathlib.Path]:
    if not rawpy or not Image:
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    target = preview_dir / f"{path.stem}.{cfg['format']}"
    if target.exists():
        return target
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
    img = resize_image(img, cfg["long_edge"])
    img.save(target, cfg["format"].upper(), quality=cfg["quality"])
    return target


def ensure_preview(path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict) -> Optional[pathlib.Path]:
    preview = preview_dir / f"{path.stem}.{cfg['format']}"
    if preview.exists():
        return preview
    return generate_preview(path, preview_dir, cfg)


def open_preview_gray(preview_path: pathlib.Path) -> Optional[np.ndarray]:
    if not Image:
        return None
    img = Image.open(preview_path).convert("L")
    return np.array(img, dtype=np.float32)


def open_preview_rgb(preview_path: pathlib.Path, size: Optional[int] = 256) -> Optional[np.ndarray]:
    if not Image:
        return None
    img = Image.open(preview_path).convert("RGB")
    if size:
        w, h = img.size
        scale = size / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def preview_path_for(path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict) -> pathlib.Path:
    return preview_dir / f"{path.stem}.{cfg['format']}"
