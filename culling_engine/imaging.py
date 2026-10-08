"""Read-only decoding, orientation, color conversion, and atomic previews."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageOps

from .discovery import RAW_EXTENSIONS, orientation_from_tags

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

_LOGGER = logging.getLogger(__name__)
MIN_RAW_PREVIEW_EDGE = 640

ORIENTATION_TRANSFORMS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def apply_orientation(image: Image.Image, orientation: int | None) -> Image.Image:
    operation = ORIENTATION_TRANSFORMS.get(orientation)
    return image.transpose(operation) if operation is not None else image


def resize_to_edge(image: Image.Image, edge: int) -> Image.Image:
    if max(image.size) <= edge:
        return image
    ratio = edge / max(image.size)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    # Avoid a full-resolution copy for every preview/thumbnail/analysis resize.
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def _to_srgb(image: Image.Image) -> Image.Image:
    if image.mode.startswith("I;16"):
        # Pillow's direct RGB conversion clamps 16-bit gray values above 255.
        image = image.point(lambda value: value * (255 / 65535)).convert("L")
    alpha = None
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        alpha = image.convert("RGBA").getchannel("A")
    profile = image.info.get("icc_profile")
    if profile:
        try:
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except (ValueError, OSError, ImageCms.PyCMSError):
            pass
    if alpha is not None:
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        background = Image.new("RGBA", rgba.size, (235, 235, 235, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image if image.mode == "RGB" else image.convert("RGB")


def _raw_orientation(raw: Any, tags: dict[str, Any]) -> int:
    orientation = orientation_from_tags(tags)
    if orientation is not None:
        return orientation
    # LibRaw's flip values are rotations, not EXIF orientation values.
    return {0: 1, 3: 3, 5: 8, 6: 6}.get(raw.sizes.flip, 1)


def _oriented_raw_thumbnail(
    image: Image.Image, orientation: int, raw_size: tuple[int, int]
) -> Image.Image:
    own_orientation = image.getexif().get(274)
    # Some cameras store upright pixels with no thumbnail EXIF. Its aspect
    # ratio tells us whether a quarter-turn has already been applied.
    raw_width, raw_height = raw_size
    if orientation in (5, 6, 7, 8) and raw_width != raw_height:
        raw_is_portrait = raw_height > raw_width
        thumb_is_portrait = image.height > image.width
        if raw_is_portrait != thumb_is_portrait:
            if own_orientation in (2, 3, 4):
                return ImageOps.exif_transpose(image)
            return image
    if own_orientation in range(2, 9):
        return ImageOps.exif_transpose(image)
    if own_orientation == 1 and orientation not in (5, 6, 7, 8):
        return image
    return apply_orientation(image, orientation)


def _decode_raw(
    path: Path, tags: dict[str, Any], *, full_resolution: bool
) -> Image.Image:
    # LibRaw can start its own OpenMP pool for each bounded Python worker.
    # A modest default avoids multiplying the hardware thread count by four.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError("RAW support unavailable: rawpy is not installed") from error
    # rawpy 0.25 uses LibRaw's wide-character path API on Windows. File-object
    # input would eagerly copy the entire RAW into a Python byte buffer.
    with rawpy.imread(str(path)) as raw:
        orientation = _raw_orientation(raw, tags)
        if not full_resolution:
            try:
                thumbnail = raw.extract_thumb()
                if thumbnail.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumbnail.data)) as embedded:
                        embedded.load()
                        image = _oriented_raw_thumbnail(
                            embedded, orientation, (raw.sizes.width, raw.sizes.height)
                        )
                        if max(image.size) >= MIN_RAW_PREVIEW_EDGE:
                            return _to_srgb(image)
                if thumbnail.format == rawpy.ThumbFormat.BITMAP:
                    image = Image.fromarray(thumbnail.data)
                    image = _oriented_raw_thumbnail(
                        image, orientation, (raw.sizes.width, raw.sizes.height)
                    )
                    if max(image.size) >= MIN_RAW_PREVIEW_EDGE:
                        return _to_srgb(image)
            except Exception:
                # Unsupported or damaged embedded previews should still decode
                # when the camera's sensor data can be read.
                _LOGGER.debug(
                    "Embedded RAW preview unavailable for %s", path, exc_info=True
                )
        pixels = raw.postprocess(
            half_size=not full_resolution,
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
            output_color=rawpy.ColorSpace.sRGB,
            user_flip=0,
        )
        return apply_orientation(Image.fromarray(pixels), orientation)


def decode_image(
    path: Path, tags: dict[str, Any], *, full_resolution: bool = False
) -> Image.Image:
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _decode_raw(path, tags, full_resolution=full_resolution)
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        return _to_srgb(source)


def atomic_save_jpeg(
    image: Image.Image,
    target: Path,
    *,
    quality: int = 90,
    before_replace: Callable[[], None] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".preview-", suffix=".jpg", dir=target.parent
    )
    os.close(descriptor)
    try:
        image.save(temporary, format="JPEG", quality=quality, optimize=False)
        if before_replace is not None:
            before_replace()
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
