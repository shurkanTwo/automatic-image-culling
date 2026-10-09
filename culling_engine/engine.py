"""Bounded analysis pipeline and stat/version-addressed disk cache."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image

from .analysis import analyze_image
from .discovery import (
    SourceIdentity,
    camera_name,
    capture_time,
    discover_images,
    display_path,
    read_metadata,
    source_identity,
)
from .grouping import group_photos
from .imaging import atomic_save_jpeg, decode_image, resize_to_edge

CONFIG_VERSION = "photo-select-preview-2048-thumb-360-analysis-640-v3"
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
PREVIEW_EDGE = 2048
THUMBNAIL_EDGE = 360
Emitter = Callable[[dict[str, Any]], None]


def _atomic_json(value: dict[str, Any], target: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=".record-", suffix=".json", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _valid_cached_metadata(photo: dict[str, Any], identity: SourceIdentity) -> bool:
    required = {
        "filename",
        "captureTime",
        "width",
        "height",
        "camera",
        "hints",
        "analysisError",
    }
    if not required.issubset(photo) or photo["analysisError"] is not None:
        return False
    if photo["filename"] != identity.path.name:
        return False
    if not isinstance(photo["captureTime"], str):
        return False
    datetime.fromisoformat(photo["captureTime"].replace("Z", "+00:00"))
    if not all(
        type(photo[key]) is int and 0 < photo[key] <= 0xFFFFFFFF
        for key in ("width", "height")
    ):
        return False
    if photo["camera"] is not None and not isinstance(photo["camera"], str):
        return False
    return isinstance(photo["hints"], list) and all(
        isinstance(value, str) for value in photo["hints"]
    )


def _load_cache(record: Path, identity: SourceIdentity) -> dict[str, Any] | None:
    try:
        with record.open(encoding="utf-8") as stream:
            photo = json.load(stream)
        if (
            not isinstance(photo, dict)
            or photo["id"] != identity.photo_id
            or photo["path"] != str(identity.path)
        ):
            return None
        if not _valid_cached_metadata(photo, identity):
            return None
        signature = photo.get("visualSignature")
        if not isinstance(signature, list) or len(signature) != 48:
            return None
        if not all(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
            for value in signature
        ):
            return None
        if not isinstance(photo.get("phash"), str) or len(photo["phash"]) != 16:
            return None
        int(photo["phash"], 16)
        score = photo.get("qualityScore")
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 100
        ):
            return None
        if photo["previewPath"] != str(record.parent / "preview.jpg"):
            return None
        if photo["thumbnailPath"] != str(record.parent / "thumbnail.jpg"):
            return None
        for key in ("previewPath", "thumbnailPath"):
            with Image.open(photo[key]) as image:
                # JPEG verify() only checks its header. Decode catches damaged
                # scan data that would otherwise show as a broken app preview.
                image.load()
        return photo
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        Image.DecompressionBombError,
    ):
        return None


def _check_source_unchanged(identity: SourceIdentity) -> None:
    stat = identity.path.stat()
    changed = (
        identity.path.resolve(strict=True) != identity.path
        or stat.st_size != identity.size
        or stat.st_mtime_ns != identity.mtime_ns
        or stat.st_dev != identity.device
        or stat.st_ino != identity.inode
    )
    if changed:
        raise RuntimeError(
            "Original changed while it was being analyzed; rescan the folder"
        )


def _base_photo(identity: SourceIdentity, tags: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identity.photo_id,
        "path": str(identity.path),
        "filename": identity.path.name,
        "previewPath": "",
        "thumbnailPath": "",
        "captureTime": capture_time(tags, identity.path),
        "width": 0,
        "height": 0,
        "camera": camera_name(tags),
        "qualityScore": 0.0,
        "hints": [],
        "phash": None,
        "analysisError": None,
    }


def analyze_photo(path: Path, cache: Path) -> dict[str, Any]:
    """One isolated unit of work. Never write to the source path."""
    identity = source_identity(path, config_version=CONFIG_VERSION)
    cache = cache.resolve()
    directory = cache / identity.cache_key[:2] / identity.cache_key
    record = directory / "analysis.json"
    if identity.path.is_relative_to(cache):
        raise ValueError("Preview cache must not contain the original file")
    if not directory.resolve().is_relative_to(cache):
        raise ValueError("Preview cache directory resolves outside the cache")
    cached = _load_cache(record, identity)
    if cached is not None:
        _check_source_unchanged(identity)
        return cached
    tags = read_metadata(identity.path)
    photo = _base_photo(identity, tags)
    try:
        if identity.path.is_relative_to(cache):
            raise ValueError("Preview cache must not contain the original file")
        if not directory.resolve().is_relative_to(cache):
            raise ValueError("Preview cache directory resolves outside the cache")
        image = decode_image(identity.path, tags)
        photo["width"], photo["height"] = image.size
        photo.update(analyze_image(image))
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(cache):
            raise ValueError("Preview cache directory resolves outside the cache")
        preview = directory / "preview.jpg"
        thumbnail = directory / "thumbnail.jpg"
        check_source = partial(_check_source_unchanged, identity)
        atomic_save_jpeg(
            resize_to_edge(image, PREVIEW_EDGE), preview, before_replace=check_source
        )
        atomic_save_jpeg(
            resize_to_edge(image, THUMBNAIL_EDGE),
            thumbnail,
            quality=85,
            before_replace=check_source,
        )
        _check_source_unchanged(identity)
        photo["previewPath"] = str(preview)
        photo["thumbnailPath"] = str(thumbnail)
        _atomic_json(photo, record)
        _check_source_unchanged(identity)
    except Exception as error:  # noqa: BLE001 - per-file decoder failure isolation
        photo["analysisError"] = f"{type(error).__name__}: {error}"
        photo["hints"] = ["Preview could not be analyzed; original remains available"]
        photo["qualityScore"] = 0.0
        photo["previewPath"] = ""
        photo["thumbnailPath"] = ""
        photo["phash"] = None
        photo.pop("visualSignature", None)
    return photo


def _public_photo(photo: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "path",
        "filename",
        "previewPath",
        "thumbnailPath",
        "captureTime",
        "width",
        "height",
        "camera",
        "qualityScore",
        "hints",
        "phash",
        "analysisError",
    )
    return {key: photo[key] for key in fields}


def scan(
    source: Path,
    cache: Path,
    emit: Emitter,
    *,
    workers: int = DEFAULT_WORKERS,
    include_subfolders: bool = True,
) -> dict[str, Any]:
    """Bound pending futures so a batch does not allocate thousands of decoders."""
    if not 1 <= workers <= 8:
        raise ValueError("Worker count must be between 1 and 8")
    source = source.resolve(strict=True)
    cache = cache.resolve()
    str(cache).encode("utf-8")
    paths = discover_images(source, cache, include_subfolders=include_subfolders)
    cache.mkdir(parents=True, exist_ok=True)
    emit({"type": "scan", "total": len(paths)})
    results: list[dict[str, Any]] = []
    failed = 0
    processed = 0
    pending: dict[Future[dict[str, Any]], Path] = {}
    iterator = iter(paths)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="photo-select"
    ) as pool:

        def submit_next() -> bool:
            path = next(iterator, None)
            if path is None:
                return False
            pending[pool.submit(analyze_photo, path, cache)] = path
            return True

        for _ in range(workers * 2):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                path = pending.pop(future)
                try:
                    photo = future.result()
                # Worker errors are isolated; protocol writes remain fatal.
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    emit(
                        {
                            "type": "error",
                            "message": f"{type(error).__name__}: {error}",
                            "path": display_path(path),
                        }
                    )
                else:
                    results.append(photo)
                    failed += int(photo["analysisError"] is not None)
                    emit({"type": "photo", "photo": _public_photo(photo)})
                processed += 1
                emit(
                    {
                        "type": "progress",
                        "processed": processed,
                        "total": len(paths),
                        "currentFile": display_path(path),
                        "failed": failed,
                    }
                )
                submit_next()
    emit({"type": "groups", "groups": group_photos(results)})
    complete = {
        "type": "complete",
        "processed": processed,
        "total": len(paths),
        "failed": failed,
    }
    emit(complete)
    return complete


def generate_detail(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    str(output).encode("utf-8")
    if source == output or (output.exists() and output.samefile(source)):
        raise ValueError("Detail output must differ from the original file")
    if output.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("Detail output must use .jpg or .jpeg")
    identity = source_identity(source, config_version=CONFIG_VERSION)
    marker = output.with_suffix(output.suffix + ".photo-select-detail.json")
    if output.exists():
        try:
            with marker.open(encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Refusing to overwrite an existing image without a Photo Select detail marker"
            ) from error
        if (
            not isinstance(previous, dict)
            or previous.get("sourcePhotoId") != identity.photo_id
        ):
            raise ValueError("Existing detail output belongs to another source")
        if previous.get("sourceCacheKey") == identity.cache_key:
            try:
                with Image.open(output) as cached:
                    cached.load()
                    _check_source_unchanged(identity)
                    return {
                        "type": "detail",
                        "detailPath": str(output),
                        "width": cached.width,
                        "height": cached.height,
                    }
            except (OSError, ValueError):
                # The marker proves ownership; damaged generated details can be rebuilt.
                output.unlink(missing_ok=True)
    image = decode_image(source, read_metadata(source), full_resolution=True)
    _check_source_unchanged(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        # New output needs an ownership marker for interruption recovery, but
        # its identity must not claim valid pixels before encoding succeeds.
        _atomic_json(
            {"sourcePhotoId": identity.photo_id, "sourceCacheKey": None}, marker
        )
    atomic_save_jpeg(
        image,
        output,
        quality=95,
        before_replace=lambda: _check_source_unchanged(identity),
    )
    _check_source_unchanged(identity)
    # Keep an existing detail's old identity until its replacement succeeds.
    # Otherwise an encoding failure could label old pixels as the new source.
    _atomic_json(
        {"sourcePhotoId": identity.photo_id, "sourceCacheKey": identity.cache_key},
        marker,
    )
    _check_source_unchanged(identity)
    return {
        "type": "detail",
        "detailPath": str(output),
        "width": image.width,
        "height": image.height,
    }
