"""Bounded analysis pipeline and stat/version-addressed disk cache."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from PIL import Image

from .analysis import analyze_image
from .discovery import (
    SourceIdentity,
    camera_name,
    capture_time,
    discover_images,
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


def _load_cache(record: Path, identity: SourceIdentity) -> dict[str, Any] | None:
    try:
        with record.open(encoding="utf-8") as stream:
            photo = json.load(stream)
        if photo["id"] != identity.photo_id or photo["path"] != str(identity.path):
            return None
        signature = photo.get("visualSignature")
        if not isinstance(signature, list) or len(signature) != 48:
            return None
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in signature
        ):
            return None
        if not isinstance(photo.get("phash"), str) or len(photo["phash"]) != 16:
            return None
        int(photo["phash"], 16)
        score = photo.get("qualityScore")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 100
        ):
            return None
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
            return None
        if not isinstance(photo["hints"], list) or not all(
            isinstance(value, str) for value in photo["hints"]
        ):
            return None
        if photo["previewPath"] != str(record.parent / "preview.jpg"):
            return None
        if photo["thumbnailPath"] != str(record.parent / "thumbnail.jpg"):
            return None
        for key in ("previewPath", "thumbnailPath"):
            with Image.open(photo[key]) as image:
                image.verify()
        return photo
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return None


def _check_source_unchanged(identity: SourceIdentity) -> None:
    stat = identity.path.stat()
    if stat.st_size != identity.size or stat.st_mtime_ns != identity.mtime_ns:
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
    directory = cache / identity.cache_key[:2] / identity.cache_key
    record = directory / "analysis.json"
    cached = _load_cache(record, identity)
    if cached is not None:
        _check_source_unchanged(identity)
        return cached
    tags = read_metadata(identity.path)
    photo = _base_photo(identity, tags)
    try:
        image = decode_image(identity.path, tags)
        photo["width"], photo["height"] = image.size
        photo.update(analyze_image(image))
        directory.mkdir(parents=True, exist_ok=True)
        preview = directory / "preview.jpg"
        thumbnail = directory / "thumbnail.jpg"
        atomic_save_jpeg(resize_to_edge(image, PREVIEW_EDGE), preview)
        atomic_save_jpeg(resize_to_edge(image, THUMBNAIL_EDGE), thumbnail, quality=85)
        _check_source_unchanged(identity)
        photo["previewPath"] = str(preview)
        photo["thumbnailPath"] = str(thumbnail)
        _atomic_json(photo, record)
    except Exception as error:  # noqa: BLE001 - per-file decoder failure isolation
        photo["analysisError"] = f"{type(error).__name__}: {error}"
        photo["hints"] = ["Preview could not be analyzed; original remains available"]
        photo["qualityScore"] = 0.0
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
    source: Path, cache: Path, emit: Emitter, *, workers: int = DEFAULT_WORKERS
) -> dict[str, Any]:
    """Bound pending futures so a batch does not allocate thousands of decoders."""
    if not 1 <= workers <= 8:
        raise ValueError("Worker count must be between 1 and 8")
    source = source.resolve(strict=True)
    cache = cache.resolve()
    paths = discover_images(source, cache)
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
                            "path": str(path),
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
                        "currentFile": str(path),
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
    # Record ownership first, so termination between atomic writes can recover.
    _atomic_json(
        {"sourcePhotoId": identity.photo_id, "sourceCacheKey": identity.cache_key},
        marker,
    )
    atomic_save_jpeg(image, output, quality=95)
    return {
        "type": "detail",
        "detailPath": str(output),
        "width": image.width,
        "height": image.height,
    }
