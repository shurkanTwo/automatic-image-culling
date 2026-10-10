"""UTF-8 JSON Lines process protocol. Human diagnostics belong on stderr."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .discovery import ImportPlan, discover_images
from .engine import CONFIG_VERSION, DEFAULT_WORKERS, generate_detail, scan


def emit_json(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def self_test() -> dict[str, Any]:
    """Exercise real file decoding, caching, EXIF rotation, and protocol data."""
    capabilities = {}
    for dependency in ("rawpy", "pillow_heif", "exifread"):
        try:
            importlib.import_module(dependency)
            capabilities[dependency] = True
        except ImportError:
            capabilities[dependency] = False
    checks = [
        "decode",
        "orientation",
        "cache",
        "detail",
        "json",
        "folder-scope",
        "automatic-selection",
        "raw-jpeg-pairs",
    ]
    with tempfile.TemporaryDirectory(prefix="photo-select-self-test-") as temporary:
        base = Path(temporary).resolve()
        source = base / "source"
        source.mkdir()
        rgb = Image.new("RGB", (120, 80), (95, 150, 220))
        rgb.paste((240, 40, 20), (12, 8, 42, 38))
        for column in range(50, 110, 4):
            rgb.paste((30, 30, 30), (column, 50, column + 2, 70))
        rgb.save(source / "synthetic.png")
        exif = Image.Exif()
        exif[274] = 6
        rgb.save(source / "rotated.jpg", exif=exif)
        Image.new("RGB", (120, 80), (0, 0, 0)).save(source / "blank.png")
        expected_count = 3
        if capabilities["pillow_heif"]:
            rgb.save(source / "phone.heic", format="HEIF")
            expected_count += 1
            checks.append("heif")
        nested = source / "nested"
        nested.mkdir()
        rgb.save(nested / "nested.png")
        records: list[dict[str, Any]] = []
        summary = scan(
            source, base / "cache", records.append, workers=1, include_subfolders=False
        )
        if summary != {
            "type": "complete",
            "processed": expected_count,
            "total": expected_count,
            "failed": 0,
        }:
            raise RuntimeError(f"Synthetic scan failed: {summary}")
        photos = [record["photo"] for record in records if record["type"] == "photo"]
        suggestions = next(
            record["suggestions"]
            for record in records
            if record["type"] == "suggestions"
        )
        black = next(photo for photo in photos if photo["filename"] == "blank.png")
        if (
            len(suggestions) != len(photos)
            or not any(
                suggestion["decision"] == "favorite" for suggestion in suggestions
            )
            or not any(
                suggestion["photoId"] == black["id"]
                and suggestion["decision"] == "pass"
                for suggestion in suggestions
            )
            or [record["type"] for record in records][-3:]
            != ["groups", "suggestions", "complete"]
        ):
            raise RuntimeError("Automatic selection check failed")
        if any(Path(photo["path"]).parent != source for photo in photos):
            raise RuntimeError("Selected-folder scope check failed")
        rotated = next(photo for photo in photos if photo["filename"] == "rotated.jpg")
        if (rotated["width"], rotated["height"]) != (80, 120):
            raise RuntimeError("EXIF orientation check failed")
        for photo in photos:
            with Image.open(photo["previewPath"]) as preview:
                preview.load()
            json.dumps(photo, allow_nan=False)
        cache_times = {
            photo["previewPath"]: Path(photo["previewPath"]).stat().st_mtime_ns
            for photo in photos
        }
        recursive_records: list[dict[str, Any]] = []
        recursive_summary = scan(
            source, base / "cache", recursive_records.append, workers=1
        )
        recursive_paths = {
            record["photo"]["path"]
            for record in recursive_records
            if record["type"] == "photo"
        }
        if recursive_summary != {
            "type": "complete",
            "processed": expected_count + 1,
            "total": expected_count + 1,
            "failed": 0,
        } or recursive_paths != {photo["path"] for photo in photos} | {
            str(nested / "nested.png")
        }:
            raise RuntimeError("Include-subfolders scope check failed")
        if any(
            Path(path).stat().st_mtime_ns != value
            for path, value in cache_times.items()
        ):
            raise RuntimeError("Cache reuse check failed")
        detail = generate_detail(source / "rotated.jpg", base / "detail.jpg")
        if (detail["width"], detail["height"]) != (80, 120):
            raise RuntimeError("Full-resolution detail check failed")
        paired_source = base / "pairs"
        paired_source.mkdir()
        raw = paired_source / "capture.NEF"
        raw.write_bytes(b"Synthetic invalid RAW for JPEG fallback check")
        jpeg = paired_source / "CAPTURE.JPG"
        rgb.save(jpeg)
        pair_paths = discover_images(paired_source, base / "pair-cache")
        pair_plan = ImportPlan(pair_paths, prefer_raw=True)
        if (
            pair_plan.initial_paths != (raw,)
            or pair_plan.record_result(raw, successful=True)
            or pair_plan.excluded_paths != (jpeg,)
        ):
            raise RuntimeError("RAW/JPEG pairing plan check failed")
        fallback_records: list[dict[str, Any]] = []
        fallback_summary = scan(
            paired_source,
            base / "pair-cache",
            fallback_records.append,
            workers=1,
            prefer_raw=True,
        )
        fallback_photos = [
            record["photo"] for record in fallback_records if record["type"] == "photo"
        ]
        if (
            fallback_summary
            != {"type": "complete", "processed": 2, "total": 2, "failed": 1}
            or {photo["path"] for photo in fallback_photos} != {str(raw), str(jpeg)}
            or next(
                record["paths"]
                for record in fallback_records
                if record["type"] == "excluded"
            )
            or not any(
                photo["path"] == str(raw) and photo["analysisError"]
                for photo in fallback_photos
            )
            or not any(
                photo["path"] == str(jpeg) and not photo["analysisError"]
                for photo in fallback_photos
            )
        ):
            raise RuntimeError("RAW/JPEG decoding fallback check failed")
    return {
        "type": "self-test",
        "success": True,
        "version": __version__,
        "checks": checks,
        "capabilities": capabilities,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Photo Select image preview and technical analysis engine"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=json.dumps(
            {"type": "version", "version": __version__, "configVersion": CONFIG_VERSION}
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser(
        "scan", help="Analyze originals without changing them"
    )
    scan_parser.add_argument("--source", required=True, type=Path)
    scan_parser.add_argument("--cache", required=True, type=Path)
    scan_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    scan_parser.add_argument(
        "--prefer-raw",
        action="store_true",
        help="Skip same-folder JPEG companions only when matching RAW analysis succeeds",
    )
    scan_parser.add_argument(
        "--selection-mode",
        choices=("cautious", "stronger"),
        default="cautious",
        help="Assertiveness of reversible technical first-pass proposals",
    )
    scan_parser.add_argument(
        "--no-subfolders",
        action="store_true",
        help="Analyze only images directly in the source folder",
    )
    detail_parser = commands.add_parser(
        "detail", help="Generate a full-resolution oriented JPEG"
    )
    detail_parser.add_argument("--source", required=True, type=Path)
    detail_parser.add_argument("--output", required=True, type=Path)
    commands.add_parser(
        "self-test", help="Validate bundled decoding and cache without user images"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "scan":
            scan(
                arguments.source,
                arguments.cache,
                emit_json,
                workers=arguments.workers,
                include_subfolders=not arguments.no_subfolders,
                selection_mode=arguments.selection_mode,
                prefer_raw=arguments.prefer_raw,
            )
        elif arguments.command == "detail":
            emit_json(generate_detail(arguments.source, arguments.output))
        else:
            emit_json(self_test())
        return 0
    except KeyboardInterrupt:
        print("Photo Select engine interrupted", file=sys.stderr, flush=True)
        return 130
    except BrokenPipeError:
        return 1
    except Exception as error:  # noqa: BLE001 - top-level JSON protocol error boundary
        message = f"{type(error).__name__}: {error}"
        emit_json({"type": "error", "message": message})
        print(message, file=sys.stderr, flush=True)
        return 1
