"""Deterministic behavior tests; no real personal images or model downloads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

from culling_engine.analysis import analyze_image
from culling_engine.cli import self_test
from culling_engine.discovery import capture_time, discover_images, source_identity
from culling_engine.engine import analyze_photo, generate_detail, scan
from culling_engine.grouping import group_photos
from culling_engine.imaging import decode_image, resize_to_edge


def patterned_image(size: tuple[int, int] = (120, 80)) -> Image.Image:
    generator = np.random.default_rng(7)
    return Image.fromarray(
        generator.integers(25, 230, (size[1], size[0], 3), dtype=np.uint8)
    )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Windows temp paths can use an 8.3 alias; compare canonical originals.
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "originals"
        self.cache = self.root / "cache"
        self.source.mkdir()

    def save_photo(self, name: str, *, orientation: int = 1) -> Path:
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif[274] = orientation
        patterned_image().save(path, exif=exif)
        return path

    def test_recursive_duplicate_filenames_keep_separate_cache_and_identity(
        self,
    ) -> None:
        first = self.save_photo("day1/photo.jpg")
        second = self.save_photo("day2/photo.jpg")
        records = []
        result = scan(self.source, self.cache, records.append, workers=2)
        photos = [record["photo"] for record in records if record["type"] == "photo"]
        self.assertEqual(result["failed"], 0)
        self.assertEqual(len({photo["id"] for photo in photos}), 2)
        self.assertEqual(len({photo["previewPath"] for photo in photos}), 2)
        self.assertEqual({photo["path"] for photo in photos}, {str(first), str(second)})
        for photo in photos:
            self.assertEqual(len(photo["id"]), 24)
            self.assertEqual(len(photo["phash"]), 16)
            self.assertNotIn("visualSignature", photo)
            self.assertNotIn("decision", photo)
            json.dumps(photo, allow_nan=False)

    def test_discovery_scope_includes_only_the_selected_folder_when_disabled(
        self,
    ) -> None:
        direct = self.save_photo("photo.jpg")
        nested = self.save_photo("day1/photo.jpg")
        deep = self.save_photo("day1/more/photo.jpg")
        self.assertEqual(
            discover_images(self.source, self.cache, include_subfolders=False),
            [direct],
        )
        self.assertEqual(
            set(discover_images(self.source, self.cache, include_subfolders=True)),
            {direct, nested, deep},
        )
        self.assertEqual(
            discover_images(self.source, self.cache),
            discover_images(self.source, self.cache, include_subfolders=True),
        )

    def test_scan_scope_filters_nested_images_and_keeps_root_photo_identity(
        self,
    ) -> None:
        direct = self.save_photo("photo.jpg")
        nested = self.save_photo("day1/photo.jpg")
        photos_by_scope = {}
        for include_subfolders, expected in (
            (False, {direct}),
            (True, {direct, nested}),
        ):
            with self.subTest(include_subfolders=include_subfolders):
                records = []
                result = scan(
                    self.source,
                    self.cache,
                    records.append,
                    workers=1,
                    include_subfolders=include_subfolders,
                )
                photos = [
                    record["photo"] for record in records if record["type"] == "photo"
                ]
                self.assertEqual(
                    result,
                    {
                        "type": "complete",
                        "processed": len(expected),
                        "total": len(expected),
                        "failed": 0,
                    },
                )
                self.assertEqual(
                    {photo["path"] for photo in photos},
                    {str(path) for path in expected},
                )
                photos_by_scope[include_subfolders] = next(
                    photo for photo in photos if photo["path"] == str(direct)
                )
        self.assertEqual(photos_by_scope[False], photos_by_scope[True])

    def test_selected_folder_scan_is_empty_when_images_are_only_nested(self) -> None:
        self.save_photo("day1/photo.jpg")
        records = []
        result = scan(
            self.source, self.cache, records.append, workers=1, include_subfolders=False
        )
        self.assertEqual(
            result, {"type": "complete", "processed": 0, "total": 0, "failed": 0}
        )
        self.assertEqual(
            records,
            [{"type": "scan", "total": 0}, {"type": "groups", "groups": []}, result],
        )

    def test_selected_folder_discovery_does_not_visit_inaccessible_descendants(
        self,
    ) -> None:
        direct = self.save_photo("photo.jpg")
        self.save_photo("analysis/hidden.jpg")
        self.save_photo("inaccessible/hidden.jpg")
        original_scandir = os.scandir
        original_is_file = Path.is_file

        def guarded_scandir(path):
            if Path(path) != self.source:
                raise PermissionError(f"Cannot list descendant: {path}")
            return original_scandir(path)

        def guarded_is_file(path):
            if path.parent != self.source:
                raise AssertionError(f"Unexpected descendant metadata probe: {path}")
            return original_is_file(path)

        with patch(
            "culling_engine.discovery.os.scandir", side_effect=guarded_scandir
        ), patch.object(Path, "is_file", guarded_is_file):
            self.assertEqual(
                discover_images(self.source, self.cache, include_subfolders=False),
                [direct],
            )
        with patch(
            "culling_engine.discovery.os.scandir", side_effect=guarded_scandir
        ), self.assertRaises(PermissionError):
            discover_images(self.source, self.cache, include_subfolders=True)

    def test_discovery_preserves_root_and_cache_safety_errors_in_both_scopes(
        self,
    ) -> None:
        for include_subfolders in (False, True):
            with self.subTest(include_subfolders=include_subfolders):
                with self.assertRaises(ValueError):
                    discover_images(
                        self.source, self.source, include_subfolders=include_subfolders
                    )
                with patch(
                    "culling_engine.discovery.os.scandir",
                    side_effect=PermissionError("Cannot list root"),
                ), self.assertRaises(PermissionError):
                    discover_images(
                        self.source,
                        self.cache,
                        include_subfolders=include_subfolders,
                    )
                records = []
                with patch(
                    "culling_engine.discovery.os.scandir",
                    side_effect=PermissionError("Cannot list root"),
                ), self.assertRaises(PermissionError):
                    scan(
                        self.source,
                        self.cache,
                        records.append,
                        workers=1,
                        include_subfolders=include_subfolders,
                    )
                self.assertEqual(records, [])

    def test_cli_no_subfolders_filters_nested_photos_and_default_remains_recursive(
        self,
    ) -> None:
        direct = self.save_photo("photo.jpg")
        nested = self.save_photo("day1/photo.jpg")
        for flags, expected in (
            (["--no-subfolders"], {direct}),
            ([], {direct, nested}),
        ):
            with self.subTest(flags=flags):
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "culling_engine",
                        "scan",
                        "--source",
                        str(self.source),
                        "--cache",
                        str(self.cache),
                        "--workers",
                        "1",
                        *flags,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                records = [json.loads(line) for line in process.stdout.splitlines()]
                self.assertEqual(records[0], {"type": "scan", "total": len(expected)})
                self.assertEqual(
                    records[-1],
                    {
                        "type": "complete",
                        "processed": len(expected),
                        "total": len(expected),
                        "failed": 0,
                    },
                )
                self.assertEqual(
                    {
                        record["photo"]["path"]
                        for record in records
                        if record["type"] == "photo"
                    },
                    {str(path) for path in expected},
                )

    def test_cache_invalidation_by_source_size_and_mtime(self) -> None:
        path = self.save_photo("photo.jpg")
        first = analyze_photo(path, self.cache)
        preview_mtime = Path(first["previewPath"]).stat().st_mtime_ns
        repeated = analyze_photo(path, self.cache)
        self.assertEqual(first, repeated)
        self.assertEqual(
            Path(repeated["previewPath"]).stat().st_mtime_ns, preview_mtime
        )
        previous = path.stat()
        Image.new("RGB", (200, 100), (40, 80, 120)).save(path)
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
        updated = analyze_photo(path, self.cache)
        self.assertEqual(first["id"], updated["id"])
        self.assertNotEqual(first["previewPath"], updated["previewPath"])
        self.assertEqual((updated["width"], updated["height"]), (200, 100))

    def test_invalid_cached_preview_is_rebuilt(self) -> None:
        path = self.save_photo("photo.jpg")
        first = analyze_photo(path, self.cache)
        Path(first["previewPath"]).write_bytes(b"not a jpeg")
        second = analyze_photo(path, self.cache)
        self.assertIsNone(second["analysisError"])
        with Image.open(second["previewPath"]) as image:
            image.verify()

    def test_truncated_jpeg_scan_data_is_rebuilt(self) -> None:
        path = self.save_photo("photo.jpg")
        first = analyze_photo(path, self.cache)
        preview = Path(first["previewPath"])
        original = preview.read_bytes()
        preview.write_bytes(original[:-100])
        with Image.open(preview) as header:
            header.verify()  # JPEG verify does not notice missing compressed scan data.
        second = analyze_photo(path, self.cache)
        self.assertIsNone(second["analysisError"])
        with Image.open(second["previewPath"]) as decoded:
            decoded.load()

    def test_invalid_cached_photo_metadata_is_rebuilt(self) -> None:
        path = self.save_photo("photo.jpg")
        for field, value in (
            ("camera", 4),
            ("captureTime", 123),
            ("width", "120"),
            ("width", 2**40),
            ("captureTime", "not a timestamp"),
        ):
            with self.subTest(field=field, value=value):
                first = analyze_photo(path, self.cache)
                record = Path(first["previewPath"]).with_name("analysis.json")
                metadata = json.loads(record.read_text(encoding="utf-8"))
                metadata[field] = value
                record.write_text(json.dumps(metadata), encoding="utf-8")
                records = []
                summary = scan(self.source, self.cache, records.append, workers=1)
                self.assertEqual(summary["failed"], 0)
                refreshed = next(
                    item["photo"] for item in records if item["type"] == "photo"
                )
                self.assertNotEqual(refreshed[field], value)

    def test_replaced_source_invalidates_cache_even_when_size_and_time_match(
        self,
    ) -> None:
        source = self.save_photo("photo.jpg")
        old_stat = source.stat()
        first = analyze_photo(source, self.cache)
        replacement = self.source / "replacement.tmp"
        replacement.write_bytes(source.read_bytes())
        os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        replacement.replace(source)
        updated = analyze_photo(source, self.cache)
        self.assertEqual(source.stat().st_size, old_stat.st_size)
        self.assertEqual(source.stat().st_mtime_ns, old_stat.st_mtime_ns)
        self.assertEqual(updated["id"], first["id"])
        self.assertNotEqual(updated["previewPath"], first["previewPath"])

    def test_corrupt_image_is_isolated_and_counted(self) -> None:
        self.save_photo("valid.jpg")
        (self.source / "corrupt.jpg").write_bytes(b"damaged")
        records = []
        complete = scan(self.source, self.cache, records.append, workers=2)
        self.assertEqual(
            complete, {"type": "complete", "processed": 2, "total": 2, "failed": 1}
        )
        failed = [
            record["photo"]
            for record in records
            if record["type"] == "photo" and record["photo"]["analysisError"]
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["filename"], "corrupt.jpg")
        self.assertEqual(failed[0]["previewPath"], "")
        self.assertIsNone(failed[0]["phash"])
        self.assertEqual(records[-1], complete)
        self.assertEqual(records[-2]["type"], "groups")

    def test_original_content_and_mtime_unchanged_by_scan_and_detail(self) -> None:
        path = self.save_photo("photo.jpg", orientation=6)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mtime = path.stat().st_mtime_ns
        scan(self.source, self.cache, lambda record: None, workers=1)
        detail = generate_detail(path, self.cache / "detail.jpg")
        self.assertEqual((detail["width"], detail["height"]), (80, 120))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        with self.assertRaises(ValueError):
            generate_detail(path, path)

    def test_nested_cache_excluded_by_path_without_excluding_user_previews(
        self,
    ) -> None:
        self.save_photo("previews/photo.jpg")
        nested_cache = self.source / "arbitrary-name"
        nested_cache.mkdir()
        patterned_image().save(nested_cache / "cached.jpg")
        self.save_photo(".photo-select/generated.jpg")
        discovered = discover_images(self.source, nested_cache)
        self.assertEqual(
            [path.relative_to(self.source).as_posix() for path in discovered],
            ["previews/photo.jpg"],
        )
        with self.assertRaises(ValueError):
            discover_images(self.source, self.source)

    def test_detail_reuses_only_own_generated_images_and_invalidates_source_changes(
        self,
    ) -> None:
        source = self.save_photo("source.jpg")
        destination = self.cache / "detail.jpg"
        first = generate_detail(source, destination)
        mtime = destination.stat().st_mtime_ns
        self.assertEqual(first, generate_detail(source, destination))
        self.assertEqual(destination.stat().st_mtime_ns, mtime)
        Image.new("RGB", (60, 100), (100, 150, 200)).save(source)
        updated = generate_detail(source, destination)
        self.assertEqual((updated["width"], updated["height"]), (60, 100))
        other_original = self.save_photo("other.jpg")
        with self.assertRaises(ValueError):
            generate_detail(source, other_original)

    def test_generated_detail_in_source_is_not_imported_as_an_original(self) -> None:
        source = self.save_photo("source.jpg")
        generate_detail(source, self.source / "generated-detail.jpg")
        self.assertEqual(discover_images(self.source, self.cache), [source])

    def test_source_changes_during_decode_do_not_publish_previews_or_details(
        self,
    ) -> None:
        source = self.save_photo("source.jpg")

        def changing_decode(path, tags, *, full_resolution=False):
            image = decode_image(path, tags, full_resolution=full_resolution)
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return image

        with patch("culling_engine.engine.decode_image", side_effect=changing_decode):
            photo = analyze_photo(source, self.cache)
            self.assertIn("Original changed", photo["analysisError"])
            self.assertEqual(photo["previewPath"], "")
            with self.assertRaisesRegex(RuntimeError, "Original changed"):
                generate_detail(source, self.cache / "changed.jpg")
        self.assertFalse((self.cache / "changed.jpg").exists())
        self.assertEqual(list(self.cache.rglob("analysis.json")), [])

    def test_source_changes_during_detail_encoding_do_not_replace_cached_detail(
        self,
    ) -> None:
        source = self.save_photo("source.jpg")
        destination = self.cache / "detail.jpg"
        generate_detail(source, destination)
        previous_bytes = destination.read_bytes()
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        original_save = Image.Image.save

        def changing_save(image, filename, *args, **kwargs):
            original_save(image, filename, *args, **kwargs)
            current = source.stat()
            os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))

        with patch.object(Image.Image, "save", changing_save), self.assertRaisesRegex(
            RuntimeError, "Original changed"
        ):
            generate_detail(source, destination)
        self.assertEqual(destination.read_bytes(), previous_bytes)
        self.assertEqual(list(destination.parent.glob(".preview-*")), [])

    def test_source_changes_while_reading_cached_detail_are_detected(self) -> None:
        source = self.save_photo("source.jpg")
        destination = self.cache / "detail.jpg"
        generate_detail(source, destination)
        original_open = Image.open

        def changing_open(path, *args, **kwargs):
            image = original_open(path, *args, **kwargs)
            if Path(path) == destination:
                current = source.stat()
                os.utime(
                    source, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000)
                )
            return image

        with patch(
            "culling_engine.engine.Image.open", side_effect=changing_open
        ), self.assertRaisesRegex(RuntimeError, "Original changed"):
            generate_detail(source, destination)

    def test_failed_detail_encoding_does_not_label_old_pixels_as_new_source(
        self,
    ) -> None:
        source = self.save_photo("source.jpg")
        destination = self.cache / "detail.jpg"
        generate_detail(source, destination)
        Image.new("RGB", (60, 100), (100, 150, 200)).save(source)
        with patch.object(
            Image.Image, "save", side_effect=OSError("Disk full")
        ), self.assertRaises(OSError):
            generate_detail(source, destination)
        result = generate_detail(source, destination)
        self.assertEqual((result["width"], result["height"]), (60, 100))

    def test_unreadable_folder_is_not_reported_as_successful_empty_scan(self) -> None:
        def unreadable_walk(source, *, followlinks, onerror):
            onerror(PermissionError(13, "Permission denied", str(source)))
            return []

        records = []
        with patch(
            "culling_engine.discovery.os.walk", side_effect=unreadable_walk
        ), self.assertRaises(PermissionError):
            scan(self.source, self.cache, records.append, workers=1)
        self.assertFalse(any(record["type"] == "complete" for record in records))

    def test_missing_file_during_scan_is_isolated(self) -> None:
        present = self.save_photo("present.jpg")
        missing = self.source / "missing.jpg"
        records = []
        with patch(
            "culling_engine.engine.discover_images", return_value=[missing, present]
        ):
            summary = scan(self.source, self.cache, records.append, workers=1)
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertTrue(
            any(
                record["type"] == "error" and record["path"] == str(missing)
                for record in records
            )
        )
        self.assertTrue(
            any(
                record["type"] == "photo"
                and record["photo"]["filename"] == "present.jpg"
                for record in records
            )
        )

    def test_directory_links_and_cache_links_cannot_escape_owned_roots(self) -> None:
        source = self.save_photo("source.jpg")
        external = self.root / "external"
        external.mkdir()
        patterned_image().save(external / "external.jpg")
        try:
            (self.source / "link").symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        self.assertEqual(discover_images(self.source, self.cache), [source])
        photo = analyze_photo(source, self.cache)
        directory = Path(photo["previewPath"]).parent
        shutil.rmtree(directory)
        directory.symlink_to(external, target_is_directory=True)
        before = sorted(path.name for path in external.iterdir())
        records = []
        summary = scan(self.source, self.cache, records.append, workers=1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(sorted(path.name for path in external.iterdir()), before)

    def test_sixteen_bit_grayscale_tiff_keeps_tonal_range(self) -> None:
        path = self.source / "gray.tiff"
        pixels = np.tile(np.array([0, 32768, 65535], dtype=np.uint16), (10, 1))
        Image.fromarray(pixels).save(path)
        decoded = np.asarray(decode_image(path, {}))
        np.testing.assert_allclose(decoded[0, :, 0], [0, 127, 255], atol=1)

    def test_icc_color_conversion_preserves_transparent_pixels(self) -> None:
        path = self.source / "transparent.png"
        image = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        image.save(path, icc_profile=profile)
        decoded = np.asarray(decode_image(path, {}))
        np.testing.assert_array_equal(decoded[0, 0], [235, 235, 235])

    def test_missing_source_cli_reports_fatal_json_error(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "culling_engine",
                "scan",
                "--source",
                str(self.source / "missing"),
                "--cache",
                str(self.cache),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(process.returncode, 1)
        records = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "error")
        self.assertNotIn("path", records[0])
        self.assertIn("FileNotFoundError", records[0]["message"])

    @unittest.skipIf(
        os.name == "nt", "Windows paths use Unicode rather than arbitrary bytes"
    )
    def test_non_utf8_filename_is_isolated_without_breaking_json_protocol(self) -> None:
        self.save_photo("valid.jpg")
        malformed_path = os.fsencode(self.source) + b"/invalid-\xff.jpg"
        with open(malformed_path, "wb") as stream:
            patterned_image().save(stream, format="JPEG")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "culling_engine",
                "scan",
                "--source",
                str(self.source),
                "--cache",
                str(self.cache),
                "--workers",
                "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        records = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(records[-1]["failed"], 1)
        self.assertEqual(records[-1]["processed"], 2)
        error = next(record for record in records if record["type"] == "error")
        self.assertIn("cannot be represented as Unicode", error["message"])
        for record in records:
            # serde_json String rejects lone surrogates even when escaped.
            for value in record.values():
                if isinstance(value, str):
                    value.encode("utf-8", errors="strict")
        photo = next(record["photo"] for record in records if record["type"] == "photo")
        self.assertEqual(photo["filename"], "valid.jpg")

    def test_resizing_does_not_make_full_size_pixel_copies(self) -> None:
        image = patterned_image((1000, 600))
        with patch.object(
            Image.Image, "copy", side_effect=AssertionError("Unexpected full-size copy")
        ):
            resized = resize_to_edge(image, 640)
            unchanged = resize_to_edge(image, 2048)
        self.assertEqual(resized.size, (640, 384))
        self.assertIs(unchanged, image)

    def test_legacy_product_directories_excluded_only_when_marked(self) -> None:
        self.save_photo("previews/old-preview.jpg")
        self.save_photo("output/selected.jpg")
        self.save_photo("original.jpg")
        analysis = self.source / "analysis"
        analysis.mkdir()
        (analysis / "analysis.json").write_text("[]", encoding="utf-8")
        paths = discover_images(self.source, self.cache)
        self.assertEqual([path.name for path in paths], ["original.jpg"])

    def test_unicode_cli_json_lines_are_parseable(self) -> None:
        self.save_photo("友人 Straße.jpg")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "culling_engine",
                "scan",
                "--source",
                str(self.source),
                "--cache",
                str(self.cache),
                "--workers",
                "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        records = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(
            [record["type"] for record in records],
            ["scan", "photo", "progress", "groups", "complete"],
        )
        self.assertEqual(records[1]["photo"]["filename"], "友人 Straße.jpg")

    def test_cache_identity_contains_config_version(self) -> None:
        path = self.save_photo("photo.jpg")
        first = source_identity(path, config_version="v1")
        second = source_identity(path, config_version="v2")
        self.assertEqual(first.photo_id, second.photo_id)
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_all_exif_orientations_match_pixel_transforms(self) -> None:
        original = patterned_image((30, 20))
        transforms = {
            1: None,
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }
        for orientation, transform in transforms.items():
            with self.subTest(orientation=orientation):
                path = self.source / f"orientation-{orientation}.png"
                exif = Image.Exif()
                exif[274] = orientation
                original.save(path, exif=exif)
                expected = (
                    original if transform is None else original.transpose(transform)
                )
                actual = decode_image(path, {})
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_exif_timestamp_subsecond_offset_and_camera_wall_clock(self) -> None:
        path = self.save_photo("photo.jpg")
        tags = {
            "EXIF DateTimeOriginal": "2024:07:05 14:03:02",
            "EXIF SubSecTimeOriginal": "125",
            "EXIF OffsetTimeOriginal": "+02:00",
        }
        self.assertEqual(capture_time(tags, path), "2024-07-05T12:03:02.125000Z")
        tags.pop("EXIF OffsetTimeOriginal")
        self.assertEqual(capture_time(tags, path), "2024-07-05T14:03:02.125000")
        fallback = datetime.fromisoformat(capture_time({}, path).replace("Z", "+00:00"))
        self.assertEqual(fallback.tzinfo, timezone.utc)

    def test_invalid_offset_preserves_valid_camera_capture_time(self) -> None:
        path = self.save_photo("photo.jpg")
        tags = {
            "EXIF DateTimeOriginal": b"2024:07:05 14:03:02\x00",
            "EXIF OffsetTimeOriginal": "bad offset",
        }
        self.assertEqual(capture_time(tags, path), "2024-07-05T14:03:02")

    def test_mean_brightness_does_not_claim_highlight_clipping(self) -> None:
        bright = analyze_image(Image.new("RGB", (100, 100), (230, 230, 230)))
        clipped = analyze_image(Image.new("RGB", (100, 100), (255, 255, 255)))
        self.assertFalse(any("clipped highlights" in hint for hint in bright["hints"]))
        self.assertTrue(any("clipped highlights" in hint for hint in clipped["hints"]))
        self.assertLess(clipped["qualityScore"], bright["qualityScore"])

    def test_technical_score_distinguishes_sharp_detail_without_saturating(
        self,
    ) -> None:
        sharp = patterned_image((640, 480))
        smooth = Image.new("RGB", sharp.size, (140, 140, 140))
        sharp_result = analyze_image(sharp)
        smooth_result = analyze_image(smooth)
        self.assertGreater(sharp_result["qualityScore"], smooth_result["qualityScore"])
        self.assertLess(sharp_result["qualityScore"], 100)
        self.assertGreater(smooth_result["qualityScore"], 0)
        for result in (sharp_result, smooth_result):
            self.assertFalse(
                any("motion" in hint or "center" in hint for hint in result["hints"])
            )

    def test_smoke_test_uses_temporary_synthetic_images(self) -> None:
        result = self_test()
        self.assertTrue(result["success"])
        self.assertIn("folder-scope", result["checks"])


class GroupingTests(unittest.TestCase):
    def photo(
        self,
        identifier: str,
        second: int,
        *,
        hash_value: int = 0,
        score: float = 70,
        aware: bool = True,
    ) -> dict:
        return {
            "id": identifier,
            "path": f"/{identifier}.jpg",
            "captureTime": f"2024-01-01T12:00:{second:02d}" + ("Z" if aware else ""),
            "phash": f"{hash_value:016x}",
            "qualityScore": score,
            "visualSignature": [
                0.2 + position / 50 + channel / 50
                for position in range(16)
                for channel in range(3)
            ],
            "analysisError": None,
        }

    def test_groups_are_stable_chronological_and_recommend_only_members(self) -> None:
        photos = [
            self.photo("a", 2, score=71),
            self.photo("b", 1, score=75),
            self.photo("c", 3, score=74),
            self.photo("d", 30),
        ]
        groups = group_photos(photos)
        self.assertEqual(groups, group_photos(list(reversed(photos))))
        self.assertEqual(groups[0]["photoIds"], ["b", "a", "c"])
        self.assertEqual(groups[0]["recommendedPhotoIds"], ["b", "c"])
        self.assertEqual(
            sorted(identifier for group in groups for identifier in group["photoIds"]),
            ["a", "b", "c", "d"],
        )
        for group in groups:
            self.assertTrue(
                set(group["recommendedPhotoIds"]).issubset(group["photoIds"])
            )
            self.assertLessEqual(len(group["recommendedPhotoIds"]), 2)

    def test_similarity_does_not_transitively_chain(self) -> None:
        first = self.photo("a", 0, hash_value=0)
        middle = self.photo("b", 3, hash_value=(1 << 10) - 1)
        last = self.photo("c", 6, hash_value=(1 << 20) - 1)
        groups = group_photos([first, middle, last])
        self.assertEqual([group["photoIds"] for group in groups], [["a", "b"], ["c"]])

    def test_burst_span_limited_and_unknown_offset_not_mixed(self) -> None:
        groups = group_photos(
            [
                self.photo("a", 0),
                self.photo("b", 7),
                self.photo("c", 14),
                self.photo("d", 21),
            ]
        )
        self.assertEqual([len(group["photoIds"]) for group in groups], [3, 1])
        groups = group_photos([self.photo("a", 0), self.photo("b", 1, aware=False)])
        self.assertEqual(len(groups), 2)

    def test_failed_photo_is_never_recommended(self) -> None:
        photo = self.photo("bad", 0)
        photo["analysisError"] = "Unreadable"
        photo["phash"] = None
        self.assertEqual(group_photos([photo])[0]["recommendedPhotoIds"], [])

    def test_flat_frames_and_different_aspect_ratios_are_not_moments(self) -> None:
        photos = [self.photo("a", 0), self.photo("b", 1)]
        for photo in photos:
            photo["visualSignature"] = [0.4] * 48
        self.assertEqual(len(group_photos(photos)), 2)
        photos = [self.photo("a", 0), self.photo("b", 1)]
        photos[0].update(width=120, height=80)
        photos[1].update(width=80, height=120)
        self.assertEqual(len(group_photos(photos)), 2)

    def test_large_burst_groups_stay_bounded(self) -> None:
        photos = [self.photo(f"photo-{index:03d}", 0) for index in range(150)]
        groups = group_photos(photos)
        self.assertTrue(all(len(group["photoIds"]) <= 48 for group in groups))
        self.assertEqual(sum(len(group["photoIds"]) for group in groups), 150)


class RawDecodeTests(unittest.TestCase):
    def test_full_raw_detail_applies_all_exif_orientations_once(self) -> None:
        pixels = np.asarray(patterned_image((30, 20)))
        test_case = self

        class FakeRaw:
            sizes = SimpleNamespace(flip=6, width=30, height=20)

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def postprocess(self, **options):
                test_case.assertEqual(options["user_flip"], 0)
                test_case.assertFalse(options["half_size"])
                return pixels

        rawpy = SimpleNamespace(
            imread=lambda path: FakeRaw(), ColorSpace=SimpleNamespace(sRGB=1)
        )
        transforms = {
            1: None,
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }
        original = Image.fromarray(pixels)
        with patch.dict(sys.modules, {"rawpy": rawpy}):
            for orientation, transform in transforms.items():
                with self.subTest(orientation=orientation):
                    actual = decode_image(
                        Path("photo.arw"),
                        {"Image Orientation": orientation},
                        full_resolution=True,
                    )
                    expected = (
                        original if transform is None else original.transpose(transform)
                    )
                    np.testing.assert_array_equal(
                        np.asarray(actual), np.asarray(expected)
                    )

    def test_raw_fallback_requests_unrotated_half_size_camera_white_balance(
        self,
    ) -> None:
        pixels = np.zeros((80, 120, 3), dtype=np.uint8)
        calls = []

        class FakeRaw:
            sizes = SimpleNamespace(flip=6, width=120, height=80)

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def extract_thumb(self):
                raise RuntimeError("Missing preview")

            def postprocess(self, **options):
                calls.append(options)
                return pixels

        rawpy = SimpleNamespace(
            imread=lambda path: FakeRaw(), ColorSpace=SimpleNamespace(sRGB=1)
        )
        with patch.dict(sys.modules, {"rawpy": rawpy}):
            preview = decode_image(Path("photo.arw"), {})
            detail = decode_image(Path("photo.arw"), {}, full_resolution=True)
        self.assertEqual(preview.size, (80, 120))
        self.assertEqual(detail.size, (80, 120))
        self.assertTrue(calls[0]["half_size"])
        self.assertFalse(calls[1]["half_size"])
        self.assertTrue(calls[1]["use_camera_wb"])
        self.assertTrue(calls[1]["no_auto_bright"])
        self.assertEqual(calls[1]["user_flip"], 0)

    def test_already_oriented_raw_jpeg_preview_not_rotated_twice(self) -> None:
        stream = io.BytesIO()
        patterned_image((800, 1200)).save(stream, format="JPEG")

        class FakeRaw:
            sizes = SimpleNamespace(flip=6, width=1200, height=800)

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def extract_thumb(self):
                return SimpleNamespace(format=1, data=stream.getvalue())

        rawpy = SimpleNamespace(
            imread=lambda path: FakeRaw(), ThumbFormat=SimpleNamespace(JPEG=1, BITMAP=2)
        )
        with patch.dict(sys.modules, {"rawpy": rawpy}):
            image = decode_image(Path("portrait.arw"), {})
        self.assertEqual(image.size, (800, 1200))

    def test_tiny_raw_thumbnail_falls_back_to_half_size_sensor_data(self) -> None:
        stream = io.BytesIO()
        patterned_image((160, 120)).save(stream, format="JPEG")
        calls = []

        class FakeRaw:
            sizes = SimpleNamespace(flip=0, width=1200, height=800)

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def extract_thumb(self):
                return SimpleNamespace(format=1, data=stream.getvalue())

            def postprocess(self, **options):
                calls.append(options)
                return np.zeros((400, 600, 3), dtype=np.uint8)

        rawpy = SimpleNamespace(
            imread=lambda path: FakeRaw(),
            ThumbFormat=SimpleNamespace(JPEG=1, BITMAP=2),
            ColorSpace=SimpleNamespace(sRGB=1),
        )
        with patch.dict(sys.modules, {"rawpy": rawpy}):
            image = decode_image(Path("tiny-thumbnail.arw"), {})
        self.assertEqual(image.size, (600, 400))
        self.assertTrue(calls[0]["half_size"])

    def test_raw_orientation_uses_pixels_when_thumbnail_exif_is_stale(self) -> None:
        for pixel_size, thumbnail_orientation in (((1200, 800), 1), ((800, 1200), 6)):
            with self.subTest(
                pixel_size=pixel_size, thumbnail_orientation=thumbnail_orientation
            ):
                stream = io.BytesIO()
                exif = Image.Exif()
                exif[274] = thumbnail_orientation
                patterned_image(pixel_size).save(stream, format="JPEG", exif=exif)

                class FakeRaw:
                    sizes = SimpleNamespace(flip=6, width=1200, height=800)
                    thumbnail_data = stream.getvalue()

                    def __enter__(self):
                        return self

                    def __exit__(self, *exception):
                        return False

                    def extract_thumb(self):
                        return SimpleNamespace(format=1, data=self.thumbnail_data)

                rawpy = SimpleNamespace(
                    imread=lambda path: FakeRaw(),
                    ThumbFormat=SimpleNamespace(JPEG=1, BITMAP=2),
                )
                with patch.dict(sys.modules, {"rawpy": rawpy}):
                    image = decode_image(Path("portrait.arw"), {})
                self.assertEqual(image.size, (800, 1200))


if __name__ == "__main__":
    unittest.main()
