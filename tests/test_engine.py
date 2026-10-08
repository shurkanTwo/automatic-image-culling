"""Deterministic behavior tests; no real personal images or model downloads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from culling_engine.analysis import analyze_image
from culling_engine.cli import self_test
from culling_engine.discovery import capture_time, discover_images, source_identity
from culling_engine.engine import analyze_photo, generate_detail, scan
from culling_engine.grouping import group_photos
from culling_engine.imaging import decode_image


def patterned_image(size: tuple[int, int] = (120, 80)) -> Image.Image:
    generator = np.random.default_rng(7)
    return Image.fromarray(
        generator.integers(25, 230, (size[1], size[0], 3), dtype=np.uint8)
    )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
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
        self.assertTrue(self_test()["success"])


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
            "visualSignature": [0.4] * 48,
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


class RawDecodeTests(unittest.TestCase):
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
        patterned_image((80, 120)).save(stream, format="JPEG")

        class FakeRaw:
            sizes = SimpleNamespace(flip=6, width=120, height=80)

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
        self.assertEqual(image.size, (80, 120))


if __name__ == "__main__":
    unittest.main()
