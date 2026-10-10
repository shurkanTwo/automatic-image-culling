"""Import preference tests using real JPEGs and isolated RAW decoder substitutes."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from culling_engine.cli import _parser
from culling_engine.discovery import RAW_EXTENSIONS, ImportPlan, discover_images
from culling_engine.engine import scan
from culling_engine.imaging import decode_image


class RawJpegTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.cache = self.root / "cache"
        self.image = Image.new("RGB", (120, 80), (90, 150, 200))
        self.image.paste((220, 35, 10), (10, 10, 45, 40))

    def original(self, relative: str) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in RAW_EXTENSIONS:
            path.write_bytes(b"Synthetic invalid RAW original")
        else:
            self.image.save(path, format="JPEG")
        return path

    def successful_raw_decode(self, path, tags):
        if path.suffix.lower() in RAW_EXTENSIONS:
            return self.image.copy()
        return decode_image(path, tags)

    def execute(self, *, prefer_raw=False, cache=None, include_subfolders=True):
        records = []
        summary = scan(
            self.source,
            cache or self.cache,
            records.append,
            workers=2,
            prefer_raw=prefer_raw,
            include_subfolders=include_subfolders,
        )
        photos = [record["photo"] for record in records if record["type"] == "photo"]
        excluded_records = [
            record for record in records if record["type"] == "excluded"
        ]
        self.assertEqual(len(excluded_records), 1)
        self.assertEqual(records[-4]["type"], "excluded")
        self.assertEqual(
            [record["type"] for record in records][-3:],
            ["groups", "suggestions", "complete"],
        )
        self.assertTrue(
            all(
                record["processed"] <= record["total"]
                for record in records
                if record["type"] == "progress"
            )
        )
        return summary, photos, excluded_records[0]["paths"], records

    def test_discovery_plan_matches_only_case_insensitive_stems_in_same_folder(self):
        raw = self.original("capture.NEF")
        companions = {
            self.original(name)
            for name in ("CAPTURE.JPG", "Capture.JPEG", "capture.JpE")
        }
        keep = {
            self.original(name)
            for name in ("capture-edit.jpg", "standalone.jpeg", "other/capture.jpg")
        }
        paths = discover_images(self.source, self.cache)
        self.assertEqual(set(paths), {raw} | companions | keep)
        self.assertEqual(set(ImportPlan(paths).initial_paths), set(paths))
        preferred = ImportPlan(paths, prefer_raw=True)
        self.assertEqual(set(preferred.initial_paths), {raw} | keep)
        self.assertFalse(preferred.excluded_paths)
        self.assertEqual(preferred.record_result(raw, successful=True), ())
        self.assertEqual(set(preferred.excluded_paths), companions)
        direct_paths = discover_images(
            self.source, self.cache, include_subfolders=False
        )
        self.assertNotIn(self.source / "other/capture.jpg", direct_paths)

    def test_preference_toggle_imports_real_jpegs_and_preserves_all_originals(self):
        raws = {self.original(name) for name in ("one.NEF", "two.CR2", "unpaired.ARW")}
        companions = {
            self.original(name) for name in ("ONE.JPG", "TWO.JPEG", "two.jpe")
        }
        independent = {
            self.original(name)
            for name in ("standalone.jpg", "one-edit.jpg", "nested/one.jpg")
        }
        originals = raws | companions | independent
        before = {
            path: (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            for path in originals
        }
        with patch(
            "culling_engine.engine.decode_image", side_effect=self.successful_raw_decode
        ):
            off_summary, off_photos, off_excluded, _ = self.execute()
            on_summary, on_photos, on_excluded, records = self.execute(prefer_raw=True)
        self.assertEqual(
            off_summary, {"type": "complete", "processed": 9, "total": 9, "failed": 0}
        )
        self.assertEqual({Path(photo["path"]) for photo in off_photos}, originals)
        self.assertEqual(off_excluded, [])
        self.assertEqual(
            on_summary, {"type": "complete", "processed": 6, "total": 6, "failed": 0}
        )
        self.assertEqual(records[0], {"type": "scan", "total": 6})
        self.assertEqual(
            {Path(photo["path"]) for photo in on_photos}, raws | independent
        )
        self.assertEqual(set(map(Path, on_excluded)), companions)
        for photo in on_photos:
            with Image.open(photo["previewPath"]) as preview:
                preview.load()
        for path, (digest, timestamp) in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_corrupt_raw_falls_back_to_real_jpeg_with_failed_raw_evidence(self):
        raw = self.original("photo.CR2")
        jpeg = self.original("PHOTO.jpg")
        summary, photos, excluded, records = self.execute(prefer_raw=True)
        self.assertEqual(records[0], {"type": "scan", "total": 1})
        self.assertEqual(
            summary, {"type": "complete", "processed": 2, "total": 2, "failed": 1}
        )
        self.assertEqual(excluded, [])
        by_path = {Path(photo["path"]): photo for photo in photos}
        self.assertEqual(set(by_path), {raw, jpeg})
        self.assertTrue(by_path[raw]["analysisError"])
        self.assertIsNone(by_path[jpeg]["analysisError"])
        suggestions = records[-2]["suggestions"]
        self.assertEqual(
            next(
                item["decision"]
                for item in suggestions
                if item["photoId"] == by_path[raw]["id"]
            ),
            "undecided",
        )
        self.assertTrue(
            any(
                record["type"] == "progress" and record["total"] == 2
                for record in records
            )
        )

    def test_any_successful_matching_raw_suppresses_jpeg_and_retains_failed_raw(self):
        good = self.original("frame.NEF")
        failed = self.original("FRAME.CR2")
        jpeg = self.original("frame.JPG")

        def one_success(path, tags):
            return self.image.copy() if path == good else decode_image(path, tags)

        with patch("culling_engine.engine.decode_image", side_effect=one_success):
            summary, photos, excluded, records = self.execute(prefer_raw=True)
        self.assertEqual(
            summary, {"type": "complete", "processed": 2, "total": 2, "failed": 1}
        )
        self.assertEqual({Path(photo["path"]) for photo in photos}, {good, failed})
        self.assertEqual(excluded, [str(jpeg)])
        failed_photo = next(photo for photo in photos if photo["path"] == str(failed))
        self.assertTrue(failed_photo["analysisError"])
        self.assertEqual(
            next(
                item["decision"]
                for item in records[-2]["suggestions"]
                if item["photoId"] == failed_photo["id"]
            ),
            "undecided",
        )

    def test_all_matching_raws_fail_before_all_jpeg_variants_are_released(self):
        raws = {self.original(name) for name in ("frame.NEF", "FRAME.CR2")}
        jpegs = {self.original(name) for name in ("Frame.JPG", "frame.jpeg")}
        summary, photos, excluded, records = self.execute(prefer_raw=True)
        self.assertEqual(records[0], {"type": "scan", "total": 2})
        self.assertEqual(
            summary, {"type": "complete", "processed": 4, "total": 4, "failed": 2}
        )
        self.assertEqual({Path(photo["path"]) for photo in photos}, raws | jpegs)
        self.assertEqual(excluded, [])
        self.assertEqual(sum(bool(photo["analysisError"]) for photo in photos), 2)
        self.assertEqual(sum(record["type"] == "progress" for record in records), 4)

    def test_cli_preference_is_opt_in(self):
        arguments = ["scan", "--source", str(self.source), "--cache", str(self.cache)]
        self.assertFalse(_parser().parse_args(arguments).prefer_raw)
        self.assertTrue(_parser().parse_args(arguments + ["--prefer-raw"]).prefer_raw)


if __name__ == "__main__":
    unittest.main()
