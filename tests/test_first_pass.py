"""Behavioral checks for conservative, reversible technical first-pass choices."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from culling_engine.analysis import analyze_image
from culling_engine.cli import _parser
from culling_engine.engine import analyze_photo, scan
from culling_engine.grouping import group_photos
from culling_engine.selection import suggest_selection, valid_metrics


def structured_image(seed: int = 18) -> Image.Image:
    generator = np.random.default_rng(seed)
    coarse = Image.fromarray(generator.integers(35, 220, (8, 8), dtype=np.uint8))
    base = np.asarray(coarse.resize((320, 240), Image.Resampling.BICUBIC), dtype=float)
    x, y = np.meshgrid(np.arange(320), np.arange(240))
    pixels = np.clip(base + 18 * np.sin(x * 2) + 18 * np.sin(y * 2), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8)).convert("RGB")


def photo(identifier: str, image: Image.Image, *, minute: int = 0) -> dict:
    return {
        "id": identifier,
        "path": f"/{identifier}.png",
        "captureTime": f"2024-01-01T12:{minute:02d}:00Z",
        "width": image.width,
        "height": image.height,
        "analysisError": None,
        **analyze_image(image),
    }


def choices(photos: list[dict], *, mode: str = "cautious") -> dict[str, dict]:
    return {
        suggestion["photoId"]: suggestion
        for suggestion in suggest_selection(photos, group_photos(photos), mode=mode)
    }


class SelectionTests(unittest.TestCase):
    def test_sharp_peer_selected_and_severe_blur_rejected_in_matching_moment(self):
        sharp = structured_image()
        photos = [
            photo("sharp", sharp),
            photo("blur", sharp.filter(ImageFilter.GaussianBlur(3))),
        ]
        self.assertEqual(len(group_photos(photos)), 1)
        result = choices(photos)
        self.assertEqual(result["sharp"]["decision"], "favorite")
        self.assertEqual(result["blur"]["decision"], "pass")
        self.assertIn("matching composition", result["blur"]["reason"])
        self.assertGreater(result["blur"]["confidence"], 0.9)

    def test_stronger_mode_rejects_moderate_weaker_peer_and_cautious_keeps_it(self):
        sharp = structured_image()
        photos = [
            photo("sharp", sharp),
            photo("weaker", sharp.filter(ImageFilter.GaussianBlur(1.05))),
        ]
        self.assertEqual(choices(photos)["weaker"]["decision"], "undecided")
        self.assertEqual(choices(photos, mode="stronger")["weaker"]["decision"], "pass")

    def test_smooth_scene_and_unrelated_blur_remain_undecided(self):
        sharp = structured_image()
        x, y = np.meshgrid(np.arange(320), np.arange(240))
        landscape = Image.fromarray((110 + x / 6 + y / 10).astype(np.uint8)).convert(
            "RGB"
        )
        photos = [
            photo("sharp", sharp),
            photo("blur", sharp.filter(ImageFilter.GaussianBlur(3)), minute=1),
            photo("smooth", landscape, minute=2),
        ]
        self.assertEqual(len(group_photos(photos)), 3)
        for mode in ("cautious", "stronger"):
            result = choices(photos, mode=mode)
            self.assertEqual(result["blur"]["decision"], "undecided")
            self.assertEqual(result["smooth"]["decision"], "undecided")

    def test_matching_group_without_a_sharp_peer_does_not_reject_motion(self):
        first = structured_image().filter(ImageFilter.GaussianBlur(3))
        second = structured_image().filter(ImageFilter.GaussianBlur(4))
        photos = [photo("motion-a", first), photo("motion-b", second)]
        self.assertEqual(len(group_photos(photos)), 1)
        for mode in ("cautious", "stronger"):
            self.assertTrue(
                all(
                    item["decision"] == "undecided"
                    for item in choices(photos, mode=mode).values()
                )
            )

    def test_blank_extremes_rejected_but_meaningful_high_and_low_key_preserved(self):
        photos = []
        for name, background, subject in (("white", 255, 210), ("black", 0, 35)):
            blank = Image.new("RGB", (320, 240), (background,) * 3)
            meaningful = blank.copy()
            meaningful.paste((subject,) * 3, (100, 70, 220, 170))
            photos.extend([photo(name, blank), photo(f"{name}-subject", meaningful)])
        for mode in ("cautious", "stronger"):
            result = choices(photos, mode=mode)
            for name in ("white", "black"):
                self.assertEqual(result[name]["decision"], "pass")
                self.assertNotEqual(result[f"{name}-subject"]["decision"], "pass")
        gray = photo("gray", Image.new("RGB", (320, 240), (140,) * 3))
        self.assertEqual(choices([gray])["gray"]["decision"], "undecided")

    def test_all_nonselected_independent_frames_are_preserved_and_quota_is_bounded(
        self,
    ):
        photos = [
            photo(f"frame-{i:02}", structured_image(i), minute=i) for i in range(20)
        ]
        for mode, expected in (("cautious", 6), ("stronger", 5)):
            result = choices(photos, mode=mode)
            self.assertEqual(
                sum(item["decision"] == "favorite" for item in result.values()),
                expected,
            )
            self.assertFalse(
                any(item["decision"] == "pass" for item in result.values())
            )
            self.assertTrue(all(item["reason"] for item in result.values()))
            self.assertTrue(
                all(0 <= item["confidence"] <= 1 for item in result.values())
            )
            json.dumps(result, allow_nan=False)

    def test_ties_keep_two_in_cautious_mode_and_are_deterministic(self):
        image = structured_image()
        photos = [photo(identifier, image) for identifier in ("c", "a", "b")]
        result = choices(photos)
        self.assertEqual(
            [key for key, value in result.items() if value["decision"] == "favorite"],
            ["a", "b"],
        )
        self.assertEqual(result, choices(list(reversed(photos))))
        stronger = choices(photos, mode="stronger")
        self.assertEqual(
            [key for key, value in stronger.items() if value["decision"] == "favorite"],
            ["a"],
        )

    def test_decoder_failure_and_invalid_metrics_cannot_be_discarded(self):
        broken = photo("broken", Image.new("RGB", (40, 40), "black"))
        broken["analysisError"] = "Decode failed"
        missing = photo("missing", Image.new("RGB", (40, 40), "white"))
        missing.pop("technicalMetrics")
        result = choices([broken, missing])
        for suggestion in result.values():
            self.assertEqual(suggestion["decision"], "undecided")
            self.assertEqual(suggestion["confidence"], 0)
        metrics = analyze_image(structured_image())["technicalMetrics"]
        for value in (float("nan"), float("inf"), -1, 5, True, "0.1"):
            self.assertFalse(valid_metrics({**metrics, "detail": value}))

    def test_cli_defaults_to_cautious_and_accepts_stronger(self):
        argv = ["scan", "--source", "/source", "--cache", "/cache"]
        self.assertEqual(_parser().parse_args(argv).selection_mode, "cautious")
        self.assertEqual(
            _parser()
            .parse_args(argv + ["--selection-mode", "stronger"])
            .selection_mode,
            "stronger",
        )
        with self.assertRaises(ValueError):
            suggest_selection([], [], mode="unsafe")

    def test_scan_emits_complete_proposals_and_modes_reuse_analysis_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            structured_image().save(source / "sharp.png")
            Image.new("RGB", (320, 240), "white").save(source / "blank.png")
            (source / "broken.png").write_bytes(b"not an image")
            records = []
            scan(source, root / "cache", records.append, workers=2)
            self.assertEqual(
                [record["type"] for record in records][-3:],
                ["groups", "suggestions", "complete"],
            )
            photos = [
                record["photo"] for record in records if record["type"] == "photo"
            ]
            suggestions = records[-2]["suggestions"]
            self.assertEqual(
                {item["photoId"] for item in suggestions},
                {item["id"] for item in photos},
            )
            broken = next(item for item in photos if item["filename"] == "broken.png")
            self.assertEqual(
                next(
                    item["decision"]
                    for item in suggestions
                    if item["photoId"] == broken["id"]
                ),
                "undecided",
            )
            cache_times = {
                item["previewPath"]: Path(item["previewPath"]).stat().st_mtime_ns
                for item in photos
                if not item["analysisError"]
            }
            repeated = []
            scan(
                source,
                root / "cache",
                repeated.append,
                workers=1,
                selection_mode="stronger",
            )
            self.assertTrue(
                all(
                    Path(path).stat().st_mtime_ns == timestamp
                    for path, timestamp in cache_times.items()
                )
            )
            for item in photos:
                self.assertNotIn("technicalMetrics", item)

    def test_corrupt_cached_metrics_force_reanalysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "photo.png"
            structured_image().save(image_path)
            original = analyze_photo(image_path, root / "cache")
            record_path = Path(original["previewPath"]).with_name("analysis.json")
            for metrics in (
                None,
                {**original["technicalMetrics"], "shadowClipping": 2},
            ):
                invalid = {**original, "technicalMetrics": metrics}
                record_path.write_text(json.dumps(invalid), encoding="utf-8")
                with patch(
                    "culling_engine.engine.decode_image",
                    wraps=lambda path, tags: Image.open(path).convert("RGB"),
                ) as decoder:
                    repaired = analyze_photo(image_path, root / "cache")
                decoder.assert_called_once()
                self.assertTrue(valid_metrics(repaired["technicalMetrics"]))


if __name__ == "__main__":
    unittest.main()
