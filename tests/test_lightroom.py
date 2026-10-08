"""Exercise the actual Lua import against a catalog test double."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lupa.lua51 import LuaRuntime

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "lightroom" / "PhotoSelect.lrplugin"


class LightroomBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute((ROOT / "tests/fixtures/lightroom_sdk.lua").read_text())
        self.lua.globals()["_PLUGIN"] = self.lua.table(path=str(PLUGIN))
        self.photos = self.lua.globals().TEST_PHOTOS

    def run_import(self, entries: list[dict], *, project_name: str = "Japan") -> None:
        self.lua.globals().TEST_SELECTION_JSON = json.dumps(
            {
                "schemaVersion": 1,
                "application": "Photo Select",
                "projectName": project_name,
                "collectionName": "Photobook",
                "photos": entries,
            }
        )
        self.lua.execute((PLUGIN / "ImportSelection.lua").read_text())

    def entry(self, path: str, **values: object) -> dict:
        return {
            "path": path,
            "rating": None,
            "decision": "favorite",
            "tags": [],
            **values,
        }

    def test_import_preserves_untouched_ratings_keywords_and_develop(self) -> None:
        self.lua.globals().testPhoto("C:\\Photos\\東京.ARW", 4, 0)
        self.run_import([self.entry("C:\\Photos\\東京.ARW", tags=["夜", "travel"])])
        photo = self.photos["C:\\Photos\\東京.ARW"]
        self.assertEqual(photo.metadata.rating, 4)
        self.assertEqual(photo.metadata.pickStatus, 1)
        self.assertTrue(photo.keywords.existing)
        self.assertTrue(photo.keywords["夜"])
        self.assertEqual(photo.developSettings, "unchanged")
        self.assertTrue(
            self.lua.globals()
            .TEST_COLLECTIONS["Japan - Photobook"]
            .photos[photo.localIdentifier]
        )

    def test_explicit_zero_rating_is_applied_and_pass_keeps_existing_flag(self) -> None:
        self.lua.globals().testPhoto("photo.jpg", 5, 1)
        self.run_import([self.entry("photo.jpg", rating=0, decision="pass")])
        self.assertEqual(self.photos["photo.jpg"].metadata.rating, 0)
        self.assertEqual(self.photos["photo.jpg"].metadata.pickStatus, 1)

    def test_missing_photos_are_counted_without_importing_or_moving_them(self) -> None:
        self.lua.globals().testPhoto("known.jpg", 2, 0)
        self.run_import([self.entry("known.jpg"), self.entry("missing.jpg")])
        self.assertEqual(self.photos["known.jpg"].metadata.rating, 2)
        message = self.lua.globals().TEST_MESSAGES[1]
        self.assertEqual(message.title, "Shortlist imported")
        self.assertIn("1 unmatched", message.message)
        self.assertIsNone(self.photos["missing.jpg"])

    def test_invalid_manifest_cannot_write_to_catalog(self) -> None:
        self.lua.globals().testPhoto("photo.jpg", 4, 0)
        self.run_import([self.entry("photo.jpg", rating=6)])
        self.assertEqual(self.lua.globals().TEST_WRITES, 0)
        self.assertEqual(self.photos["photo.jpg"].metadata.rating, 4)
        self.assertEqual(
            self.lua.globals().TEST_MESSAGES[1].title, "Photo Select import failed"
        )

    def test_cancel_before_apply_leaves_catalog_unchanged(self) -> None:
        self.lua.globals().testPhoto("photo.jpg", 4, 0)
        self.lua.globals().TEST_CONFIRM = "cancel"
        self.run_import([self.entry("photo.jpg", rating=1)])
        self.assertEqual(self.lua.globals().TEST_WRITES, 0)

    def test_duplicate_and_repeated_imports_do_not_duplicate_collection_members(
        self,
    ) -> None:
        self.lua.globals().testPhoto("photo.jpg", 4, 0)
        entries = [self.entry("photo.jpg"), self.entry("photo.jpg")]
        self.run_import(entries)
        self.run_import(entries)
        members = list(
            self.lua.globals().TEST_COLLECTIONS["Japan - Photobook"].photos.keys()
        )
        self.assertEqual(members, ["photo.jpg"])

    def test_catalog_timeouts_never_report_a_successful_import(self) -> None:
        for transaction in (1, 2):
            with self.subTest(transaction=transaction):
                self.setUp()
                self.lua.globals().testPhoto("photo.jpg", 4, 0)
                self.lua.globals().TEST_ABORT_WRITE = transaction
                self.run_import([self.entry("photo.jpg", rating=1, tags=["travel"])])
                photo = self.photos["photo.jpg"]
                self.assertEqual(photo.metadata.rating, 4)
                self.assertEqual(photo.metadata.pickStatus, 0)
                self.assertIsNone(photo.keywords.travel)
                message = self.lua.globals().TEST_MESSAGES[1]
                self.assertEqual(message.title, "Photo Select import failed")
                self.assertIn("Lightroom is busy", message.message)

    def test_metadata_failure_rolls_back_the_entire_selection(self) -> None:
        self.lua.globals().testPhoto("one.jpg", 4, 0)
        self.lua.globals().testPhoto("two.jpg", 3, 0)
        self.lua.globals().TEST_FAIL_PHOTO = "two.jpg"
        self.run_import(
            [
                self.entry("one.jpg", rating=1, tags=["travel"]),
                self.entry("two.jpg", rating=2),
            ]
        )
        self.assertEqual(self.photos["one.jpg"].metadata.rating, 4)
        self.assertEqual(self.photos["one.jpg"].metadata.pickStatus, 0)
        self.assertIsNone(self.photos["one.jpg"].keywords.travel)
        self.assertEqual(self.photos["two.jpg"].metadata.rating, 3)
        self.assertEqual(
            list(self.lua.globals().TEST_COLLECTIONS["Japan - Photobook"].photos), []
        )
        self.assertEqual(
            self.lua.globals().TEST_MESSAGES[1].title, "Photo Select import failed"
        )

    def test_unicode_limits_match_the_application_character_limits(self) -> None:
        self.lua.globals().testPhoto("photo.jpg", 4, 0)
        name, tag = "旅" * 200, "夜" * 100
        self.run_import([self.entry("photo.jpg", tags=[tag])], project_name=name)
        self.assertTrue(self.photos["photo.jpg"].keywords[tag])
        self.assertEqual(
            self.lua.globals().TEST_MESSAGES[1].title, "Shortlist imported"
        )

    def test_object_or_sparse_photo_and_tag_lists_are_rejected_before_writing(
        self,
    ) -> None:
        cases = [
            {"wrong": self.entry("photo.jpg")},
            [None, self.entry("photo.jpg")],
            [self.entry("photo.jpg", tags={"wrong": "travel"})],
            [self.entry("photo.jpg", tags=[None, "travel"])],
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                self.setUp()
                self.lua.globals().testPhoto("photo.jpg", 4, 0)
                self.run_import(entries)
                self.assertEqual(self.lua.globals().TEST_WRITES, 0)
                self.assertEqual(
                    self.lua.globals().TEST_MESSAGES[1].title,
                    "Photo Select import failed",
                )


if __name__ == "__main__":
    unittest.main()
