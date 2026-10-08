"""Exercise the actual Lua import against a catalog test double."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lupa import LuaRuntime

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "lightroom" / "PhotoSelect.lrplugin"


class LightroomBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute((ROOT / "tests/fixtures/lightroom_sdk.lua").read_text())
        self.lua.globals()["_PLUGIN"] = self.lua.table(path=str(PLUGIN))
        self.photos = self.lua.globals().TEST_PHOTOS

    def run_import(self, entries: list[dict]) -> None:
        self.lua.globals().TEST_SELECTION_JSON = json.dumps(
            {
                "schemaVersion": 1,
                "application": "Photo Select",
                "projectName": "Japan",
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


if __name__ == "__main__":
    unittest.main()
