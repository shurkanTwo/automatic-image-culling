"""Check that upgrade preservation verifies owned projects and original files."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.verify_windows_installation import (
    assert_preserved,
    seed_projects,
    snapshot,
)


class InstallationPreservationTests(unittest.TestCase):
    def test_runtime_profile_changes_do_not_hide_changes_to_originals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, originals = root / "data", root / "originals"
            seed_projects(data, originals)
            profile = data / "EBWebView" / "Cache"
            profile.mkdir(parents=True)
            cache_file = profile / "runtime-cache"
            cache_file.write_bytes(b"browser cache before launch")
            # An original folder with the same name is still user-owned data.
            original_folder = originals / "EBWebView"
            original_folder.mkdir()
            original = original_folder / "photo.jpg"
            original.write_bytes(b"original before launch")
            sidecar = originals / "photo-notes-wal"
            sidecar.write_bytes(b"user-owned sidecar")
            before = snapshot(data, ignore_browser_cache=True), snapshot(originals)
            cache_file.write_bytes(b"mutable browser cache after launch")
            assert_preserved(data, originals, before)
            original.write_bytes(b"changed original")
            with self.assertRaises(AssertionError):
                assert_preserved(data, originals, before)
            original.write_bytes(b"original before launch")
            sidecar.write_bytes(b"changed user-owned sidecar")
            with self.assertRaises(AssertionError):
                assert_preserved(data, originals, before)

    def test_database_reviews_and_generated_photo_caches_remain_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, originals = root / "data", root / "originals"
            projects = seed_projects(data, originals)
            before = snapshot(data, ignore_browser_cache=True), snapshot(originals)
            project_dir = data / "projects" / projects[0][0]
            preview = project_dir / "cache" / "preview.jpg"
            old_preview = preview.read_bytes()
            preview.write_bytes(b"changed preview")
            with self.assertRaisesRegex(AssertionError, "preview.jpg"):
                assert_preserved(data, originals, before)
            preview.write_bytes(old_preview)
            assert_preserved(data, originals, before)
            database = project_dir / "project.cullproj"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE photos SET data = '{}' WHERE id = ?", ("a" * 24,)
                )
            with self.assertRaisesRegex(AssertionError, "project.cullproj"):
                assert_preserved(data, originals, before)


if __name__ == "__main__":
    unittest.main()
