import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import fetch_windows_baseline as baseline


class BaselineTests(unittest.TestCase):
    def test_numeric_version_order(self):
        self.assertLess(
            baseline.version_tuple("0.2.9"), baseline.version_tuple("0.2.10")
        )
        self.assertRaises(ValueError, baseline.version_tuple, "v0.2.2")

    def test_manifest_integrity_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"verified installer fixture"
            manifest = {
                "version": "0.2.2",
                "platform": "windows-x64",
                "files": {
                    "Photo-Select-0.2.2-Windows-x64-Setup.exe": {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                    }
                },
            }
            for valid in (True, False):
                archive = root / f"{valid}.zip"
                with zipfile.ZipFile(archive, "w") as package:
                    package.writestr("manifest.json", json.dumps(manifest))
                    package.writestr(
                        "Photo-Select-0.2.2-Windows-x64-Setup.exe",
                        content if valid else b"corrupt",
                    )
                if valid:
                    (root / "valid").mkdir()
                    (root / "valid/stale.exe").write_bytes(b"stale artifact")
                    self.assertEqual(
                        baseline.extract_and_verify(
                            archive, root / "valid", "0.2.2"
                        ).read_bytes(),
                        content,
                    )
                    self.assertFalse((root / "valid/stale.exe").exists())
                else:
                    self.assertRaises(
                        RuntimeError,
                        baseline.extract_and_verify,
                        archive,
                        root / "invalid",
                        "0.2.2",
                    )
            with zipfile.ZipFile(root / "unsafe.zip", "w") as package:
                package.writestr("../escaped", "bad")
            self.assertRaises(
                RuntimeError,
                baseline.extract_and_verify,
                root / "unsafe.zip",
                root / "unsafe",
                "0.2.2",
            )
            self.assertFalse((root / "escaped").exists())

    def test_selection_excludes_current_expired_and_nonolder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "desktop").mkdir()
            (root / "desktop/package.json").write_text('{"version":"0.2.3"}')

            def api(endpoint):
                if "/workflows/" in endpoint:
                    self.assertIn("branch=codex%2Ftest", endpoint)
                    return {
                        "workflow_runs": [
                            {"id": 99, "html_url": "current"},
                            {"id": 98, "html_url": "prior"},
                        ]
                    }
                self.assertIn("/98/", endpoint)
                return {
                    "artifacts": [
                        {
                            "id": 1,
                            "name": "Photo-Select-0.2.2-Windows-x64",
                            "expired": True,
                        },
                        {
                            "id": 2,
                            "name": "Photo-Select-0.2.3-Windows-x64",
                            "expired": False,
                        },
                        {
                            "id": 3,
                            "name": "Photo-Select-0.2.4-Windows-x64",
                            "expired": False,
                        },
                        {
                            "id": 4,
                            "name": "Photo-Select-0.2.2-Windows-x64",
                            "expired": False,
                        },
                    ]
                }

            def download(endpoint, destination):
                self.assertIn("/4/", endpoint)
                destination.write_bytes(b"archive")

            environment = {
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_HEAD_REF": "codex/test",
                "GITHUB_REF_NAME": "irrelevant",
                "GITHUB_RUN_ID": "99",
                "RUNNER_TEMP": str(root),
            }
            with patch.dict(baseline.os.environ, environment), patch.object(
                baseline, "ROOT", root
            ), patch.object(baseline, "api_json", side_effect=api), patch.object(
                baseline, "download_artifact", side_effect=download
            ), patch.object(
                baseline, "extract_and_verify", return_value=root / "old.exe"
            ):
                baseline.main()
            result = json.loads(
                (root / "photo-select-upgrade-baseline/baseline.json").read_text()
            )
            self.assertEqual(result["artifactId"], 4)
            self.assertEqual(result["version"], "0.2.2")

    def test_no_available_old_package_reports_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "desktop").mkdir()
            (root / "desktop/package.json").write_text('{"version":"0.2.3"}')
            environment = {
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_HEAD_REF": "",
                "GITHUB_REF_NAME": "main",
                "GITHUB_RUN_ID": "99",
                "RUNNER_TEMP": str(root),
            }
            with patch.dict(baseline.os.environ, environment), patch.object(
                baseline, "ROOT", root
            ), patch.object(
                baseline, "api_json", return_value={"workflow_runs": []}
            ), patch.object(
                baseline, "download_artifact"
            ) as download:
                baseline.main()
            download.assert_not_called()
            result = json.loads(
                (root / "photo-select-upgrade-baseline/baseline.json").read_text()
            )
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["branch"], "main")
            self.assertIn("No unexpired older package", result["reason"])


if __name__ == "__main__":
    unittest.main()
