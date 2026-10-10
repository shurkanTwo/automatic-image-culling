"""Check real RAW decoding with the bundled worker in Unicode paths.

The pinned upstream test photograph is downloaded only for verification; it is
never included in the distributed application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

FIXTURE_URL = (
    "https://raw.githubusercontent.com/letmaik/rawpy/"
    "a39c2e7a44911889c3360891012f862f904ba551/test/RAW_CANON_40D_SRAW_V103.CR2"
)
FIXTURE_SHA256 = "152382ce4dbf644899d12b41b4c577f07638aa3ec5745ac344c36bad93826125"


def download_fixture(destination: Path) -> None:
    if (
        destination.is_file()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == FIXTURE_SHA256
    ):
        return
    with urllib.request.urlopen(FIXTURE_URL, timeout=60) as response:
        contents = response.read()
    if hashlib.sha256(contents).hexdigest() != FIXTURE_SHA256:
        raise RuntimeError(
            "The upstream RAW test photograph failed its integrity check"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)


def run_worker(worker: Path, arguments: list[str], workdir: Path) -> list[dict]:
    result = subprocess.run(
        [str(worker), *arguments],
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines() if line]
    errors = [record for record in records if record.get("type") == "error"]
    if errors:
        raise RuntimeError(f"The bundled RAW worker reported errors: {errors}")
    return records


def check_raw_jpeg_pairs(
    worker: Path, source: Path, cache: Path, preview: Path, workdir: Path
) -> dict:
    """Exercise the import preference with a real RAW and ordinary JPEG shots."""
    original = source / "東京の街.CR2"
    companion = source / "東京の街.JPEG"
    standalone = source / "JPEG only.jpg"
    edited = source / "東京の街-edited.jpg"
    png = source / "東京の街.png"
    separate = source / "separate folder"
    separate.mkdir()
    other_folder = separate / "東京の街.jpg"
    with Image.open(preview) as image:
        for path in (companion, standalone, edited, png, other_folder):
            image.save(path)
    arguments = ["scan", "--source", str(source), "--cache", str(cache)]

    def scan_paths(options: list[str], expected: set[Path], failed: int = 0):
        records = run_worker(worker, [*arguments, *options], workdir)
        photos = [record["photo"] for record in records if record["type"] == "photo"]
        actual = {Path(photo["path"]) for photo in photos}
        if actual != expected:
            raise RuntimeError(
                f"RAW + JPEG preference included wrong originals: "
                f"expected {expected}, got {actual}"
            )
        complete = records[-1]
        if complete != {
            "type": "complete",
            "processed": len(expected),
            "total": len(expected),
            "failed": failed,
        } or any(
            record["processed"] > record["total"]
            for record in records
            if record["type"] == "progress"
        ):
            raise RuntimeError(
                f"RAW + JPEG progress totals are inconsistent: {complete}"
            )
        exclusions = [record for record in records if record["type"] == "excluded"]
        if len(exclusions) != 1:
            raise RuntimeError("RAW + JPEG scan did not report one exclusion set")
        wanted = {str(companion)} if "--prefer-raw" in options else set()
        if set(exclusions[0]["paths"]) != wanted:
            raise RuntimeError(f"RAW + JPEG exclusions are wrong: {exclusions}")
        return photos

    all_paths = {original, companion, standalone, edited, png, other_folder}
    fingerprints = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in all_paths
    }
    scan_paths([], all_paths)
    scan_paths(["--prefer-raw"], all_paths - {companion})
    scan_paths(
        ["--prefer-raw", "--no-subfolders"], all_paths - {companion, other_folder}
    )
    broken = source / "broken.CR2"
    broken.write_bytes(b"An intentionally invalid RAW for fallback verification")
    fallback = source / "broken.JPG"
    with Image.open(preview) as image:
        image.save(fallback)
    for path in (broken, fallback):
        fingerprints[path] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
    photos = scan_paths(
        ["--prefer-raw"], (all_paths - {companion}) | {broken, fallback}, failed=1
    )
    failed_raw = next(photo for photo in photos if Path(photo["path"]) == broken)
    jpeg_fallback = next(photo for photo in photos if Path(photo["path"]) == fallback)
    if not failed_raw["analysisError"] or jpeg_fallback["analysisError"]:
        raise RuntimeError("An unreadable RAW hid its usable JPEG companion")
    for path, fingerprint in fingerprints.items():
        if (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        ) != fingerprint:
            raise RuntimeError(f"RAW + JPEG import modified an original: {path}")
    return {
        "rawJpegPreferenceToggle": True,
        "rawJpegFilenameAndFolderMatching": True,
        "rawJpegFolderScope": True,
        "rawJpegFailureFallback": True,
        "rawJpegOriginalsUnchanged": True,
    }


def check_fixture(worker: Path, fixture: Path, workdir: Path) -> dict:
    source = workdir / "旅の写真 with spaces"
    source.mkdir()
    original = source / "東京の街.CR2"
    shutil.copyfile(fixture, original)
    before = (hashlib.sha256(original.read_bytes()).hexdigest(), original.stat())
    cache = workdir / "写真 cache"
    arguments = ["scan", "--source", str(source), "--cache", str(cache)]
    records = run_worker(worker, arguments, workdir)
    photos = [record["photo"] for record in records if record.get("type") == "photo"]
    if len(photos) != 1 or photos[0].get("analysisError"):
        raise RuntimeError(f"The real RAW photograph could not be decoded: {photos}")
    photo = photos[0]
    preview = Path(photo["previewPath"])
    with Image.open(preview) as image:
        image.load()
        preview_size = image.size
    preview_time = preview.stat().st_mtime_ns
    run_worker(worker, arguments, workdir)
    if preview.stat().st_mtime_ns != preview_time:
        raise RuntimeError("The unchanged RAW preview was unnecessarily regenerated")
    detail_path = cache / "detail" / "東京 detail.jpg"
    detail_args = ["detail", "--source", str(original), "--output", str(detail_path)]
    detail = run_worker(worker, detail_args, workdir)
    if len(detail) != 1 or detail[0].get("type") != "detail":
        raise RuntimeError(f"Invalid full-resolution RAW response: {detail}")
    with Image.open(detail_path) as image:
        image.load()
        detail_size = image.size
    if detail_size != (1944, 1296):
        raise RuntimeError(f"Incorrect full-resolution RAW dimensions: {detail_size}")
    detail_time = detail_path.stat().st_mtime_ns
    run_worker(worker, detail_args, workdir)
    if detail_path.stat().st_mtime_ns != detail_time:
        raise RuntimeError("The unchanged full-resolution RAW was regenerated")
    after = original.stat()
    if (
        hashlib.sha256(original.read_bytes()).hexdigest() != before[0]
        or after.st_mtime_ns != before[1].st_mtime_ns
        or after.st_size != before[1].st_size
    ):
        raise RuntimeError("The RAW original changed during verification")
    pairing = check_raw_jpeg_pairs(worker, source, cache, preview, workdir)
    return {
        "success": True,
        "fixtureSha256": before[0],
        "previewSize": preview_size,
        "detailSize": detail_size,
        "unicodePaths": True,
        "cacheReused": True,
        "originalUnchanged": True,
        **pairing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--download-fixture", type=Path)
    args = parser.parse_args()
    if args.download_fixture:
        download_fixture(args.download_fixture)
        print(f"Verified RAW fixture: {args.download_fixture}")
        return
    if not args.worker:
        parser.error("--worker is required unless --download-fixture is used")
    with tempfile.TemporaryDirectory(prefix="photo-select-raw-check-") as temporary:
        # Windows TEMP may use an 8.3 alias; workers return canonical long paths.
        workdir = Path(temporary).resolve(strict=True)
        fixture = args.fixture
        if fixture is None:
            fixture = workdir / "fixture.CR2"
            download_fixture(fixture)
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != FIXTURE_SHA256:
            raise RuntimeError(
                "The upstream RAW test photograph failed its integrity check"
            )
        report = check_fixture(args.worker.resolve(strict=True), fixture, workdir)
    contents = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(contents, encoding="utf-8")
    print(contents)


if __name__ == "__main__":
    main()
