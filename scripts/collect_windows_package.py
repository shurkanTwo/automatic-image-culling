"""Assemble the Windows installer, portable app, and integrity manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.2.0"


def zip_directory(source: Path, destination: Path, prefix: str) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(prefix) / path.relative_to(source))


def main() -> None:
    release = ROOT / "desktop" / "src-tauri" / "target" / "release"
    installers = list((release / "bundle" / "nsis").glob("*-setup.exe"))
    if len(installers) != 1:
        raise RuntimeError(f"Expected one NSIS installer, found {installers}")
    output = ROOT / "artifacts" / "windows"
    output.mkdir(parents=True, exist_ok=True)
    name = f"Photo-Select-{VERSION}-Windows-x64"
    shutil.copy2(installers[0], output / f"{name}-Setup.exe")
    portable = ROOT / "build" / name
    if portable.exists():
        shutil.rmtree(portable)
    portable.mkdir(parents=True)
    shutil.copy2(release / "photo-select.exe", portable / "Photo Select.exe")
    shutil.copytree(
        ROOT / "desktop" / "src-tauri" / "resources", portable / "resources"
    )
    shutil.copy2(ROOT / "LICENSE", portable / "LICENSE.txt")
    shutil.copy2(ROOT / "scripts" / "WINDOWS-TESTING.txt", portable / "START-HERE.txt")
    notices = ROOT / "build" / "THIRD-PARTY-NOTICES.txt"
    if notices.exists():
        shutil.copy2(notices, portable / notices.name)
        shutil.copy2(notices, output / notices.name)
    zip_directory(portable, output / f"{name}-Portable.zip", name)
    zip_directory(
        ROOT / "lightroom" / "PhotoSelect.lrplugin",
        output / f"Photo-Select-{VERSION}-Lightroom-Plugin.zip",
        "PhotoSelect.lrplugin",
    )
    shutil.copy2(ROOT / "scripts" / "WINDOWS-TESTING.txt", output / "START-HERE.txt")
    manifest = {
        "application": "Photo Select",
        "version": VERSION,
        "platform": "windows-x64",
        "files": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.suffix in {".exe", ".zip"}
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
