"""Assemble the Windows installer, portable app, and integrity manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def release_version() -> str:
    """Refuse to bundle an application whose components disagree on version."""
    version = json.loads((ROOT / "desktop/package.json").read_text())["version"]
    components = {
        "Tauri": json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())[
            "version"
        ],
        "contract": json.loads((ROOT / "contracts/app-v1.json").read_text())["app"][
            "version"
        ],
    }
    for name, path, pattern in (
        ("worker", "culling_engine/__init__.py", r'__version__ = "([^"]+)"'),
        ("Rust", "desktop/src-tauri/Cargo.toml", r'(?m)^version = "([^"]+)"'),
    ):
        match = re.search(pattern, (ROOT / path).read_text())
        components[name] = match.group(1) if match else None
    plugin = (ROOT / "lightroom/PhotoSelect.lrplugin/Info.lua").read_text()
    numbers = re.search(r"major = (\d+), minor = (\d+), revision = (\d+)", plugin)
    components["Lightroom"] = ".".join(numbers.groups()) if numbers else None
    if any(value != version for value in components.values()):
        raise RuntimeError(f"Component versions disagree with {version}: {components}")
    return version


def zip_directory(source: Path, destination: Path, prefix: str) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(prefix) / path.relative_to(source))


def main() -> None:
    version = release_version()
    release = ROOT / "desktop" / "src-tauri" / "target" / "release"
    installers = list((release / "bundle" / "nsis").glob(f"*_{version}_*-setup.exe"))
    if len(installers) != 1:
        raise RuntimeError(f"Expected one NSIS installer, found {installers}")
    output = ROOT / "artifacts" / "windows"
    # This directory contains generated packages only. Never mix stale builds in a manifest.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f"Photo-Select-{version}-Windows-x64"
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
        output / f"Photo-Select-{version}-Lightroom-Plugin.zip",
        "PhotoSelect.lrplugin",
    )
    shutil.copy2(ROOT / "scripts" / "WINDOWS-TESTING.txt", output / "START-HERE.txt")
    manifest = {
        "application": "Photo Select",
        "version": version,
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
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as github_output:
            github_output.write(f"artifact-name={name}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
