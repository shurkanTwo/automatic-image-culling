"""Collect bundled Python license notices and the app's dependency inventory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    version = json.loads((ROOT / "desktop/package.json").read_text())["version"]
    installer_notice = (
        (ROOT / "desktop/src-tauri/windows/installer.nsi")
        .read_text(encoding="utf-8")
        .split("Unicode true", 1)[0]
    )
    installer_notice = "\n".join(
        line.removeprefix(";").removeprefix(" ")
        for line in installer_notice.splitlines()
    )
    sections = [
        f"Photo Select {version} - third-party dependencies\n",
        "Application source and its license: " + (ROOT / "LICENSE").read_text(),
        "\nWindows installer template (Tauri CLI 2.12.1):\n" + installer_notice + "\n",
    ]
    for name in ("Pillow", "numpy", "rawpy", "ExifRead", "pillow-heif", "PyInstaller"):
        distribution = importlib.metadata.distribution(name)
        sections.append(f"\n{'=' * 72}\n{name} {distribution.version}\n")
        for item in distribution.files or []:
            filename = str(item)
            if "dist-info" not in filename:
                continue
            if any(
                part.startswith(("license", "copying", "notice"))
                for part in filename.lower().split("/")
            ):
                file = distribution.locate_file(item)
                if file.is_file():
                    sections.append(
                        f"\n{item}\n{file.read_text(encoding='utf-8', errors='replace')}\n"
                    )
    for filename in ("LICENSE.txt", "LICENSE"):
        path = Path(sys.base_prefix) / filename
        if path.is_file():
            sections.append("\nPython runtime:\n" + path.read_text(errors="replace"))
    sections.append("\nFrontend production dependencies:\n")
    package = json.loads((ROOT / "desktop/package-lock.json").read_text())
    for location, metadata in package["packages"].items():
        if not location.startswith("node_modules/") or metadata.get("dev"):
            continue
        sections.append(f"{location}: {metadata.get('version', 'unknown')}\n")
        directory = ROOT / "desktop" / location
        for pattern in ("LICENSE*", "license*", "COPYING*", "NOTICE*"):
            for path in directory.glob(pattern):
                if path.is_file():
                    sections.append(
                        path.read_text(encoding="utf-8", errors="replace") + "\n"
                    )
    sections.append("\nRust dependencies (available target sources):\n")
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    registry = cargo_home / "registry/src"
    lock = tomllib.loads(
        (ROOT / "desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
    )
    for dependency in lock["package"]:
        if not dependency.get("source", "").startswith("registry+"):
            continue
        name = dependency["name"]
        version = dependency["version"]
        for directory in registry.glob(f"*/{name}-{version}"):
            manifest = tomllib.loads(
                (directory / "Cargo.toml").read_text(encoding="utf-8")
            )
            metadata = manifest.get("package", {})
            sections.append(
                f"\n{'=' * 72}\n{name} {version}\n"
                f"License: {metadata.get('license', 'See bundled source notices')}\n"
                f"Source: {metadata.get('repository', dependency['source'])}\n"
            )
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.name.lower().startswith(
                    ("license", "copying", "notice")
                ):
                    sections.append(
                        f"\n{path.relative_to(directory)}\n"
                        + path.read_text(encoding="utf-8", errors="replace")
                        + "\n"
                    )
    sections.append(
        "\nLua JSON decoder:\n"
        + (ROOT / "lightroom/PhotoSelect.lrplugin/json.lua")
        .read_text()
        .split("local json =", 1)[0]
    )
    output = ROOT / "build/THIRD-PARTY-NOTICES.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(sections), encoding="utf-8")
    destination = ROOT / "desktop/src-tauri/resources/THIRD-PARTY-NOTICES.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output.read_bytes())
    print(f"Dependency notices: {output}")


if __name__ == "__main__":
    main()
