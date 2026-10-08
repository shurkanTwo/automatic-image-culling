"""Collect bundled Python license notices and the app's dependency inventory."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    sections = [
        "Photo Select 0.2.0 - third-party dependencies\n",
        "Application source and its license: " + (ROOT / "LICENSE").read_text(),
    ]
    for name in ("Pillow", "numpy", "rawpy", "ExifRead", "pillow-heif"):
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
    sections.append("\nFrontend direct dependencies:\n")
    package = json.loads((ROOT / "desktop/package.json").read_text())
    for name, version in package["dependencies"].items():
        sections.append(f"{name}: {version}\n")
        directory = ROOT / "desktop/node_modules" / name
        for pattern in ("LICENSE*", "license*", "COPYING*", "NOTICE*"):
            for path in directory.glob(pattern):
                if path.is_file():
                    sections.append(
                        path.read_text(encoding="utf-8", errors="replace") + "\n"
                    )
    sections.append(
        "\nRust dependencies: see desktop/src-tauri/Cargo.lock in the source checkout.\n"
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
