"""Bundle the image worker and smoke-test it outside the source checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def validate_self_test(records: list[dict]) -> None:
    expected = json.loads((ROOT / "desktop/package.json").read_text())["version"]
    probes = [record for record in records if record.get("type") == "self-test"]
    required = {"decode", "orientation", "cache", "detail", "json", "heif"}
    if len(probes) != 1:
        raise RuntimeError("The bundled worker did not produce one self-test result")
    probe = probes[0]
    capabilities = probe.get("capabilities", {})
    if (
        probe.get("success") is not True
        or probe.get("version") != expected
        or not required.issubset(probe.get("checks", []))
        or any(
            capabilities.get(name) is not True
            for name in ("rawpy", "pillow_heif", "exifread")
        )
    ):
        raise RuntimeError(f"The bundled worker failed its capability checks: {probe}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()
    build = ROOT / "build"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "photo-select-engine",
        "--collect-all",
        "rawpy",
        "--collect-all",
        "pillow_heif",
        "--distpath",
        str(build / "engine-dist"),
        "--workpath",
        str(build / "engine-work"),
        "--specpath",
        str(build),
        str(ROOT / "engine_entry.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    source = build / "engine-dist" / "photo-select-engine"
    destination = ROOT / "desktop" / "src-tauri" / "resources" / "engine"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    executable = destination / (
        "photo-select-engine.exe" if sys.platform == "win32" else "photo-select-engine"
    )
    if not args.skip_smoke_test:
        with tempfile.TemporaryDirectory(prefix="photo-select-bundle-test-") as workdir:
            result = subprocess.run(
                [str(executable), "self-test"],
                cwd=workdir,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=90,
                check=True,
            )
            records = [json.loads(line) for line in result.stdout.splitlines() if line]
            validate_self_test(records)
            print(json.dumps({"bundleSmokeTest": records}, ensure_ascii=False))
    plugin_source = ROOT / "lightroom" / "PhotoSelect.lrplugin"
    plugin_destination = (
        ROOT
        / "desktop"
        / "src-tauri"
        / "resources"
        / "lightroom"
        / "PhotoSelect.lrplugin"
    )
    if plugin_destination.exists():
        shutil.rmtree(plugin_destination)
    shutil.copytree(plugin_source, plugin_destination)
    print(f"Bundled engine: {executable}")


if __name__ == "__main__":
    main()
