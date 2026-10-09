"""Exercise native install, real old-version upgrade, and repair on an empty CI VM."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import closing
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ARP_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Photo Select"
VENDOR_KEY = r"Software\shurkantwo\Photo Select"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path, *, ignore_browser_cache: bool = False) -> dict:
    # The app rewrites this convenience index on startup; compare its content.
    # WebView2 owns EBWebView: its browser profile is mutable and can stay locked
    # after the parent app exits. Photo Select projects and image caches are in
    # projects/, so keep hashing every app-owned file while excluding that profile.
    paths = []
    for child in root.iterdir():
        if ignore_browser_cache and child.name == "EBWebView":
            continue
        paths.extend(child.rglob("*") if child.is_dir() else [child])
    return {
        str(path.relative_to(root)): (
            json.loads(path.read_text())
            if ignore_browser_cache and path == root / "recent-projects.json"
            else digest(path)
        )
        for path in sorted(paths)
        if path.is_file()
    }


def registration() -> dict | None:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            ARP_KEY,
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            return {
                name: winreg.QueryValueEx(key, name)[0]
                for name in ("DisplayVersion", "InstallLocation")
            }
    except FileNotFoundError:
        return None


def verify_registration(location: Path, version: str) -> None:
    import winreg

    registered = registration()
    assert registered, "Installer failed to register the application"
    assert registered["DisplayVersion"] == version, registered
    assert (
        Path(registered["InstallLocation"].strip('"')).resolve() == location.resolve()
    )
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        VENDOR_KEY,
        0,
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
    ) as key:
        assert Path(winreg.QueryValueEx(key, "")[0]).resolve() == location.resolve()


def installer_run(installer: Path, location: Path | None = None, expected=0) -> None:
    # NSIS /D consumes the rest of the raw command line, including spaces.
    command = subprocess.list2cmdline([str(installer), "/S"])
    if location is not None:
        command += f" /D={location}"
    result = subprocess.run(command, timeout=180, check=False)
    assert (
        result.returncode == expected
    ), f"{installer.name}: expected exit {expected}, got {result.returncode}"


def wait_uninstalled(location: Path, timeout: float = 45) -> None:
    # NSIS launches a temporary copy of its uninstaller and may let the original
    # launcher exit before that copy finishes removing files and registration.
    executable = location / "photo-select.exe"
    deadline = time.monotonic() + timeout
    while True:
        registered = registration()
        executable_present = executable.exists()
        if registered is None and not executable_present:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Fresh fixture uninstallation did not finish within {timeout:g}s: "
                f"remaining registration={registered!r}; "
                f"executable present={executable_present} ({executable})"
            )
        time.sleep(0.2)


def smoke(executable: Path, report: Path, version: str, project_ids=()) -> dict:
    environment = dict(os.environ, PHOTO_SELECT_SMOKE_TEST_OUTPUT=str(report))
    result = subprocess.run(
        [str(executable)],
        cwd=executable.parent,
        env=environment,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, f"App smoke exited {result.returncode}"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["version"] == version and data["engineAvailable"], data
    assert (Path(data["lightroomPluginPath"]) / "Info.lua").is_file()
    if project_ids:
        summaries = {project["id"]: project for project in data["projects"]}
        for project_id, scope in project_ids:
            project = summaries[project_id]
            assert project["includeSubfolders"] is scope, project
            assert project["photoCount"] == 1 and project["favoriteCount"] == 1, project
    print(f"Verified native app {version}: {report.name}", flush=True)
    return data


def seed_projects(data_root: Path, originals: Path) -> list[tuple[str, bool]]:
    originals.mkdir()
    image = originals / "reviewed original.jpg"
    Image.new("RGB", (64, 48), (42, 87, 123)).save(image)
    nested = originals / "subfolder"
    nested.mkdir()
    Image.new("RGB", (32, 24), (12, 56, 90)).save(nested / "untouched.jpg")
    projects = []
    index = {}
    for scope in (False, True):
        project_id = str(uuid.uuid4())
        projects.append((project_id, scope))
        folder = data_root / "projects" / project_id
        cache = folder / "cache"
        cache.mkdir(parents=True)
        for name in ("preview.jpg", "thumbnail.jpg", "detail.jpg"):
            shutil.copy2(image, cache / name)
        database = folder / "project.cullproj"
        index[project_id] = str(database)
        meta = {
            "id": project_id,
            "name": f"Preserved review scope={scope}",
            "sourceDir": str(originals),
            "includeSubfolders": scope,
            "projectPath": str(database),
            "createdAt": "2026-10-01T12:00:00Z",
            "updatedAt": "2026-10-01T12:00:00Z",
            "photos": [],
            "groups": [],
            "collections": [],
            "importStatus": "completed",
            "importError": None,
        }
        photo_id = "a" * 24
        photo = {
            "id": photo_id,
            "path": str(image),
            "filename": image.name,
            "previewPath": str(cache / "preview.jpg"),
            "thumbnailPath": str(cache / "thumbnail.jpg"),
            "detailPath": str(cache / "detail.jpg"),
            "captureTime": "2026-10-01T12:00:00Z",
            "width": 64,
            "height": 48,
            "camera": "Synthetic CI fixture",
            "groupId": "preserved-group",
            "qualityScore": 73.5,
            "hints": ["preserved hint"],
            "rating": 4,
            "ratingTouched": True,
            "decision": "favorite",
            "reviewed": True,
            "tags": ["photobook", "family"],
            "analysisError": None,
        }
        group = {
            "id": "preserved-group",
            "label": "Preserved moment",
            "photoIds": [photo_id],
            "recommendedPhotoIds": [photo_id],
        }
        collection = {
            "id": "preserved-collection",
            "name": "Photobook shortlist",
            "photoIds": [photo_id],
        }
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executescript(
                "CREATE TABLE meta(singleton INTEGER PRIMARY KEY,data TEXT NOT NULL);"
                "CREATE TABLE photos(id TEXT PRIMARY KEY,data TEXT NOT NULL);"
                "CREATE TABLE groups_data(id TEXT PRIMARY KEY,data TEXT NOT NULL);"
                "CREATE TABLE collections(id TEXT PRIMARY KEY,data TEXT NOT NULL);"
                "CREATE TABLE import_progress(singleton INTEGER PRIMARY KEY,data TEXT NOT NULL);"
                "PRAGMA user_version=1;"
            )
            connection.execute("INSERT INTO meta VALUES(1,?)", (json.dumps(meta),))
            for table, value in (
                ("photos", photo),
                ("groups_data", group),
                ("collections", collection),
            ):
                connection.execute(
                    f"INSERT INTO {table} VALUES(?,?)",
                    (value["id"], json.dumps(value)),
                )
            connection.execute(
                "INSERT INTO import_progress VALUES(1,?)",
                (json.dumps({"status": "completed", "processed": 1, "total": 1}),),
            )
    (data_root / "recent-projects.json").write_text(json.dumps(index))
    return projects


def windows_for_process(pid: int) -> list[int]:
    from ctypes import wintypes

    handles = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32 = ctypes.windll.user32
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]

    @callback_type
    def callback(handle, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(handle):
            handles.append(handle)
        return True

    user32.EnumWindows(callback, 0)
    return handles


def close_app(process: subprocess.Popen) -> None:
    from ctypes import wintypes

    post = ctypes.windll.user32.PostMessageW
    post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    for handle in windows_for_process(process.pid):
        post(handle, 0x0010, 0, 0)  # WM_CLOSE requests the app's normal shutdown.
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # This CI-owned process is only killed after graceful teardown failed.
        process.terminate()
        process.wait(timeout=15)
        raise AssertionError(
            "Fixture app failed to close gracefully with WM_CLOSE"
        ) from None


def assert_preserved(data: Path, originals: Path, before: tuple) -> None:
    actual = snapshot(data, ignore_browser_cache=True)
    changed = [
        key
        for key in sorted(actual.keys() | before[0].keys())
        if actual.get(key) != before[0].get(key)
    ]
    assert not changed, f"Project database, reviews, or cache changed: {changed[:15]}"
    assert snapshot(originals) == before[1], "Original photograph bytes changed"


def blocked_running_app(
    executable: Path, installer: Path, data: Path, originals: Path
) -> None:
    environment = dict(os.environ)
    environment.pop("PHOTO_SELECT_SMOKE_TEST_OUTPUT", None)
    process = subprocess.Popen(
        [str(executable)], cwd=executable.parent, env=environment
    )
    try:
        deadline = time.monotonic() + 45
        while not windows_for_process(process.pid):
            assert (
                process.poll() is None
            ), "Fixture application exited before opening its window"
            if time.monotonic() >= deadline:
                raise AssertionError("Fixture application failed to open its window")
            time.sleep(0.2)
        # Let startup persistence and the initial state request complete before hashing.
        time.sleep(2)
        before = (
            snapshot(executable.parent),
            snapshot(data, ignore_browser_cache=True),
            snapshot(originals),
        )
        installer_run(installer, expected=2)
        assert process.poll() is None, "Installer terminated the running application"
        assert (
            snapshot(executable.parent) == before[0]
        ), "Blocked installer changed app files"
        assert_preserved(data, originals, before[1:])
    finally:
        if process.poll() is None:
            close_app(process)
    assert process.returncode == 0, "Fixture application did not close normally"
    print(
        "Verified running application blocks installation without termination",
        flush=True,
    )


def blocked_running_worker(
    executable: Path, installer: Path, data: Path, originals: Path
) -> None:
    # Start an actual scan before pausing it so Restart Manager sees an
    # initialized worker rather than an image suspended at process creation.
    fixture = Path(
        tempfile.mkdtemp(
            prefix="Photo Select active worker ", dir=os.environ["RUNNER_TEMP"]
        )
    )
    source = fixture / "synthetic scan source"
    source.mkdir()
    for number in range(100):
        Image.new("RGB", (256, 192), (number, 87, 123)).save(
            source / f"synthetic-{number:03d}.png"
        )
    process = subprocess.Popen(
        [
            str(executable),
            "scan",
            "--source",
            str(source),
            "--cache",
            str(fixture / "synthetic scan cache"),
            "--workers",
            "1",
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    startup = queue.Queue()

    def read_start_record() -> None:
        try:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Worker closed stdout before reporting scan startup")
            startup.put(json.loads(line))
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            startup.put(error)

    reader = threading.Thread(target=read_start_record, daemon=True)
    reader.start()
    suspended = False
    try:
        try:
            started = startup.get(timeout=45)
        except queue.Empty:
            raise AssertionError(
                "Fixture worker did not report scan startup within 45s"
            ) from None
        reader.join(timeout=5)
        assert not reader.is_alive(), "Fixture worker startup reader did not finish"
        assert not isinstance(
            started, Exception
        ), f"Fixture worker failed to start analysis: {started}"
        assert started == {"type": "scan", "total": 100}, started
        assert process.poll() is None, "Fixture worker exited before it could be paused"
        suspend = ctypes.windll.ntdll.NtSuspendProcess
        suspend.argtypes = [ctypes.c_void_p]
        suspend.restype = ctypes.c_long
        assert suspend(int(process._handle)) == 0, "Could not pause initialized worker"
        suspended = True
        assert process.poll() is None, "Fixture worker exited while being paused"
        before = (
            snapshot(executable.parent.parent.parent),
            snapshot(data, ignore_browser_cache=True),
            snapshot(originals),
        )
        installer_run(installer, expected=2)
        assert process.poll() is None, "Installer terminated the running worker"
        assert snapshot(executable.parent.parent.parent) == before[0]
        assert_preserved(data, originals, before[1:])
    finally:
        # Resume only after successful suspension and let real analysis complete.
        if suspended:
            resume = ctypes.windll.ntdll.NtResumeProcess
            resume.argtypes = [ctypes.c_void_p]
            resume.restype = ctypes.c_long
            assert resume(int(process._handle)) == 0
        try:
            output, error = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.communicate(timeout=15)
            raise AssertionError(
                "Fixture worker failed to finish analysis during teardown"
            ) from None
    assert process.returncode == 0, (output, error)
    completed = [
        record
        for line in output.splitlines()
        if line and (record := json.loads(line)).get("type") == "complete"
    ]
    assert completed == [
        {"type": "complete", "processed": 100, "total": 100, "failed": 0}
    ], (completed, error)
    print("Verified running worker blocks installation without termination", flush=True)


def stale_files(installed: Path) -> list[Path]:
    paths = [
        installed / "resources/engine/obsolete-upgrade-fixture/stale.dll",
        installed
        / "resources/lightroom/PhotoSelect.lrplugin/obsolete-upgrade-fixture.lua",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale bundled file: must be removed during update")
    return paths


def blocked_resource_junction(
    installed: Path, installer: Path, temporary: Path, data: Path, originals: Path
) -> None:
    before = (
        snapshot(installed),
        snapshot(data, ignore_browser_cache=True),
        snapshot(originals),
    )
    outside = temporary / "outside bundled resources"
    outside.mkdir()
    sentinel = outside / "must survive.txt"
    sentinel.write_text("A junction must never redirect recursive bundle cleanup")
    expected = digest(sentinel)
    junction = installed / "resources/engine/upgrade-test-junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
    )
    try:
        installer_run(installer, expected=3)
        assert digest(sentinel) == expected, "Installer traversed a resource junction"
        assert_preserved(data, originals, before[1:])
    finally:
        # On Windows rmdir removes a directory junction without traversing its target.
        os.rmdir(junction)
    assert (
        snapshot(installed) == before[0]
    ), "Refused junction install changed app files"
    print(
        "Verified resource junction blocks cleanup without deleting outside files",
        flush=True,
    )


def main() -> None:
    assert os.name == "nt", "This verification requires native Windows"
    assert (
        os.environ.get("GITHUB_ACTIONS") == "true"
    ), "Use an ephemeral GitHub Actions VM"
    assert (
        registration() is None
    ), "Refusing to alter an existing Photo Select installation"
    packages = ROOT / "artifacts/windows"
    version = json.loads((packages / "manifest.json").read_text())["version"]
    data_root = Path(os.environ["LOCALAPPDATA"]) / "com.shurkantwo.photoselect"
    assert (
        not data_root.exists()
    ), "Refusing to alter pre-existing Photo Select user data"
    temporary = Path(
        tempfile.mkdtemp(
            prefix="Photo Select native checks ", dir=os.environ["RUNNER_TEMP"]
        )
    )
    installers = list(packages.glob("*-Setup.exe"))
    assert len(installers) == 1
    installer = installers[0]
    portable = list(packages.glob("*-Portable.zip"))
    assert len(portable) == 1
    with zipfile.ZipFile(portable[0]) as archive:
        archive.extractall(temporary / "portable")
    portable_executables = list((temporary / "portable").rglob("Photo Select.exe"))
    assert len(portable_executables) == 1
    smoke(portable_executables[0], packages / "portable-report.json", version)

    fresh = temporary / "fresh install with spaces"
    installer_run(installer, fresh)
    verify_registration(fresh, version)
    smoke(fresh / "photo-select.exe", packages / "fresh-install-report.json", version)
    installer_run(fresh / "uninstall.exe")
    wait_uninstalled(fresh)
    print("Verified fresh installation and fixture cleanup", flush=True)

    originals = temporary / "original photographs"
    projects = seed_projects(data_root, originals)
    before = snapshot(data_root, ignore_browser_cache=True), snapshot(originals)
    baseline_path = (
        Path(os.environ["RUNNER_TEMP"]) / "photo-select-upgrade-baseline/baseline.json"
    )
    baseline = json.loads(baseline_path.read_text())
    (packages / "upgrade-baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n"
    )
    installed = temporary / "custom prior install with spaces"
    if baseline["status"] == "available":
        installer_run(Path(baseline["installer"]), installed)
        verify_registration(installed, baseline["version"])
        smoke(
            installed / "photo-select.exe",
            packages / "old-install-report.json",
            baseline["version"],
        )
        assert_preserved(data_root, originals, before)
    else:
        installer_run(installer, installed)
        print(f"Actual old-version upgrade skipped: {baseline['reason']}", flush=True)
    user_file = installed / "user-owned-note.txt"
    user_file.write_text("Files outside managed resources must survive installation")
    user_file_hash = digest(user_file)
    obsolete = stale_files(installed)
    blocked_running_app(installed / "photo-select.exe", installer, data_root, originals)
    blocked_running_worker(
        installed / "resources/engine/photo-select-engine.exe",
        installer,
        data_root,
        originals,
    )
    blocked_resource_junction(installed, installer, temporary, data_root, originals)
    installer_run(installer)  # No /D: preserve the old custom registered location.
    verify_registration(installed, version)
    assert digest(user_file) == user_file_hash
    assert all(
        not path.exists() for path in obsolete
    ), "Upgrade retained obsolete bundled resources"
    assert_preserved(data_root, originals, before)
    smoke(
        installed / "photo-select.exe",
        packages / "upgrade-report.json",
        version,
        projects,
    )
    assert_preserved(data_root, originals, before)

    clean_resources = snapshot(installed / "resources")
    obsolete = stale_files(installed)
    # Repair must restore deleted required resources as well as removing stale files.
    (installed / "resources/lightroom/PhotoSelect.lrplugin/Info.lua").unlink()
    installer_run(installer)
    verify_registration(installed, version)
    assert digest(user_file) == user_file_hash
    assert all(
        not path.exists() for path in obsolete
    ), "Repair retained obsolete bundled resources"
    assert (
        snapshot(installed / "resources") == clean_resources
    ), "Repair failed to restore packaged resources"
    assert_preserved(data_root, originals, before)
    smoke(
        installed / "photo-select.exe",
        packages / "repair-report.json",
        version,
        projects,
    )
    assert_preserved(data_root, originals, before)
    report = {
        "newVersion": version,
        "freshInstall": "passed",
        "oldVersionUpgrade": (
            "passed" if baseline["status"] == "available" else "skipped"
        ),
        "oldVersion": baseline.get("version"),
        "oldVersionSkipReason": baseline.get("reason"),
        "sameVersionRepair": "passed",
        "registeredCustomLocationRetained": True,
        "runningAppAndWorkerBlockedWithoutTermination": True,
        "resourceJunctionRejectedBeforeCleanup": True,
        "staleBundledResourcesRemoved": True,
        "missingBundledPluginRestored": True,
        "filesOutsideBundledResourcesPreserved": True,
        "reviewTagsScopeCollectionsDatabaseAndCachePreserved": True,
        "originalPhotographBytesPreserved": True,
        "projects": [
            {"id": project_id, "includeSubfolders": scope}
            for project_id, scope in projects
        ],
    }
    (packages / "installation-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(
                f"\nNative Windows install verification: fresh and repair passed. "
                f"Old-version upgrade: {report['oldVersionUpgrade']} "
                f"({baseline.get('version', baseline.get('reason'))} → {version}).\n"
            )


if __name__ == "__main__":
    main()
