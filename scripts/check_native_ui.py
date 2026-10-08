"""Exercise the actual Tauri webview, IPC, previews, and saved review decisions.

Requires tauri-driver and the platform WebDriver. On Linux, run under
xvfb-run and dbus-run-session. No bridge mocks or browser-only demo are used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
CONTROL, NULL, ENTER = "\ue009", "\ue000", "\ue007"


class NativeWebview:
    """A small client for the standard W3C WebDriver endpoints used here."""

    def __init__(self, application: Path, port: int) -> None:
        self.address = f"http://127.0.0.1:{port}"
        self.session = ""
        self.driver = subprocess.Popen(
            ["tauri-driver", "--port", str(port), "--native-port", str(port + 1)],
            stdout=subprocess.DEVNULL,
        )
        try:
            self.wait(lambda: self.request("GET", "/status"), timeout=20)
            result = self.request(
                "POST",
                "/session",
                {
                    "capabilities": {
                        "alwaysMatch": {
                            "browserName": "wry",
                            "tauri:options": {"application": str(application)},
                        }
                    }
                },
            )
            self.session = result["sessionId"]
            self.command("POST", "/timeouts", {"script": 30000})
        except Exception:
            self.close()
            raise

    def request(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.address + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)["value"]
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result)
        return result

    def command(self, method: str, path: str, payload: dict | None = None):
        return self.request(method, f"/session/{self.session}{path}", payload)

    def execute(self, script: str, arguments: list | None = None):
        return self.command(
            "POST", "/execute/sync", {"script": script, "args": arguments or []}
        )

    def invoke(self, name: str, arguments: dict | None = None):
        result = self.command(
            "POST",
            "/execute/async",
            {
                "script": """
                    const done = arguments[arguments.length - 1];
                    window.__TAURI_INTERNALS__.invoke(arguments[0], arguments[1])
                        .then(value => done({ok: true, value}))
                        .catch(error => done({ok: false, error: String(error)}));
                """,
                "args": [name, arguments or {}],
            },
        )
        if not result["ok"]:
            raise RuntimeError(f"Native command {name}: {result['error']}")
        return result.get("value")

    def element(self, selector: str) -> str:
        return self.command(
            "POST", "/element", {"using": "css selector", "value": selector}
        )[ELEMENT_KEY]

    def click(self, selector: str) -> None:
        self.command("POST", f"/element/{self.element(selector)}/click", {})

    def type(self, selector: str, text: str, *, clear: bool = False) -> None:
        element = self.element(selector)
        if clear:
            self.command("POST", f"/element/{element}/clear", {})
        self.command("POST", f"/element/{element}/value", {"text": text})

    @staticmethod
    def wait(check: Callable, *, timeout: int = 30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = check()
                if result:
                    return result
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.1)
        raise TimeoutError("Native UI verification did not reach the expected state")

    def close(self) -> None:
        try:
            if self.session:
                self.command("DELETE", "")
        except (urllib.error.URLError, ConnectionError):
            pass
        finally:
            self.driver.terminate()
            try:
                self.driver.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.driver.kill()
                self.driver.wait(timeout=10)


def make_photos(source: Path) -> dict[Path, tuple[str, int]]:
    source.mkdir()
    for index in range(1, 4):
        image = Image.new("RGB", (1800, 1200), (50 * index, 75, 120))
        draw = ImageDraw.Draw(image)
        for offset in range(0, 1800, 120):
            draw.rectangle((offset, 120, offset + 60, 1080), fill=(240, 200, 80))
        image.save(source / f"写真{index}.png")
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in source.iterdir()
    }


def review_journey(view: NativeWebview, source: Path, output: Path) -> dict:
    view.wait(lambda: view.execute("return Boolean(window.__TAURI_INTERNALS__);"))
    state = view.invoke("get_app_state")
    if not state["engineAvailable"]:
        raise RuntimeError("The native application cannot access its image worker")
    project = view.invoke(
        "create_project", {"name": "Native review", "sourceDir": str(source)}
    )
    project_id = project["id"]
    arguments = {"projectId": project_id}
    view.invoke("start_import", arguments)

    def completed():
        current = view.invoke("get_project", arguments)
        if current["importStatus"] == "failed":
            raise RuntimeError(current["importError"])
        return current if current["importStatus"] == "completed" else None

    project = view.wait(completed)
    if len(project["photos"]) != 3:
        raise RuntimeError("Native import did not produce all three photographs")
    ids = {photo["filename"]: photo["id"] for photo in project["photos"]}
    view.invoke(
        "update_photo",
        {**arguments, "photoId": ids["写真2.png"], "patch": {"tags": ["family"]}},
    )
    view.command("POST", "/refresh", {})
    view.wait(
        lambda: view.execute("return Boolean(document.querySelector('.recent-card'));")
    )
    view.click(".recent-card")
    view.wait(
        lambda: view.execute(
            "return document.querySelectorAll('.photo-card img').length === 3 && [...document.querySelectorAll('.photo-card img')].every(image => image.naturalWidth > 0);"
        )
    )
    view.click('button.photo-card[aria-label^="写真1.png,"]')
    view.click('button[title="Favorite (F)"]')
    view.click('button[aria-label="5 stars"]')
    view.type("#photo-tags", "street" + ENTER, clear=True)
    view.wait(
        lambda: next(
            photo
            for photo in view.invoke("get_project", arguments)["photos"]
            if photo["id"] == ids["写真1.png"]
        )["tags"]
        == ["street"]
    )
    view.type("body", CONTROL + "a" + NULL)
    view.type("#photo-tags", "photobook" + ENTER)
    view.wait(
        lambda: all(
            "photobook" in photo["tags"]
            for photo in view.invoke("get_project", arguments)["photos"]
        )
    )
    current = view.invoke("get_project", arguments)
    if (
        "family"
        not in next(
            photo for photo in current["photos"] if photo["id"] == ids["写真2.png"]
        )["tags"]
    ):
        raise RuntimeError("Bulk tag addition replaced existing tags")
    view.click('button[title^="Undo Add tags"]')
    view.wait(
        lambda: all(
            "photobook" not in photo["tags"]
            for photo in view.invoke("get_project", arguments)["photos"]
        )
    )
    view.click('button.photo-card[aria-label^="写真3.png,"]')
    view.click('button[aria-label="Compare selected photos"]')
    view.wait(
        lambda: view.execute(
            "return document.querySelectorAll('.viewer-pane').length >= 2;"
        )
    )
    view.click('button[aria-label="Single photo view"]')
    view.click('button[title^="Show actual image pixels"]')
    view.wait(
        lambda: view.execute(
            "return document.querySelector('.zoom-label')?.textContent !== 'Fit' && document.querySelector('.photo-canvas img')?.naturalWidth === 1800;"
        )
    )

    def painted_detail():
        bounds = view.execute(
            "const r = document.querySelector('.photo-canvas').getBoundingClientRect(); return [r.x, r.y, r.right, r.bottom];"
        )
        image = base64.b64decode(view.command("GET", "/screenshot"))
        with Image.open(io.BytesIO(image)) as screenshot:
            pixels = (
                screenshot.crop(tuple(map(int, bounds))).resize((32, 32)).convert("RGB")
            )
            colorful = sum(
                count
                for count, pixel in pixels.getcolors(1024) or []
                if max(pixel) - min(pixel) > 30
            )
        if colorful < 100:
            return False
        (output / "native-viewer.png").write_bytes(image)
        return True

    view.wait(painted_detail)
    view.click('button[aria-label="Create collection"]')
    view.type(".modal input", "Native photobook")
    view.click('.modal button[type="submit"]')
    view.wait(lambda: len(view.invoke("get_project", arguments)["collections"]) == 1)
    collection = view.invoke("get_project", arguments)["collections"][0]
    view.invoke(
        "update_collection",
        {**arguments, "collectionId": collection["id"], "photoIds": list(ids.values())},
    )
    exported = output / "native-selection.json"
    view.invoke(
        "export_selection",
        {
            **arguments,
            "destination": str(exported),
            "collectionId": collection["id"],
            "onlyFavorites": False,
        },
    )
    payload = json.loads(exported.read_text(encoding="utf-8"))
    favorite = next(
        photo for photo in payload["photos"] if photo["path"].endswith("写真1.png")
    )
    if (
        len(payload["photos"]) != 3
        or favorite["rating"] != 5
        or favorite["decision"] != "favorite"
    ):
        raise RuntimeError(
            "The native Lightroom selection did not retain review decisions"
        )
    # Trigger a real native close while the tag field still has an uncommitted draft.
    view.type("#photo-tags", "close-draft", clear=True)
    view.execute(
        "setTimeout(() => window.__TAURI_INTERNALS__.invoke('plugin:window|close', {label: 'main'}), 200); return true;"
    )
    database = Path(project["projectPath"])

    def draft_saved():
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT data FROM photos WHERE id = ?", (ids["写真3.png"],)
            ).fetchone()
        return row and json.loads(row[0])["tags"] == ["close-draft"]

    view.wait(draft_saved)
    return {
        "success": True,
        "version": state["version"],
        "nativeIpc": True,
        "nativePreviewAssets": True,
        "favoriteAndRating": True,
        "bulkTagsPreserved": True,
        "atomicUndo": True,
        "compareAtLastPhoto": True,
        "fullResolution": True,
        "collectionExport": True,
        "focusedDraftSavedOnNativeClose": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=4444)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="photo-select-native-ui-") as temporary:
        workdir = Path(temporary)
        os.environ["XDG_DATA_HOME"] = str(workdir / "data")
        os.environ["XDG_CACHE_HOME"] = str(workdir / "cache")
        originals = make_photos(workdir / "旅 photos with spaces")
        view = NativeWebview(args.application.resolve(strict=True), args.port)
        try:
            report = review_journey(view, next(iter(originals)).parent, output)
        finally:
            view.close()
        for path, fingerprint in originals.items():
            if (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            ) != fingerprint:
                raise RuntimeError("An original changed during the native UI review")
        report["originalsUnchanged"] = True
    contents = json.dumps(report, indent=2) + "\n"
    (output / "native-ui-report.json").write_text(contents, encoding="utf-8")
    print(contents)


if __name__ == "__main__":
    main()
