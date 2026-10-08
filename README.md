# Photo Select

A local desktop companion for the first pass through a trip, event, or everyday photo collection. Compare similar photographs, choose favorites, and build purpose-specific collections before returning to Lightroom Classic.

Version 0.2.0 rebuilds the application around a persistent review workspace. Originals are read-only: Favorite and Pass record your choices without moving, deleting, or rewriting photographs.

## Windows test package

The **Build Photo Select desktop** GitHub Actions workflow produces an x64 installer, a portable ZIP, a Lightroom plugin ZIP, and `START-HERE.txt`. Download its `Photo-Select-0.2.0-Windows-x64` artifact.

Run `Photo-Select-0.2.0-Windows-x64-Setup.exe`. Python, Rust, and Node are bundled or unnecessary at runtime. The installer installs WebView2 if it is missing; that first installation may need an internet connection. This test build is unsigned.

For portable use, extract the entire portable ZIP and run `Photo Select.exe` with its `resources` folder beside it. Portable use requires an installed WebView2 runtime. Both versions store projects under `%LOCALAPPDATA%\com.shurkantwo.photoselect\projects`.

## Review a photo collection

1. Choose the folder containing a trip or event. Subfolders are scanned, and previews appear as processing progresses. Import can be cancelled and restarted.
2. Browse the grid, inspect a photograph, or compare two to four photographs with linked zoom and pan. Request full-resolution previews before using 100% to assess fine detail.
3. Use **F** for Favorite, **P** for Pass, **U** for Undecided, and **0–5** for a rating. Arrow keys navigate, **Enter** opens inspection, **G** returns to the grid, and **C** opens comparison. Ctrl/Shift-click selects several photographs; Ctrl+Z undoes review changes.
4. Add tags and collections such as “Japan photobook.” Reviews save to a local SQLite project automatically. Rescanning preserves ratings, decisions, tags, and collections.
5. Export favorites or a collection to Lightroom Classic using the included plugin.

Technical estimates suggest which images deserve comparison. They measure visible detail and tonal clipping, and can favor texture or noise over smooth scenes. They do not evaluate artistic merit, identify subjects, detect closed eyes, or automatically choose favorites. Moment grouping is deliberately conservative. RAW previews reflect camera rendering rather than Lightroom develop adjustments.

JPEG, PNG, TIFF, WebP, HEIF/HEIC, and LibRaw-supported RAW formats are supported, including Sony ARW. Individual corrupt or unsupported photographs appear with an error while the rest of the batch continues. Camera timestamps without an EXIF timezone retain their recorded wall-clock time.

## Lightroom Classic handoff

Use **Lightroom connection** in the app to locate `PhotoSelect.lrplugin`. Add that folder through Lightroom Classic’s **File → Plug-in Manager**. The separate plugin ZIP contains the same folder.

After exporting a selection JSON, run **Library → Plug-in Extras → Import Photo Select shortlist...**. Photographs must already be imported into the catalog at the same paths. The plugin shows a confirmation before applying changes and reports unmatched paths.

The import creates or adds to a collection within the `Photo Select` collection set. Favorites become Picks. Only ratings explicitly set in Photo Select are applied; untouched ratings stay unchanged. Tags become additional keywords. Pass does not set Lightroom’s Reject flag. Develop settings are preserved.

This is an additive, one-way handoff. Reimporting does not remove collection members or previously applied keywords and flags. Lightroom ratings and edits are not synchronized back to Photo Select. The Lua importer is tested against an SDK test double; the actual Lightroom catalog integration still needs testing in Lightroom Classic.

## Architecture

| Component | Responsibility |
| --- | --- |
| `desktop/src/` | React and TypeScript review interface |
| `desktop/src-tauri/` | Rust/Tauri desktop shell, SQLite persistence, background jobs, export |
| `culling_engine/` | Python image decoding, previews, technical hints, grouping |
| `lightroom/PhotoSelect.lrplugin/` | Lua plugin using Lightroom Classic’s SDK |
| `contracts/app-v1.json` | Frontend, desktop, worker, and export interfaces |
| `scripts/` | Worker bundling, notices, package assembly, test instructions |

The image worker is bundled with PyInstaller and communicates through UTF-8 JSON lines. Processing stays on the computer; no account, cloud service, or model download is required. The desktop architecture supports Windows, macOS, and Linux; this workflow currently packages Windows x64. Builds for the other operating systems are not supplied yet.

## Development

Use Node 22, stable Rust, Python 3.12, and the platform prerequisites from [Tauri](https://v2.tauri.app/start/prerequisites/).

```sh
python -m pip install -r requirements-engine.txt -r requirements-dev.txt
cd desktop
npm ci
npm run tauri -- icon app-icon.svg
cd ..
python scripts/build_engine.py
python scripts/dependency_notices.py
cd desktop
npm run tauri -- dev
```

For development with an unbundled worker, set `PHOTO_SELECT_PYTHON` to the interpreter containing the engine dependencies. `PHOTO_SELECT_ENGINE` overrides the worker executable for integration testing. Browser previews show sample data only when explicitly requested with `?demo=1`; normal browser mode cannot access a photo library.

Checks:

```sh
python -m black --check culling_engine engine_entry.py scripts tests
python -m ruff check culling_engine engine_entry.py scripts tests
python -m unittest discover -s tests -v
python -m culling_engine self-test
cd desktop
npm test
npm run build
cd src-tauri
cargo fmt --check
cargo test --no-default-features --locked
cargo test --no-default-features --locked --test engine_integration -- --ignored
```

The optional Rust integration test requires `PHOTO_SELECT_PYTHON` for generating test images. It exercises the real worker, corrupt-image isolation, full-resolution details, rescan persistence, export, and unchanged original bytes. Set `PHOTO_SELECT_ENGINE` to test the bundled worker instead of the Python module.

Windows packaging runs on a native Windows runner. After the checks, run `npm run tauri -- build --bundles nsis` in `desktop`, then `python scripts/collect_windows_package.py` from the repository root. The workflow also checks the bundled worker outside the source checkout.

For Linux development without installed toolchains, `scripts/Dockerfile.dev` provides a development image. Run it with the repository mounted at `/workspace` and port 1420 published. Follow the same dependency installation and checks inside the container.

## Legacy prototype and license

The original Python/Tkinter and static-report prototype remains in `src/` for reference. Its dependencies and configuration are separate from the rebuilt app. Its old move/copy decisions workflow is not used by Photo Select. The legacy executable workflow is manual only.

This repository remains source-available and proprietary under [LICENSE](LICENSE). Third-party dependency notices and the JSON decoder’s MIT license are included in the Windows package.
