# Photo Select

A local desktop companion for the first pass through a trip, event, or everyday photo collection. Compare similar photographs, choose favorites, and build purpose-specific collections before returning to Lightroom Classic.

Version 0.2.3 adds an automatic first pass with Cautious and Stronger modes, Lightroom Reject flag export, direct Windows updates, and automatic loading as you scroll through the photo grid. The saved choice to include subfolders is reused for rescans. Originals are read-only: Favorite and Discard record your choices without moving, deleting, or rewriting photographs.

## Windows test package

The **Build Photo Select desktop** GitHub Actions workflow produces an x64 installer, a portable ZIP, a Lightroom plugin ZIP, and `START-HERE.txt`. Download its `Photo-Select-0.2.3-Windows-x64` artifact.

Run `Photo-Select-0.2.3-Windows-x64-Setup.exe`. Python, Rust, and Node are bundled or unnecessary at runtime. The installer installs WebView2 if it is missing; that first installation may need an internet connection. This test build is unsigned.

To update an installed version, close Photo Select and run the newer Setup file. Setup keeps the existing installation location and updates the app directly, preserving projects, previews, review choices, and import settings. There is no need to uninstall the previous version. Running the same installer again repairs that version. If the app or its analysis worker is still running, interactive Setup asks you to close it and retry; silent Setup stops with an error. Setup never forces the app to close with unsaved reviews.

For portable use, extract the entire portable ZIP and run `Photo Select.exe` with its `resources` folder beside it. Portable use requires an installed WebView2 runtime. Both versions store projects under `%LOCALAPPDATA%\com.shurkantwo.photoselect\projects`.

## Review a photo collection

1. Choose the folder containing a trip or event. Before starting import, choose whether to **Include subfolders**; new projects default to photos directly inside the selected folder. The scope is shown in the project sidebar, and rescans reuse it. Existing projects retain their previous inclusion of subfolders. To choose a different scope, create a new project from the same folder. Enable **Automatic first pass** and choose **Cautious** or **Stronger**. New projects default to Cautious with automatic selection enabled. Previews appear as processing progresses; automatic choices apply after a successful complete scan. Import can be cancelled and restarted.
2. Browse the grid; more photographs load automatically as you approach the bottom, keeping your scroll position. Inspect a photograph, or compare two to four photographs with linked zoom and pan. **100%** prepares full-resolution previews and maps one image pixel to one physical display pixel, including scaled displays and comparison images with different dimensions. **Fit** returns to the full frame.
3. Use **F** for Favorite, **P** for Discard, **U** for Undecided, and **0–5** for a rating. Arrow keys navigate, **Enter** opens inspection, **G** returns to the grid, and **C** opens comparison. Ctrl/Shift-click selects several photographs; Ctrl+Z undoes review changes.
4. Add tags and collections such as “Japan photobook.” When several photos are selected, entered tags are added to each photo's existing tags. Reviews save to a local SQLite project automatically; closing commits a focused tag draft before waiting for saves. Rescanning preserves ratings, decisions, tags, and collections.
5. Export favorites or a collection to Lightroom Classic using the included plugin.

The automatic first pass chooses technically adequate representatives within similar moments and a limited shortlist of stronger unrelated previews. Cautious keeps close alternatives; Stronger makes a tighter shortlist and accepts more relative-blur evidence for discards. Almost completely featureless black/white previews and clearly much blurrier near-identical alternatives can become Discards. Other unselected photos remain Undecided; decoding failures never become automatic discards. Each automatic choice has an explanation and is marked Auto favorite or Auto discard.

Change the mode in the sidebar and run the first pass again. Same-mode cached proposals apply immediately; a changed mode needs a new analysis. **Undo automatic choices** resets choices still owned by automation and disables automatic selection on future rescans. Manual decisions, ratings, tags, and collection choices are protected. Existing projects start with automatic selection off. Automatic choices do not assign stars or mark photos manually reviewed.

Technical estimates measure visible detail and tonal clipping, and can favor texture or noise over smooth scenes. They do not evaluate artistic merit, identify subjects, detect closed eyes, or understand a photobook topic. Confidence describes technical evidence rather than a probability that you will like a photo. Moment grouping is deliberately conservative. RAW previews reflect camera rendering rather than Lightroom develop adjustments.

JPEG, PNG, TIFF, WebP, HEIF/HEIC, and LibRaw-supported RAW formats are supported, including Sony ARW. Individual corrupt or unsupported photographs appear with an error while the rest of the batch continues. Camera timestamps without an EXIF timezone retain their recorded wall-clock time.

If originals are unavailable during a successful rescan, their cached previews, full-resolution details, and review choices remain available. They receive an availability note and are removed from current analysis suggestions. Reviewed selections can still be exported while the source drive is offline. An unreadable folder stops the rescan with an error instead of reporting a misleading empty result.

## Lightroom Classic handoff

Use **Lightroom connection** in the app to locate `PhotoSelect.lrplugin`. Add that folder through Lightroom Classic’s **File → Plug-in Manager**. The separate plugin ZIP contains the same folder.

After exporting a selection JSON, run **Library → Plug-in Extras → Import Photo Select shortlist...**. Photographs must already be imported into the catalog at the same paths. The plugin shows a confirmation before applying changes and reports unmatched paths.

The import creates or adds to a collection within the `Photo Select` collection set. Favorites become Picks. **Include discards as Lightroom Rejects** is enabled by default in the export dialog; it includes project discards even when they are outside your chosen collection, without adding those extra discards to that collection. The plugin applies actual Reject flags; it never deletes photographs. Only ratings explicitly set in Photo Select are applied; untouched ratings stay unchanged. Tags become additional keywords. Develop settings are preserved. A busy catalog reports an import failure that can be retried; the metadata selection is applied in one transaction.

Use the bundled 0.2.3 plugin for new schema 2 exports. It also accepts old schema 1 exports, preserving their earlier behavior where Pass did not set Reject flags.

This is an additive, one-way handoff. Reimporting does not remove collection members or previously applied keywords. Exported Pick/Reject choices overwrite flags for those photos; undecided entries leave existing flags unchanged. Lightroom ratings and edits are not synchronized back to Photo Select. The Lua importer is tested against an SDK test double; the actual Lightroom catalog integration still needs testing in Lightroom Classic.

## Architecture

| Component | Responsibility |
| --- | --- |
| `desktop/src/` | React and TypeScript review interface |
| `desktop/src-tauri/` | Rust/Tauri desktop shell, SQLite persistence, background jobs, export |
| `culling_engine/` | Python image decoding, previews, technical hints, grouping, first-pass proposals |
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

The optional Rust integration test requires `PHOTO_SELECT_PYTHON` for generating test images. It exercises the real worker, automatic favorites/discards, both selection modes, protected manual Undecided choices, clearing automation, corrupt-image isolation, full-resolution details, rescan persistence, Reject export, and unchanged original bytes. Set `PHOTO_SELECT_ENGINE` to test the bundled worker instead of the Python module.

Windows packaging runs on a native Windows runner. After the checks, run `npm run tauri -- build --bundles nsis` in `desktop`, then `python scripts/collect_windows_package.py` from the repository root. The workflow verifies matching component versions, bundled decoder capabilities, real RAW decoding in Unicode paths, and actual launches of both the portable and installed app. It also checks same-version repair, preserved project data, and safe refusal to replace a running app. When an unexpired earlier build is available on the same branch, it downloads that installer and verifies an actual upgrade in the earlier installation's custom location. The pinned upstream RAW test photograph is downloaded for verification and is not included in the app.

The Windows installer uses the MIT-licensed Tauri CLI 2.12.1 NSIS template in `desktop/src-tauri/windows/`, with targeted update and process-check changes. Review those changes when updating the Tauri CLI.

For Linux development without installed toolchains, `scripts/Dockerfile.dev` provides a development image. Run it with the repository mounted at `/workspace` and port 1420 published. Follow the same dependency installation and checks inside the container.

The optional native Linux UI check uses the real webview, Rust IPC and image assets. After building the app and worker, install `tauri-driver` and run:

```sh
cargo install tauri-driver --version 2.1.0 --locked
cd desktop/src-tauri
cargo build --locked
cd ../..
xvfb-run -a dbus-run-session -- python scripts/check_native_ui.py \
  --application desktop/src-tauri/target/debug/photo-select --output build/native-review
```

Keep the development server running for a debug build. The check uses a temporary data directory and synthetic originals; it verifies automatic first-pass choices and manual protection across modes, undo and rescan behavior, Lightroom Reject exports, actual large-grid scrolling, dark controls and unobstructed thumbnail selection, folder-only and recursive scopes across rescans and reopening, rendered previews, stars and favorites, bulk tags, undo, comparison, full-resolution viewing, export, and a focused draft saved on native close.

## Legacy prototype and license

The original Python/Tkinter and static-report prototype remains in `src/` for reference. Its dependencies and configuration are separate from the rebuilt app. Its old move/copy decisions workflow is not used by Photo Select. The legacy executable workflow is manual only.

This repository remains source-available and proprietary under [LICENSE](LICENSE). Third-party dependency notices and the JSON decoder’s MIT license are included in the Windows package.
