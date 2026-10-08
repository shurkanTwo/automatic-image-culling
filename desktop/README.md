# Photo Select desktop UI

React and TypeScript frontend for the Tauri desktop application. Install Node 22+, then run `npm ci`, `npm test`, and `npm run build` in this directory. `npm run tauri dev` runs the native application; the root README describes the analysis engine setup.

`npm run dev` serves the UI on port 1420. A regular browser explains that photo access requires the desktop application. For an explicit sample-only preview, open `http://localhost:1420/?demo=1` or set `VITE_DEMO=true`. This mode uses labeled sample images from Unsplash and stores sample decisions only in browser local storage. It does not analyze a user's photographs.

Browser integration tests can install `window.__PHOTO_SELECT_BRIDGE__` before the app loads. Its `invoke(command, camelCaseArgs)` returns the same values as `contracts/app-v1.json`. Optional `listen(event, callback)`, `chooseFolder()`, `chooseProject()`, and `savePath(name)` functions supply events and test file choices. Native desktop builds use Tauri commands, native dialogs, and native asset URLs directly.

Manual updates are serialized and immediately saved to the project database. Undo retains the prior state of each edited photo, including whether a star rating had ever been assigned. Project refreshes wait for pending edits, and responses from a previous project cannot overwrite a newly opened project. Suggestions never become favorites or manual star ratings automatically.

Shortcuts: arrows navigate; F favorites; P passes; U resets to undecided; 0–5 assigns a rating; Enter opens the viewer; G returns to the grid; C compares; Ctrl/Cmd+Z undoes; Ctrl/Cmd+A selects the displayed scope; / focuses search. Ctrl/Cmd-click toggles selection and Shift-click selects a range. Shortcuts pause while editing text or using a dialog.
