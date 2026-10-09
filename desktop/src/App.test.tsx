// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, chooseFolder, chooseProject } from "./api";
import type { PhotoPatch, Project } from "./types";
import { testProject } from "./test/fixtures";
const native = vi.hoisted(() => ({
  close: vi.fn(async () => {}),
  handler: null as
    | null
    | ((event: { preventDefault: () => void }) => Promise<void>),
}));
vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    close: native.close,
    onCloseRequested: async (handler: typeof native.handler) => {
      native.handler = handler;
      return () => {
        native.handler = null;
      };
    },
  }),
}));
vi.mock("./api", () => ({
  isAvailable: true,
  isNative: true,
  isDemo: false,
  api: {
    state: vi.fn(),
    createProject: vi.fn(),
    start: vi.fn(),
    openProject: vi.fn(),
    project: vi.fn(),
    updatePhotos: vi.fn(),
    updatePhotoPatches: vi.fn(),
    createCollection: vi.fn(),
    plugin: vi.fn(),
    export: vi.fn(),
    detail: vi.fn(),
  },
  photoSrc: (path: string) => path,
  subscribe: vi.fn(async () => () => {}),
  exportPath: vi.fn(async () => "/export.json"),
  revealPath: vi.fn(async () => {}),
  chooseFolder: vi.fn(),
  chooseProject: vi.fn(),
  progressLabel: vi.fn(),
}));
let stored: Project;
beforeEach(() => {
  vi.resetAllMocks();
  native.handler = null;
  HTMLElement.prototype.scrollIntoView = vi.fn();
  HTMLElement.prototype.setPointerCapture = vi.fn();
  stored = testProject();
  vi.mocked(api.state).mockImplementation(async () => ({
    version: "0.2.2",
    engineAvailable: true,
    projects: [
      {
        id: stored.id,
        name: stored.name,
        sourceDir: stored.sourceDir,
        includeSubfolders: stored.includeSubfolders,
        projectPath: stored.projectPath,
        photoCount: stored.photos.length,
        favoriteCount: 0,
        updatedAt: "",
      },
    ],
  }));
  vi.mocked(api.openProject).mockImplementation(async () =>
    structuredClone(stored),
  );
  vi.mocked(api.project).mockImplementation(async () =>
    structuredClone(stored),
  );
  vi.mocked(chooseFolder).mockResolvedValue("/photos/New trip");
  vi.mocked(api.createProject).mockImplementation(
    async (name, sourceDir, includeSubfolders) => {
      stored = { ...testProject(), name, sourceDir, includeSubfolders };
      return structuredClone(stored);
    },
  );
  vi.mocked(api.start).mockResolvedValue({ jobId: "test-job" });
  vi.mocked(api.updatePhotos).mockImplementation(
    async (_id, ids, patch: PhotoPatch) => {
      const targets = new Set(ids);
      stored.photos = stored.photos.map((photo) =>
        targets.has(photo.id)
          ? {
              ...photo,
              ...patch,
              ratingTouched:
                patch.ratingTouched ??
                (patch.rating !== undefined ? true : photo.ratingTouched),
            }
          : photo,
      );
      return structuredClone(
        stored.photos.filter((photo) => targets.has(photo.id)),
      );
    },
  );
  vi.mocked(api.updatePhotoPatches).mockImplementation(async (_id, updates) => {
    const patches = new Map(
      updates.map((update) => [update.photoId, update.patch]),
    );
    stored.photos = stored.photos.map((photo) => {
      const patch = patches.get(photo.id);
      return patch ? { ...photo, ...patch } : photo;
    });
    return structuredClone(
      stored.photos.filter((photo) => patches.has(photo.id)),
    );
  });
  vi.mocked(api.plugin).mockResolvedValue("/PhotoSelect.lrplugin");
});
afterEach(cleanup);
async function openWorkspace() {
  const rendered = render(<App />);
  fireEvent.click(
    await screen.findByRole("button", { name: /Review trip.*photographs/ }),
  );
  await screen.findByRole("button", { name: /^DSC_0.jpg,/ });
  return rendered;
}
async function chooseImportFolder() {
  render(<App />);
  const choose = await screen.findByRole("button", {
    name: "Choose a photo folder",
  });
  await waitFor(() => expect(choose.hasAttribute("disabled")).toBe(false));
  act(() => choose.focus());
  fireEvent.click(choose);
  return screen.findByRole("dialog", { name: "Import photographs" });
}
describe("folder import scope", () => {
  it.each([false, true])(
    "creates a project with includeSubfolders=%s only after Start import and preserves scope through rescans",
    async (includeSubfolders) => {
      const dialog = await chooseImportFolder();
      const checkbox = within(dialog).getByRole("checkbox", {
        name: "Include subfolders",
      }) as HTMLInputElement;
      expect(checkbox.checked).toBe(false);
      expect(dialog.textContent).toContain("/photos/New trip");
      expect(dialog.textContent).toContain("Subfolders are skipped");
      expect(api.createProject).not.toHaveBeenCalled();
      expect(api.start).not.toHaveBeenCalled();
      if (includeSubfolders) {
        fireEvent.click(checkbox);
        expect(dialog.textContent).toContain("all of its subfolders");
      }
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Start import" }),
      );
      await screen.findByRole("button", { name: "DSC_0.jpg, undecided" });
      expect(api.createProject).toHaveBeenCalledExactlyOnceWith(
        "New trip",
        "/photos/New trip",
        includeSubfolders,
      );
      await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
      expect(
        screen.getByText(
          includeSubfolders ? "Includes subfolders" : "Folder only",
        ),
      ).toBeTruthy();
      for (let count = 2; count <= 3; count++) {
        const rescan = screen.getByRole("button", { name: "Rescan folder" });
        await waitFor(() =>
          expect(rescan.hasAttribute("disabled")).toBe(false),
        );
        fireEvent.click(rescan);
        await waitFor(() => expect(api.start).toHaveBeenCalledTimes(count));
      }
      expect(vi.mocked(api.start).mock.calls).toEqual([
        [stored.id],
        [stored.id],
        [stored.id],
      ]);
      expect(stored.includeSubfolders).toBe(includeSubfolders);
      expect(api.createProject).toHaveBeenCalledTimes(1);
    },
  );
  it("cancels without creating or starting a project and resets the choice for a new folder", async () => {
    const dialog = await chooseImportFolder();
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: "Include subfolders" }),
    );
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.start).not.toHaveBeenCalled();
    const choose = screen.getByRole("button", {
      name: "Choose a photo folder",
    });
    expect(document.activeElement).toBe(choose);
    fireEvent.click(choose);
    await screen.findByRole("dialog");
    expect(
      (
        screen.getByRole("checkbox", {
          name: "Include subfolders",
        }) as HTMLInputElement
      ).checked,
    ).toBe(false);
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.createProject).not.toHaveBeenCalled();
  });
  it("focuses and describes the scope checkbox and traps Tab inside the dialog", async () => {
    const dialog = await chooseImportFolder();
    const checkbox = within(dialog).getByRole("checkbox", {
      name: "Include subfolders",
    });
    const start = within(dialog).getByRole("button", { name: "Start import" });
    expect(document.activeElement).toBe(checkbox);
    expect(checkbox.getAttribute("aria-describedby")).toBe("import-scope");
    expect(document.getElementById("import-scope")?.textContent).toContain(
      "only photographs directly in this folder",
    );
    fireEvent.keyDown(checkbox, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(start);
    fireEvent.keyDown(start, { key: "Tab" });
    expect(document.activeElement).toBe(checkbox);
    fireEvent.keyDown(checkbox, { key: "f" });
    expect(api.updatePhotos).not.toHaveBeenCalled();
  });
  it("retains scope after creation failure for retry and prevents cancellation and duplicate submission while busy", async () => {
    const dialog = await chooseImportFolder();
    const checkbox = within(dialog).getByRole("checkbox", {
      name: "Include subfolders",
    }) as HTMLInputElement;
    fireEvent.click(checkbox);
    vi.mocked(api.createProject).mockRejectedValueOnce(
      new Error("Disk is full"),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await within(dialog).findByRole("alert");
    expect(dialog.textContent).toContain("Disk is full");
    expect(checkbox.checked).toBe(true);
    expect(api.start).not.toHaveBeenCalled();
    let resolveCreate!: (value: Project) => void;
    vi.mocked(api.createProject).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveCreate = resolve;
        }),
    );
    const start = within(dialog).getByRole("button", { name: "Start import" });
    fireEvent.click(start);
    await waitFor(() => expect(api.createProject).toHaveBeenCalledTimes(2));
    expect(start.hasAttribute("disabled")).toBe(true);
    expect(checkbox.disabled).toBe(true);
    expect(
      within(dialog)
        .getByRole("button", { name: "Cancel" })
        .hasAttribute("disabled"),
    ).toBe(true);
    fireEvent.keyDown(document.body, { key: "Escape" });
    fireEvent.submit(dialog);
    expect(screen.getByRole("dialog")).toBe(dialog);
    expect(api.createProject).toHaveBeenCalledTimes(2);
    await act(async () =>
      resolveCreate({ ...stored, includeSubfolders: true }),
    );
    await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
  it("offers a rescan retry when starting an already created project fails", async () => {
    const dialog = await chooseImportFolder();
    vi.mocked(api.start).mockRejectedValueOnce(
      new Error("Engine could not start"),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await screen.findByRole("alert");
    expect(screen.getByRole("alert").textContent).toContain(
      "Engine could not start",
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Rescan folder" }));
    await waitFor(() => expect(api.start).toHaveBeenCalledTimes(2));
    expect(api.createProject).toHaveBeenCalledTimes(1);
  });
  it("prevents rescans and project switching while the first import is starting", async () => {
    const dialog = await chooseImportFolder();
    let resolveStart!: (value: { jobId: string }) => void;
    vi.mocked(api.start).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveStart = resolve;
        }),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
    const rescan = screen.getByRole("button", { name: "Rescan folder" });
    const projects = screen.getByRole("button", { name: "Projects" });
    expect(rescan.hasAttribute("disabled")).toBe(true);
    expect(projects.hasAttribute("disabled")).toBe(true);
    fireEvent.click(rescan);
    expect(api.start).toHaveBeenCalledTimes(1);
    await act(async () => resolveStart({ jobId: "test-job" }));
    await waitFor(() => expect(rescan.hasAttribute("disabled")).toBe(false));
    expect(projects.hasAttribute("disabled")).toBe(false);
  });
  it("does nothing if the native folder picker is cancelled", async () => {
    vi.mocked(chooseFolder).mockResolvedValue(null);
    render(<App />);
    const choose = await screen.findByRole("button", {
      name: "Choose a photo folder",
    });
    await waitFor(() => expect(choose.hasAttribute("disabled")).toBe(false));
    fireEvent.click(choose);
    await waitFor(() => expect(choose.hasAttribute("disabled")).toBe(false));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.start).not.toHaveBeenCalled();
  });
  it.each([false, true])(
    "opens existing scope=%s from recent projects without an import prompt",
    async (includeSubfolders) => {
      stored.includeSubfolders = includeSubfolders;
      await openWorkspace();
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(
        screen.getByText(
          includeSubfolders ? "Includes subfolders" : "Folder only",
        ),
      ).toBeTruthy();
      expect(api.openProject).toHaveBeenCalledExactlyOnceWith(
        stored.projectPath,
      );
      expect(api.createProject).not.toHaveBeenCalled();
      expect(api.start).not.toHaveBeenCalled();
    },
  );
  it("opens a project from the native picker without asking for import scope", async () => {
    vi.mocked(chooseProject).mockResolvedValue(stored.projectPath);
    render(<App />);
    const open = await screen.findByRole("button", { name: "Open project" });
    await waitFor(() => expect(open.hasAttribute("disabled")).toBe(false));
    fireEvent.click(open);
    await screen.findByRole("button", { name: "DSC_0.jpg, undecided" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.start).not.toHaveBeenCalled();
  });
});
describe("whole application review flows", () => {
  it("ignores stale recommendations for failed photos when reopening an older project", async () => {
    stored.photos[0].analysisError = "Preview generation failed";
    stored.photos[0].previewPath = "";
    stored.photos[0].thumbnailPath = "";
    stored.groups = [
      {
        id: "older-moment",
        label: "Older moment",
        photoIds: ["photo-0", "photo-1"],
        recommendedPhotoIds: ["photo-0", "photo-1", "removed-photo"],
      },
    ];
    await openWorkspace();
    const failed = screen.getByRole("button", { name: "DSC_0.jpg, undecided" });
    const eligible = screen.getByRole("button", {
      name: "DSC_1.jpg, undecided",
    });
    expect(failed.querySelector(".recommendation-badge")).toBeNull();
    expect(eligible.querySelector(".recommendation-badge")).not.toBeNull();
    expect(stored.photos[0].decision).toBe("undecided");
    expect(api.updatePhotos).not.toHaveBeenCalled();
  });
  it("saves a focused tag draft before returning to Projects", async () => {
    await openWorkspace();
    const tags = screen.getByLabelText("Tags");
    act(() => {
      tags.focus();
    });
    fireEvent.change(tags, { target: { value: "family, album" } });
    fireEvent.click(screen.getByRole("button", { name: "Projects" }));
    await screen.findByText(/Find the ones/);
    expect(stored.photos[0].tags).toEqual(["family", "album"]);
  });
  it("retains the collection name and shows a creation error inside its dialog", async () => {
    await openWorkspace();
    vi.mocked(api.createCollection).mockRejectedValueOnce(
      new Error("Project is read only"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Create collection" }));
    fireEvent.change(screen.getByLabelText("Collection name"), {
      target: { value: "Our album" },
    });
    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create collection",
      }),
    );
    await waitFor(() =>
      expect(screen.getByRole("dialog").textContent).toContain(
        "Project is read only",
      ),
    );
    expect(
      (screen.getByLabelText("Collection name") as HTMLInputElement).value,
    ).toBe("Our album");
  });
  it("commits a still-focused tag draft before native closing", async () => {
    await openWorkspace();
    const tags = screen.getByLabelText("Tags");
    act(() => {
      tags.focus();
    });
    fireEvent.change(tags, { target: { value: "family, print" } });
    fireEvent.keyDown(tags, { key: "f" });
    expect(api.updatePhotos).not.toHaveBeenCalled();
    const preventDefault = vi.fn();
    await act(async () => {
      await native.handler!({ preventDefault });
    });
    expect(preventDefault).toHaveBeenCalled();
    expect(api.updatePhotos).toHaveBeenCalledWith("test-project", ["photo-0"], {
      tags: ["family", "print"],
    });
    expect(stored.photos[0].tags).toEqual(["family", "print"]);
    expect(native.close).toHaveBeenCalledTimes(1);
  });
  it("keeps a failed tag draft and shows save failure, then retries it on a second close", async () => {
    await openWorkspace();
    vi.mocked(api.updatePhotos).mockRejectedValueOnce(
      new Error("Disk is full"),
    );
    const tags = screen.getByLabelText("Tags") as HTMLTextAreaElement;
    act(() => {
      tags.focus();
    });
    fireEvent.change(tags, { target: { value: "family, print" } });
    await act(async () => {
      await native.handler!({ preventDefault: vi.fn() });
    });
    expect(native.close).not.toHaveBeenCalled();
    expect(tags.value).toBe("family, print");
    expect(screen.getByText("Last change failed to save")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("Disk is full");
    await act(async () => {
      await native.handler!({ preventDefault: vi.fn() });
    });
    expect(stored.photos[0].tags).toEqual(["family", "print"]);
    expect(native.close).toHaveBeenCalledTimes(1);
  });
  it("compares two photographs at the end of the library and edits the clicked pane", async () => {
    const { container } = await openWorkspace();
    fireEvent.click(
      screen.getByRole("button", { name: "DSC_3.jpg, undecided" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Compare selected photos" }),
    );
    expect(container.querySelectorAll(".viewer-pane")).toHaveLength(2);
    expect(container.querySelector(".viewer-pane")?.textContent).toContain(
      "DSC_2.jpg",
    );
    fireEvent.pointerDown(container.querySelector(".photo-canvas")!);
    fireEvent.keyDown(document.body, { key: "f" });
    await waitFor(() =>
      expect(api.updatePhotos).toHaveBeenCalledWith(
        "test-project",
        ["photo-2"],
        { decision: "favorite", reviewed: true },
      ),
    );
    expect(stored.photos[3].decision).toBe("undecided");
  });
  it("adds bulk tags without erasing different existing tags and undoes in one atomic call", async () => {
    await openWorkspace();
    fireEvent.click(
      screen.getByRole("button", { name: "DSC_0.jpg, undecided" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "DSC_1.jpg, undecided" }),
      { ctrlKey: true },
    );
    const tags = screen.getByLabelText("Tags") as HTMLTextAreaElement;
    expect(tags.value).toBe("");
    expect(screen.getByText(/existing tags are kept/)).toBeTruthy();
    fireEvent.change(tags, { target: { value: "print" } });
    fireEvent.blur(tags);
    await waitFor(() =>
      expect(api.updatePhotoPatches).toHaveBeenCalledTimes(1),
    );
    expect(stored.photos[0].tags).toEqual(["family", "print"]);
    expect(stored.photos[1].tags).toEqual(["travel", "print"]);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Undo" }).hasAttribute("disabled"),
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() =>
      expect(api.updatePhotoPatches).toHaveBeenCalledTimes(2),
    );
    expect(stored.photos[0].tags).toEqual(["family"]);
    expect(stored.photos[1].tags).toEqual(["travel"]);
  });
  it("closes only the top dialog on Escape and returns to the export dialog", async () => {
    stored.photos[0].decision = "favorite";
    await openWorkspace();
    fireEvent.click(screen.getByRole("button", { name: "Export selection" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Locate the Lightroom plugin ↗" }),
    );
    await screen.findByRole("dialog", { name: "Connect Lightroom Classic." });
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(
      screen.queryByRole("dialog", { name: "Connect Lightroom Classic." }),
    ).toBeNull();
    expect(
      screen.getByRole("dialog", { name: "Take your selection to Lightroom." }),
    ).toBeTruthy();
  });
  it("bounds initial rendering for a large library and omits singleton suggestions", async () => {
    stored = testProject(2200);
    stored.groups = stored.photos.map((photo) => ({
      id: photo.id,
      label: photo.filename,
      photoIds: [photo.id],
      recommendedPhotoIds: [photo.id],
    }));
    const { container } = await openWorkspace();
    expect(container.querySelectorAll(".photo-card")).toHaveLength(180);
    expect(container.querySelectorAll(".recommendation-badge")).toHaveLength(0);
    expect(container.querySelectorAll(".moment-list button")).toHaveLength(0);
    fireEvent.keyDown(document.body, { key: "a", ctrlKey: true });
    expect(screen.getByText("2200 photos")).toBeTruthy();
  });
});
