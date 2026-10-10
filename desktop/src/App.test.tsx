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
    automaticFirstPass: vi.fn(),
    clearAutomaticSelection: vi.fn(),
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
    version: "0.2.4",
    engineAvailable: true,
    projects: [
      {
        id: stored.id,
        name: stored.name,
        sourceDir: stored.sourceDir,
        includeSubfolders: stored.includeSubfolders,
        preferRaw: stored.preferRaw,
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
    async (
      name,
      sourceDir,
      includeSubfolders,
      automaticSelectionEnabled = true,
      selectionMode = "cautious",
      preferRaw = false,
    ) => {
      stored = {
        ...testProject(),
        name,
        sourceDir,
        includeSubfolders,
        automaticSelectionEnabled,
        selectionMode,
        preferRaw,
      };
      return structuredClone(stored);
    },
  );
  vi.mocked(api.start).mockResolvedValue({ jobId: "test-job" });
  vi.mocked(api.automaticFirstPass).mockImplementation(
    async (_id, selectionMode) => {
      stored.automaticSelectionEnabled = true;
      stored.selectionMode = selectionMode;
      stored.photos = stored.photos.map((photo) =>
        !photo.reviewed &&
        !photo.ratingTouched &&
        !photo.decisionTouched &&
        !photo.analysisError &&
        photo.suggestedDecision &&
        (photo.decisionSource === "automatic" || photo.decision === "undecided")
          ? {
              ...photo,
              decision: photo.suggestedDecision,
              decisionSource: "automatic",
            }
          : photo,
      );
      return structuredClone(stored);
    },
  );
  vi.mocked(api.clearAutomaticSelection).mockImplementation(async () => {
    stored.automaticSelectionEnabled = false;
    stored.photos = stored.photos.map((photo) =>
      photo.decisionSource === "automatic"
        ? { ...photo, decision: "undecided", decisionSource: "manual" }
        : photo,
    );
    return structuredClone(stored);
  });
  vi.mocked(api.export).mockImplementation(
    async (_id, path, collectionId, onlyFavorites, includeDiscards) => {
      const collection = stored.collections.find(
        (value) => value.id === collectionId,
      );
      const selectedCount = stored.photos.filter(
        (photo) =>
          photo.decision !== "pass" &&
          (!onlyFavorites || photo.decision === "favorite") &&
          (!collection || collection.photoIds.includes(photo.id)),
      ).length;
      const discardCount = includeDiscards
        ? stored.photos.filter((photo) => photo.decision === "pass").length
        : 0;
      return {
        path,
        count: selectedCount + discardCount,
        selectedCount,
        discardCount,
      };
    },
  );
  vi.mocked(api.updatePhotos).mockImplementation(
    async (_id, ids, patch: PhotoPatch) => {
      const targets = new Set(ids);
      stored.photos = stored.photos.map((photo) =>
        targets.has(photo.id)
          ? {
              ...photo,
              ...patch,
              decisionSource: patch.decisionSource ?? "manual",
              decisionTouched: patch.decisionTouched ?? true,
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
      return patch
        ? {
            ...photo,
            ...patch,
            decisionSource: patch.decisionSource ?? "manual",
            decisionTouched: patch.decisionTouched ?? true,
          }
        : photo;
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
  await waitFor(() =>
    expect(
      rendered.container.querySelector(".photo-card.focused"),
    ).toBeTruthy(),
  );
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
describe("automatic grid loading", () => {
  let observers: GridObserver[];
  class GridObserver {
    target: Element | null = null;
    observe = vi.fn((target: Element) => {
      this.target = target;
    });
    disconnect = vi.fn();
    constructor(
      readonly callback: IntersectionObserverCallback,
      readonly options: IntersectionObserverInit,
    ) {
      observers.push(this);
    }
    intersect(isIntersecting = true) {
      this.callback(
        [{ target: this.target!, isIntersecting } as IntersectionObserverEntry],
        this as unknown as IntersectionObserver,
      );
    }
  }
  beforeEach(() => {
    observers = [];
    vi.stubGlobal("IntersectionObserver", GridObserver);
  });
  afterEach(() => vi.unstubAllGlobals());
  function loadPage() {
    act(() => observers.at(-1)!.intersect());
  }
  it("prefetches 180-photo pages inside the scrolling grid without moving the selection or duplicating a page", async () => {
    stored = testProject(450);
    const { container } = await openWorkspace();
    const grid = container.querySelector(".grid-scroll") as HTMLDivElement;
    const first = observers.at(-1)!;
    expect(first.options.root).toBe(grid);
    expect(first.options.rootMargin).toBe("0px 0px 600px 0px");
    expect(first.target).toBe(grid.querySelector(".grid-sentinel"));
    expect(container.querySelectorAll(".photo-card")).toHaveLength(180);
    expect(screen.queryByRole("button", { name: /Show next/ })).toBeNull();
    act(() => first.intersect(false));
    expect(container.querySelectorAll(".photo-card")).toHaveLength(180);
    grid.scrollTop = 1400;
    vi.mocked(HTMLElement.prototype.scrollIntoView).mockClear();
    act(() => {
      first.intersect();
      first.intersect();
    });
    expect(container.querySelectorAll(".photo-card")).toHaveLength(360);
    expect(first.disconnect).toHaveBeenCalledTimes(1);
    expect(grid.scrollTop).toBe(1400);
    expect(HTMLElement.prototype.scrollIntoView).not.toHaveBeenCalled();
    act(() => first.intersect());
    expect(container.querySelectorAll(".photo-card")).toHaveLength(360);
    const final = observers.at(-1)!;
    loadPage();
    expect(container.querySelectorAll(".photo-card")).toHaveLength(450);
    expect(container.querySelector(".grid-sentinel")).toBeNull();
    expect(final.disconnect).toHaveBeenCalledTimes(1);
    act(() => final.intersect());
    expect(container.querySelectorAll(".photo-card")).toHaveLength(450);
  });
  it.each(["filter", "scope", "search", "sort"])(
    "resets pages after a %s change and ignores stale observers",
    async (change) => {
      stored = testProject(500);
      stored.photos.slice(0, 400).forEach((photo) => {
        photo.decision = "favorite";
      });
      stored.collections[0].photoIds = stored.photos
        .slice(0, 400)
        .map((photo) => photo.id);
      const { container } = await openWorkspace();
      const sidebar = within(
        container.querySelector(".sidebar") as HTMLElement,
      );
      const count = () =>
        container.querySelectorAll(".grid-scroll .photo-card").length;
      loadPage();
      expect(count()).toBe(360);
      const stale = observers.at(-1)!;
      if (change === "filter")
        fireEvent.click(sidebar.getByRole("button", { name: /^Favorites/ }));
      else if (change === "scope")
        fireEvent.click(sidebar.getByRole("button", { name: /^Photo book/ }));
      else if (change === "search")
        fireEvent.change(
          screen.getByLabelText("Search photos, tags, or camera"),
          { target: { value: "Nikon" } },
        );
      else
        fireEvent.change(screen.getByLabelText("Sort photographs"), {
          target: { value: "quality" },
        });
      expect(count()).toBe(180);
      expect(stale.disconnect).toHaveBeenCalledTimes(1);
      act(() => stale.intersect());
      expect(count()).toBe(180);
      loadPage();
      expect(count()).toBe(360);
    },
  );
  it("resets on view changes and safely restarts after an empty search", async () => {
    stored = testProject(450);
    const { container } = await openWorkspace();
    const controls = within(
      container.querySelector(".browse-controls") as HTMLElement,
    );
    const count = () =>
      container.querySelectorAll(".grid-scroll .photo-card").length;
    loadPage();
    const stale = observers.at(-1)!;
    fireEvent.click(
      controls.getByRole("button", { name: "Single photo view" }),
    );
    expect(stale.disconnect).toHaveBeenCalledTimes(1);
    act(() => stale.intersect());
    fireEvent.click(controls.getByRole("button", { name: "Grid view" }));
    expect(count()).toBe(180);
    const beforeEmpty = observers.at(-1)!;
    fireEvent.change(screen.getByLabelText("Search photos, tags, or camera"), {
      target: { value: "no matching photo" },
    });
    expect(container.querySelector(".grid-scroll")).toBeNull();
    expect(beforeEmpty.disconnect).toHaveBeenCalledTimes(1);
    act(() => beforeEmpty.intersect());
    fireEvent.click(controls.getByRole("button", { name: "Clear search" }));
    expect(count()).toBe(180);
  });
  it("disconnects on project close and starts a reopened project at the first page", async () => {
    stored = testProject(500);
    const { container, unmount } = await openWorkspace();
    loadPage();
    const stale = observers.at(-1)!;
    fireEvent.click(
      within(container.querySelector(".sidebar") as HTMLElement).getByRole(
        "button",
        { name: "Projects" },
      ),
    );
    await screen.findByRole("button", { name: /Review trip.*photographs/ });
    expect(stale.disconnect).toHaveBeenCalledTimes(1);
    act(() => stale.intersect());
    stored = { ...testProject(500), id: "second-project" };
    fireEvent.click(
      screen.getByRole("button", { name: /Review trip.*photographs/ }),
    );
    await screen.findByRole("button", { name: /^DSC_0.jpg,/ });
    expect(container.querySelectorAll(".grid-scroll .photo-card")).toHaveLength(
      180,
    );
    const current = observers.at(-1)!;
    unmount();
    expect(current.disconnect).toHaveBeenCalledTimes(1);
  });
  it("reveals a photograph reached by keyboard beyond the rendered page and keeps it available when returning from the viewer", async () => {
    stored = testProject(450);
    const { container } = await openWorkspace();
    fireEvent.click(
      screen.getByRole("button", { name: "DSC_179.jpg, undecided" }),
    );
    vi.mocked(HTMLElement.prototype.scrollIntoView).mockClear();
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    expect(container.querySelectorAll(".grid-scroll .photo-card")).toHaveLength(
      360,
    );
    expect(
      container.querySelector(".photo-card.focused")?.textContent,
    ).toContain("DSC_180.jpg");
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Single photo view" }));
    fireEvent.click(screen.getByRole("button", { name: "Grid view" }));
    expect(
      container.querySelector(".grid-scroll .photo-card.focused")?.textContent,
    ).toContain("DSC_180.jpg");
  });
  it("automatically loads on scroll and resize when IntersectionObserver is unavailable and removes fallback listeners", async () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    stored = testProject(450);
    const { container, unmount } = await openWorkspace();
    const grid = container.querySelector(".grid-scroll") as HTMLDivElement;
    const removeScroll = vi.spyOn(grid, "removeEventListener");
    const removeResize = vi.spyOn(window, "removeEventListener");
    let contentHeight: number | null = null;
    Object.defineProperties(grid, {
      clientHeight: { configurable: true, value: 800 },
      scrollHeight: {
        configurable: true,
        get: () =>
          contentHeight ??
          (grid.querySelectorAll(".photo-card").length / 180) * 6000,
      },
    });
    grid.scrollTop = 4599;
    fireEvent.scroll(grid);
    expect(container.querySelectorAll(".photo-card")).toHaveLength(180);
    grid.scrollTop = 4600;
    fireEvent.scroll(grid);
    expect(container.querySelectorAll(".photo-card")).toHaveLength(360);
    grid.scrollTop = 10600;
    fireEvent.scroll(grid);
    expect(container.querySelectorAll(".photo-card")).toHaveLength(450);
    expect(container.querySelector(".grid-sentinel")).toBeNull();
    expect(removeScroll).toHaveBeenCalledWith("scroll", expect.any(Function));
    expect(removeResize).toHaveBeenCalledWith("resize", expect.any(Function));
    fireEvent.change(screen.getByLabelText("Sort photographs"), {
      target: { value: "quality" },
    });
    expect(container.querySelectorAll(".photo-card")).toHaveLength(180);
    contentHeight = 1200;
    fireEvent(window, new Event("resize"));
    expect(container.querySelectorAll(".photo-card")).toHaveLength(450);
    unmount();
    vi.restoreAllMocks();
  });
  it("fills a short viewport automatically when IntersectionObserver is unavailable", async () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    stored = testProject(450);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(
      function (this: HTMLElement) {
        return this.classList.contains("grid-scroll") ? 800 : 0;
      },
    );
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(
      function (this: HTMLElement) {
        if (!this.classList.contains("grid-scroll")) return 0;
        return this.querySelectorAll(".photo-card").length <= 180 ? 500 : 2500;
      },
    );
    try {
      const { container } = await openWorkspace();
      await waitFor(() =>
        expect(container.querySelectorAll(".photo-card")).toHaveLength(360),
      );
      expect(container.querySelector(".grid-sentinel")).toBeTruthy();
    } finally {
      vi.restoreAllMocks();
    }
  });
});
describe("automatic first pass and Lightroom discards", () => {
  function proposals() {
    stored.automaticSelectionEnabled = true;
    stored.firstPassReady = true;
    Object.assign(stored.photos[0], {
      decision: "favorite",
      decisionSource: "automatic",
      suggestedDecision: "favorite",
      suggestionReason: "Strong detail and balanced exposure in this moment.",
      suggestionConfidence: 0.96,
    });
    Object.assign(stored.photos[1], {
      decision: "pass",
      decisionSource: "automatic",
      suggestedDecision: "pass",
      suggestionReason:
        "Substantially weaker detail than a near-matching photograph.",
      suggestionConfidence: 0.94,
    });
    Object.assign(stored.photos[2], {
      decision: "favorite",
      decisionSource: "manual",
      decisionTouched: true,
      reviewed: true,
      rating: 4,
      ratingTouched: true,
    });
  }
  it("enables the first pass by default and sends the chosen import mode", async () => {
    const dialog = await chooseImportFolder();
    expect(
      (
        within(dialog).getByRole("checkbox", {
          name: "Automatic first pass",
        }) as HTMLInputElement
      ).checked,
    ).toBe(true);
    fireEvent.change(within(dialog).getByLabelText("Import selection mode"), {
      target: { value: "stronger" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await waitFor(() =>
      expect(api.createProject).toHaveBeenCalledWith(
        "New trip",
        "/photos/New trip",
        false,
        true,
        "stronger",
        false,
      ),
    );
    await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
    expect(api.automaticFirstPass).not.toHaveBeenCalled();
  });
  it("allows an import without automatic choices", async () => {
    const dialog = await chooseImportFolder();
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: "Automatic first pass" }),
    );
    expect(
      (
        within(dialog).getByLabelText(
          "Import selection mode",
        ) as HTMLSelectElement
      ).disabled,
    ).toBe(true);
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await waitFor(() =>
      expect(api.createProject).toHaveBeenCalledWith(
        "New trip",
        "/photos/New trip",
        false,
        false,
        "cautious",
        false,
      ),
    );
  });
  it("explains automatic choices, lets manual review override them, clears only automatic choices, and reruns either mode", async () => {
    proposals();
    const { container } = await openWorkspace();
    expect(
      within(container.querySelector(".photo-grid") as HTMLElement).getByText(
        "Auto favorite",
      ),
    ).toBeTruthy();
    expect(screen.getByText("Auto discard")).toBeTruthy();
    expect(
      screen.getByText("Strong detail and balanced exposure in this moment."),
    ).toBeTruthy();
    expect(screen.getByText("Confidence: 96%")).toBeTruthy();
    expect(container.querySelector(".selection-dot")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "DSC_1.jpg, auto discard" }),
    );
    fireEvent.keyDown(document.body, { key: "f" });
    await waitFor(() => expect(stored.photos[1].decisionSource).toBe("manual"));
    expect(stored.photos[1].reviewed).toBe(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Undo automatic choices" }),
    );
    await waitFor(() =>
      expect(api.clearAutomaticSelection).toHaveBeenCalledWith("test-project"),
    );
    await screen.findByText("0 automatic choices · Off for rescans");
    expect(stored.photos[0].decision).toBe("undecided");
    expect(stored.photos[1].decision).toBe("favorite");
    expect(stored.photos[2].rating).toBe(4);
    expect(stored.photos[2].decisionSource).toBe("manual");
    fireEvent.change(screen.getByLabelText("Automatic first-pass mode"), {
      target: { value: "stronger" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Run automatic first pass" }),
    );
    await waitFor(() =>
      expect(api.automaticFirstPass).toHaveBeenCalledWith(
        "test-project",
        "stronger",
      ),
    );
    await screen.findByText("1 automatic choice · Enabled for rescans");
    expect(stored.photos[1].decision).toBe("favorite");
    expect(stored.photos[2].rating).toBe(4);
    expect(stored.photos[0].reviewed).toBe(false);
    fireEvent.change(screen.getByLabelText("Automatic first-pass mode"), {
      target: { value: "cautious" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Run automatic first pass" }),
    );
    await waitFor(() =>
      expect(api.automaticFirstPass).toHaveBeenLastCalledWith(
        "test-project",
        "cautious",
      ),
    );
  });
  it("drains a focused metadata draft before running and prevents a duplicate action", async () => {
    let resolveSave!: (photos: Project["photos"]) => void;
    vi.mocked(api.updatePhotos).mockImplementationOnce((_id, ids, patch) => {
      stored.photos = stored.photos.map((photo) =>
        ids.includes(photo.id) ? { ...photo, ...patch } : photo,
      );
      return new Promise((resolve) => {
        resolveSave = resolve;
      });
    });
    await openWorkspace();
    const tags = screen.getByLabelText("Tags");
    act(() => tags.focus());
    fireEvent.change(tags, { target: { value: "family, print" } });
    const run = screen.getByRole("button", {
      name: "Run automatic first pass",
    });
    fireEvent.click(run);
    fireEvent.click(run);
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(1));
    expect(api.automaticFirstPass).not.toHaveBeenCalled();
    await act(async () => resolveSave(structuredClone([stored.photos[0]])));
    await waitFor(() =>
      expect(api.automaticFirstPass).toHaveBeenCalledTimes(1),
    );
    expect(stored.photos[0].tags).toEqual(["family", "print"]);
  });
  it("starts analysis for an uncached project and disables automatic commands while it runs", async () => {
    vi.mocked(api.automaticFirstPass).mockImplementationOnce(async () => ({
      ...stored,
      importStatus: "running",
    }));
    await openWorkspace();
    fireEvent.click(
      screen.getByRole("button", { name: "Run automatic first pass" }),
    );
    await screen.findByText("Preparing your photographs");
    expect(
      screen
        .getByRole("button", { name: "Run automatic first pass" })
        .hasAttribute("disabled"),
    ).toBe(true);
    expect(
      (screen.getByLabelText("Automatic first-pass mode") as HTMLSelectElement)
        .disabled,
    ).toBe(true);
  });
  it.each([true, false])(
    "exports Lightroom Rejects with includeDiscards=%s without removing originals",
    async (includeDiscards) => {
      proposals();
      await openWorkspace();
      fireEvent.click(screen.getByRole("button", { name: "Export selection" }));
      const dialog = screen.getByRole("dialog", {
        name: "Take your selection to Lightroom.",
      });
      const rejects = within(dialog).getByRole("checkbox", {
        name: "Include discards as Lightroom Rejects",
      });
      expect((rejects as HTMLInputElement).checked).toBe(true);
      expect(dialog.textContent).toContain("Favorites become Lightroom Picks");
      expect(dialog.textContent).toContain("No original files are deleted");
      if (!includeDiscards) fireEvent.click(rejects);
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Save selection file" }),
      );
      await waitFor(() =>
        expect(api.export).toHaveBeenCalledWith(
          "test-project",
          "/export.json",
          null,
          true,
          includeDiscards,
        ),
      );
      await screen.findByText(
        `Exported 2 selected photographs and ${includeDiscards ? 1 : 0} Lightroom Rejects to /export.json`,
      );
      expect(stored.photos).toHaveLength(4);
    },
  );
  it("exports a reject-only selection and keeps an empty shortlist disabled when discards are excluded", async () => {
    stored.photos[0].decision = "pass";
    await openWorkspace();
    fireEvent.click(screen.getByRole("button", { name: "Export selection" }));
    const dialog = screen.getByRole("dialog");
    const save = within(dialog).getByRole("button", {
      name: "Save selection file",
    });
    expect(save.hasAttribute("disabled")).toBe(false);
    fireEvent.click(
      within(dialog).getByRole("checkbox", {
        name: "Include discards as Lightroom Rejects",
      }),
    );
    expect(save.hasAttribute("disabled")).toBe(true);
  });
});
describe("folder import scope", () => {
  it.each([false, true])(
    "saves preferRaw=%s at import and keeps the saved choice through rescans",
    async (preferRaw) => {
      const dialog = await chooseImportFolder();
      const preference = within(dialog).getByRole("checkbox", {
        name: "Prefer RAW when a matching JPEG exists",
      }) as HTMLInputElement;
      expect(preference.checked).toBe(false);
      expect(preference.getAttribute("aria-describedby")).toBe(
        "import-raw-preference",
      );
      expect(
        document.getElementById("import-raw-preference")?.textContent,
      ).toContain("in the same folder");
      expect(dialog.textContent).toContain("JPEG-only photos remain included");
      expect(dialog.textContent).toContain("stays as a fallback");
      if (preferRaw) fireEvent.click(preference);
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Start import" }),
      );
      await waitFor(() =>
        expect(api.createProject).toHaveBeenCalledExactlyOnceWith(
          "New trip",
          "/photos/New trip",
          false,
          true,
          "cautious",
          preferRaw,
        ),
      );
      const choice = preferRaw
        ? "RAW preferred for pairs"
        : "RAW + JPEG separately";
      await screen.findByText(choice);
      expect(screen.getByText(choice).getAttribute("title")).toBe(
        "Create a new project to change folder scope or RAW preference.",
      );
      expect(
        screen.queryByRole("checkbox", {
          name: "Prefer RAW when a matching JPEG exists",
        }),
      ).toBeNull();
      const rescan = screen.getByRole("button", { name: "Rescan folder" });
      await waitFor(() => expect(rescan.hasAttribute("disabled")).toBe(false));
      fireEvent.click(rescan);
      await waitFor(() => expect(api.start).toHaveBeenCalledTimes(2));
      expect(stored.preferRaw).toBe(preferRaw);
      expect(screen.getByText(choice)).toBeTruthy();
      expect(api.createProject).toHaveBeenCalledTimes(1);
    },
  );
  it("shows the separate RAW and JPEG default when opening a legacy project", async () => {
    delete (stored as Partial<Project>).preferRaw;
    await openWorkspace();
    expect(screen.getByText("RAW + JPEG separately")).toBeTruthy();
    expect(screen.queryByText("RAW preferred for pairs")).toBeNull();
    expect(api.createProject).not.toHaveBeenCalled();
  });
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
        true,
        "cautious",
        false,
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
    fireEvent.click(
      within(dialog).getByRole("checkbox", {
        name: "Prefer RAW when a matching JPEG exists",
      }),
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
    expect(
      (
        screen.getByRole("checkbox", {
          name: "Prefer RAW when a matching JPEG exists",
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
    const rawPreference = within(dialog).getByRole("checkbox", {
      name: "Prefer RAW when a matching JPEG exists",
    }) as HTMLInputElement;
    fireEvent.click(rawPreference);
    vi.mocked(api.createProject).mockRejectedValueOnce(
      new Error("Disk is full"),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Start import" }),
    );
    await within(dialog).findByRole("alert");
    expect(dialog.textContent).toContain("Disk is full");
    expect(checkbox.checked).toBe(true);
    expect(rawPreference.checked).toBe(true);
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
    expect(rawPreference.disabled).toBe(true);
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
  it("keeps grid and filmstrip selection on the card frame without covering thumbnails", async () => {
    stored.photos[1].decision = "favorite";
    stored.photos[2].decision = "pass";
    stored.photos[3].analysisError = "Preview generation failed";
    stored.groups = [
      {
        id: "moment",
        label: "Similar moment",
        photoIds: ["photo-0", "photo-1"],
        recommendedPhotoIds: ["photo-1"],
      },
    ];
    const { container } = await openWorkspace();
    const grid = container.querySelector(".photo-grid") as HTMLElement;
    const first = within(grid).getByRole("button", {
      name: "DSC_0.jpg, undecided",
    });
    expect(first.classList.contains("focused")).toBe(true);
    expect(first.classList.contains("selected")).toBe(false);
    expect(first.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(first);
    fireEvent.click(
      within(grid).getByRole("button", {
        name: "DSC_1.jpg, favorite",
      }),
      { ctrlKey: true },
    );
    for (const card of grid.querySelectorAll(".photo-card.selected"))
      expect(card.getAttribute("aria-pressed")).toBe("true");
    expect(grid.querySelectorAll(".photo-card.selected")).toHaveLength(2);
    expect(first.classList.contains("selected")).toBe(true);
    expect(first.classList.contains("focused")).toBe(false);
    expect(grid.querySelector(".selection-dot")).toBeNull();
    for (const badge of ["favorite", "pass", "recommendation", "error"])
      expect(grid.querySelector(`.${badge}-badge`)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Single photo view" }));
    const filmstrip = container.querySelector(".filmstrip") as HTMLElement;
    expect(filmstrip.querySelectorAll(".photo-card.selected")).toHaveLength(2);
    expect(
      within(filmstrip)
        .getByRole("button", {
          name: "DSC_0.jpg, undecided, selected",
        })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    fireEvent.click(
      within(filmstrip).getByRole("button", {
        name: "DSC_2.jpg, discard",
      }),
    );
    const selected = within(filmstrip).getByRole("button", {
      name: "DSC_2.jpg, discard, selected",
    });
    expect(selected.classList.contains("selected")).toBe(true);
    expect(selected.getAttribute("aria-pressed")).toBe("true");
    expect(filmstrip.querySelectorAll(".photo-card.selected")).toHaveLength(1);
    expect(
      within(filmstrip)
        .getByRole("button", {
          name: "DSC_0.jpg, undecided",
        })
        .getAttribute("aria-pressed"),
    ).toBe("false");
    expect(filmstrip.querySelector(".selection-dot")).toBeNull();
    for (const badge of ["favorite", "pass", "recommendation", "error"])
      expect(filmstrip.querySelector(`.${badge}-badge`)).toBeTruthy();
  });
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
      decisionSource: "manual",
      decisionTouched: true,
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
        {
          decision: "favorite",
          reviewed: true,
          decisionSource: "manual",
          decisionTouched: true,
        },
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
