// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Photo, PhotoPatch, Project } from "./types";
import { useProject } from "./useProject";
import { api } from "./api";

vi.mock("./api", () => ({
  api: {
    updatePhotos: vi.fn(),
    updatePhoto: vi.fn(),
    updatePhotoPatches: vi.fn(),
    project: vi.fn(),
    automaticFirstPass: vi.fn(),
    clearAutomaticSelection: vi.fn(),
  },
  subscribe: vi.fn(async () => () => {}),
}));
const original: Photo = {
  id: "photo",
  filename: "family.jpg",
  path: "/photos/family.jpg",
  previewPath: "",
  thumbnailPath: "",
  detailPath: null,
  captureTime: "2026-09-01T08:00:00Z",
  width: 6000,
  height: 4000,
  camera: null,
  groupId: null,
  qualityScore: 98,
  hints: ["Good exposure"],
  rating: 0,
  ratingTouched: false,
  decision: "undecided",
  decisionSource: "manual",
  decisionTouched: false,
  suggestedDecision: null,
  suggestionReason: null,
  suggestionConfidence: null,
  reviewed: false,
  tags: ["family"],
  analysisError: null,
};
const project: Project = {
  id: "project",
  name: "Family",
  sourceDir: "/photos",
  includeSubfolders: true,
  automaticSelectionEnabled: false,
  selectionMode: "cautious",
  firstPassReady: false,
  projectPath: "/project.cullproj",
  createdAt: "",
  updatedAt: "",
  photos: [original],
  groups: [],
  collections: [],
  importStatus: "completed",
  importError: null,
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
beforeEach(() => {
  vi.clearAllMocks();
});
describe("durable manual review", () => {
  it.each(["decision", "rating", "tags", "bulk tags"])(
    "restores automatic provenance and eligibility after undoing a %s edit",
    async (kind) => {
      const automatic: Photo = {
        ...original,
        decision: "favorite",
        decisionSource: "automatic",
        decisionTouched: false,
        suggestedDecision: "favorite",
        suggestionReason: "Strong detail",
        suggestionConfidence: 0.95,
      };
      let snapshot = [structuredClone(automatic)];
      vi.mocked(api.updatePhotos).mockImplementation(
        async (_id, ids, patch) => {
          snapshot = snapshot.map((photo) =>
            ids.includes(photo.id)
              ? {
                  ...photo,
                  ...patch,
                  ratingTouched:
                    patch.rating !== undefined ? true : photo.ratingTouched,
                }
              : photo,
          );
          return structuredClone(snapshot);
        },
      );
      vi.mocked(api.updatePhotoPatches).mockImplementation(
        async (_id, updates) => {
          snapshot = snapshot.map((photo) => ({
            ...photo,
            ...updates.find((update) => update.photoId === photo.id)?.patch,
          }));
          return structuredClone(snapshot);
        },
      );
      const { result } = renderHook(() => useProject());
      act(() => result.current.load({ ...project, photos: [automatic] }));
      await act(async () => {
        if (kind === "bulk tags")
          await result.current.addTags(["photo"], ["print"]);
        else
          await result.current.edit(
            ["photo"],
            kind === "decision"
              ? { decision: "pass", reviewed: true }
              : kind === "rating"
                ? { rating: 4, reviewed: true }
                : { tags: ["print"] },
            "Manual edit",
          );
      });
      expect(result.current.project?.photos[0].decisionSource).toBe("manual");
      expect(result.current.project?.photos[0].decisionTouched).toBe(true);
      await act(async () => {
        await result.current.undo();
      });
      expect(api.updatePhotoPatches).toHaveBeenLastCalledWith("project", [
        {
          photoId: "photo",
          patch: expect.objectContaining({
            decisionSource: "automatic",
            decisionTouched: false,
          }),
        },
      ]);
      expect(result.current.project?.photos[0]).toEqual(automatic);
    },
  );
  it("records an explicit Undecided override, ignores a repeated override, and restores untouched eligibility on undo", async () => {
    let snapshot = structuredClone(original);
    vi.mocked(api.updatePhotos).mockImplementation(async (_id, _ids, patch) => {
      snapshot = { ...snapshot, ...patch };
      return [structuredClone(snapshot)];
    });
    vi.mocked(api.updatePhotoPatches).mockImplementation(
      async (_id, updates) => {
        snapshot = { ...snapshot, ...updates[0].patch };
        return [structuredClone(snapshot)];
      },
    );
    const { result } = renderHook(() => useProject());
    act(() => result.current.load(structuredClone(project)));
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { decision: "undecided", reviewed: false },
        "Undecided",
      );
    });
    expect(result.current.project?.photos[0].decisionTouched).toBe(true);
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { decision: "undecided", reviewed: false },
        "Undecided",
      );
    });
    expect(api.updatePhotos).toHaveBeenCalledTimes(1);
    await act(async () => {
      await result.current.undo();
    });
    expect(result.current.project?.photos[0]).toEqual(original);
  });
  it("ignores an automatic first-pass response after another project has loaded", async () => {
    const requested = deferred<Project>();
    vi.mocked(api.automaticFirstPass).mockReturnValueOnce(requested.promise);
    const { result } = renderHook(() => useProject());
    act(() => result.current.load(structuredClone(project)));
    let task!: Promise<boolean>;
    act(() => {
      task = result.current.automaticFirstPass("stronger");
    });
    await waitFor(() =>
      expect(api.automaticFirstPass).toHaveBeenCalledWith(
        "project",
        "stronger",
      ),
    );
    const other = { ...project, id: "other-project" };
    act(() => result.current.load(other));
    await act(async () => {
      requested.resolve({ ...project, automaticSelectionEnabled: true });
      await task;
    });
    expect(result.current.project?.id).toBe("other-project");
    expect(result.current.project?.automaticSelectionEnabled).toBe(false);
  });
  it("does not manufacture writes or undo entries for repeated identical review decisions", async () => {
    const favorite = {
      ...original,
      decision: "favorite" as const,
      reviewed: true,
      decisionTouched: true,
    };
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load({ ...structuredClone(project), photos: [favorite] });
    });
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { decision: "favorite", reviewed: true },
        "Favorite",
      );
    });
    expect(api.updatePhotos).not.toHaveBeenCalled();
    expect(result.current.undoLabel).toBeNull();
    vi.mocked(api.updatePhotos).mockResolvedValueOnce([
      { ...favorite, ratingTouched: true },
    ]);
    await act(async () => {
      await result.current.edit(["photo"], { rating: 0 }, "Clear rating");
    });
    expect(api.updatePhotos).toHaveBeenCalledTimes(1);
    expect(result.current.project?.photos[0].ratingTouched).toBe(true);
  });
  it("keeps a failed atomic batch undo retryable without showing a partial restoration", async () => {
    const second = { ...original, id: "second", tags: ["travel"] };
    const edited = [
      { ...original, decision: "favorite" as const },
      { ...second, decision: "favorite" as const },
    ];
    vi.mocked(api.updatePhotos).mockResolvedValueOnce(edited);
    vi.mocked(api.updatePhotoPatches)
      .mockRejectedValueOnce(new Error("Transaction failed"))
      .mockResolvedValueOnce([original, second]);
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load({
        ...structuredClone(project),
        photos: [original, second],
      });
    });
    await act(async () => {
      await result.current.edit(
        ["photo", "second"],
        { decision: "favorite" },
        "Favorite",
      );
    });
    let undone!: boolean;
    await act(async () => {
      undone = await result.current.undo();
    });
    expect(undone).toBe(false);
    expect(result.current.project?.photos).toEqual(edited);
    expect(result.current.undoLabel).toBe("Favorite");
    expect(result.current.saveError).toContain("Transaction failed");
    expect(api.updatePhotoPatches).toHaveBeenCalledWith("project", [
      {
        photoId: "photo",
        patch: {
          decision: "undecided",
          decisionSource: "manual",
          decisionTouched: false,
        },
      },
      {
        photoId: "second",
        patch: {
          decision: "undecided",
          decisionSource: "manual",
          decisionTouched: false,
        },
      },
    ]);
    await act(async () => {
      undone = await result.current.undo();
    });
    expect(undone).toBe(true);
    expect(result.current.project?.photos).toEqual([original, second]);
    expect(result.current.undoLabel).toBeNull();
    expect(result.current.saveError).toBeNull();
  });
  it("rolls back every optimistic bulk tag addition if the atomic write fails", async () => {
    const second = { ...original, id: "second", tags: ["travel"] };
    vi.mocked(api.updatePhotoPatches).mockRejectedValueOnce(
      new Error("Write failed"),
    );
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load({
        ...structuredClone(project),
        photos: [original, second],
      });
    });
    let saved!: boolean;
    await act(async () => {
      saved = await result.current.addTags(["photo", "second"], ["print"]);
    });
    expect(saved).toBe(false);
    expect(result.current.project?.photos).toEqual([original, second]);
    expect(result.current.undoLabel).toBeNull();
    expect(api.updatePhotoPatches).toHaveBeenCalledWith("project", [
      {
        photoId: "photo",
        patch: {
          tags: ["family", "print"],
          decisionSource: "manual",
          decisionTouched: true,
        },
      },
      {
        photoId: "second",
        patch: {
          tags: ["travel", "print"],
          decisionSource: "manual",
          decisionTouched: true,
        },
      },
    ]);
  });
  it("drains saves added during closing and reports a write failure instead of allowing close", async () => {
    const first = deferred<Photo[]>();
    const second = deferred<Photo[]>();
    vi.mocked(api.updatePhotos)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    let drained!: Promise<boolean>;
    act(() => {
      void result.current.edit(["photo"], { rating: 3 }, "Rating");
      drained = result.current.waitForSaves();
    });
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(1));
    let resolved = false;
    void drained.then(() => {
      resolved = true;
    });
    act(() => {
      void result.current.edit(["photo"], { rating: 5 }, "Rating");
    });
    await act(async () => {
      first.resolve([{ ...original, rating: 3, ratingTouched: true }]);
      await first.promise;
    });
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(2));
    expect(resolved).toBe(false);
    expect(result.current.hasPendingSaves()).toBe(true);
    let saved!: boolean;
    await act(async () => {
      second.resolve([{ ...original, rating: 5, ratingTouched: true }]);
      saved = await drained;
    });
    expect(saved).toBe(true);
    expect(result.current.project?.photos[0].rating).toBe(5);
    expect(result.current.hasPendingSaves()).toBe(false);
    vi.mocked(api.updatePhotos).mockRejectedValueOnce(
      new Error("Cannot write project"),
    );
    await act(async () => {
      void result.current.edit(["photo"], { rating: 1 }, "Rating");
      saved = await result.current.waitForSaves();
    });
    expect(saved).toBe(false);
    expect(result.current.error).toContain("Cannot write project");
    expect(result.current.project?.photos[0].rating).toBe(5);
  });
  it("ignores a refresh that was requested before a completed manual save", async () => {
    const oldSnapshot = deferred<Project>();
    vi.mocked(api.project).mockReturnValueOnce(oldSnapshot.promise);
    vi.mocked(api.updatePhotos).mockResolvedValueOnce([
      { ...original, decision: "favorite", reviewed: true },
    ]);
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    let refreshTask!: Promise<unknown>;
    act(() => {
      refreshTask = result.current.refresh();
    });
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { decision: "favorite", reviewed: true },
        "Favorite",
      );
    });
    await act(async () => {
      oldSnapshot.resolve(structuredClone(project));
      await refreshTask;
    });
    expect(result.current.project?.photos[0].decision).toBe("favorite");
    expect(result.current.project?.photos[0].reviewed).toBe(true);
  });
  it("keeps outstanding saves and queued edits attached to their original project", async () => {
    const first = deferred<Photo[]>();
    vi.mocked(api.updatePhotos).mockReturnValueOnce(first.promise);
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    let firstTask!: Promise<unknown>;
    let queuedTask!: Promise<unknown>;
    act(() => {
      firstTask = result.current.edit(
        ["photo"],
        { decision: "favorite" },
        "Favorite",
      );
      queuedTask = result.current.edit(["photo"], { decision: "pass" }, "Pass");
    });
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(1));
    const other = {
      ...structuredClone(project),
      id: "other-project",
      name: "Another event",
    };
    act(() => {
      result.current.load(other);
    });
    await act(async () => {
      first.resolve([{ ...original, decision: "favorite" }]);
      await firstTask;
      await queuedTask;
    });
    expect(api.updatePhotos).toHaveBeenCalledTimes(1);
    expect(result.current.project).toEqual(other);
    expect(result.current.undoLabel).toBeNull();
    expect(result.current.saving).toBe(0);
  });
  it("serializes rapid decisions so a slow first save cannot overwrite the last choice", async () => {
    const first = deferred<Photo[]>();
    const second = deferred<Photo[]>();
    vi.mocked(api.updatePhotos)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    let firstTask!: Promise<unknown>;
    let secondTask!: Promise<unknown>;
    act(() => {
      firstTask = result.current.edit(
        ["photo"],
        { decision: "favorite", reviewed: true },
        "Favorite",
      );
      secondTask = result.current.edit(
        ["photo"],
        { decision: "pass", reviewed: true },
        "Pass",
      );
    });
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(1));
    expect(result.current.saving).toBe(2);
    expect(result.current.project?.photos[0].decision).toBe("favorite");
    await act(async () => {
      first.resolve([{ ...original, decision: "favorite", reviewed: true }]);
      await firstTask;
    });
    await waitFor(() => expect(api.updatePhotos).toHaveBeenCalledTimes(2));
    expect(result.current.project?.photos[0].decision).toBe("pass");
    await act(async () => {
      second.resolve([{ ...original, decision: "pass", reviewed: true }]);
      await secondTask;
    });
    expect(result.current.project?.photos[0].decision).toBe("pass");
    expect(result.current.saving).toBe(0);
    expect(result.current.undoLabel).toBe("Pass");
    vi.mocked(api.updatePhotoPatches).mockResolvedValueOnce([
      {
        ...original,
        decision: "favorite",
        reviewed: true,
      },
    ]);
    await act(async () => {
      await result.current.undo();
    });
    expect(api.updatePhotoPatches).toHaveBeenLastCalledWith("project", [
      {
        photoId: "photo",
        patch: {
          decision: "favorite",
          decisionSource: "manual",
          decisionTouched: false,
          reviewed: true,
        },
      },
    ]);
    expect(result.current.project?.photos[0].decision).toBe("favorite");
  });
  it("undoes the first manual rating to an unrated photo, preserving analysis and tags", async () => {
    vi.mocked(api.updatePhotos).mockImplementation(
      async (_projectId, _ids, patch) => [
        { ...original, ...patch, ratingTouched: true },
      ],
    );
    vi.mocked(api.updatePhotoPatches).mockImplementation(
      async (_projectId, updates) =>
        updates.map((update) => ({ ...original, ...update.patch })),
    );
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { rating: 4, reviewed: true },
        "Rate photographs",
      );
    });
    expect(result.current.project?.photos[0].ratingTouched).toBe(true);
    await act(async () => {
      await result.current.undo();
    });
    expect(api.updatePhotoPatches).toHaveBeenCalledWith("project", [
      {
        photoId: "photo",
        patch: {
          rating: 0,
          ratingTouched: false,
          reviewed: false,
          decisionSource: "manual",
          decisionTouched: false,
        },
      },
    ]);
    expect(result.current.project?.photos[0]).toEqual(original);
    expect(result.current.undoLabel).toBeNull();
  });
  it("rolls back a failed save and shows the storage error without recording a false undo", async () => {
    vi.mocked(api.updatePhotos).mockRejectedValue(new Error("Disk is full"));
    const { result } = renderHook(() => useProject());
    act(() => {
      result.current.load(structuredClone(project));
    });
    await act(async () => {
      await result.current.edit(
        ["photo"],
        { decision: "favorite", reviewed: true },
        "Favorite",
      );
    });
    expect(result.current.project?.photos[0]).toEqual(original);
    expect(result.current.error).toContain("Disk is full");
    expect(result.current.undoLabel).toBeNull();
    expect(result.current.saving).toBe(0);
  });
});
