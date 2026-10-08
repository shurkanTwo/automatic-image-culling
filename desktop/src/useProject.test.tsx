// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Photo, PhotoPatch, Project } from "./types";
import { useProject } from "./useProject";
import { api } from "./api";

vi.mock("./api", () => ({
  api: { updatePhotos: vi.fn(), updatePhoto: vi.fn(), project: vi.fn() },
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
  reviewed: false,
  tags: ["family"],
  analysisError: null,
};
const project: Project = {
  id: "project",
  name: "Family",
  sourceDir: "/photos",
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
    let refreshTask!: Promise<void>;
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
    let firstTask!: Promise<void>;
    let queuedTask!: Promise<void>;
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
    let firstTask!: Promise<void>;
    let secondTask!: Promise<void>;
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
    vi.mocked(api.updatePhoto).mockResolvedValueOnce({
      ...original,
      decision: "favorite",
      reviewed: true,
    });
    await act(async () => {
      await result.current.undo();
    });
    expect(api.updatePhoto).toHaveBeenLastCalledWith("project", "photo", {
      decision: "favorite",
      reviewed: true,
    });
    expect(result.current.project?.photos[0].decision).toBe("favorite");
  });
  it("undoes the first manual rating to an unrated photo, preserving analysis and tags", async () => {
    vi.mocked(api.updatePhotos).mockImplementation(
      async (_projectId, _ids, patch) => [
        { ...original, ...patch, ratingTouched: true },
      ],
    );
    vi.mocked(api.updatePhoto).mockImplementation(
      async (_projectId, _id, patch: PhotoPatch) => ({ ...original, ...patch }),
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
    expect(api.updatePhoto).toHaveBeenCalledWith("project", "photo", {
      rating: 0,
      ratingTouched: false,
      reviewed: false,
    });
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
