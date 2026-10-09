import { describe, expect, it } from "vitest";
import {
  mergePhotos,
  nextPhotoId,
  parseTags,
  previousPatch,
  visiblePhotos,
} from "./domain";
import type { BrowseOptions, Photo, Project } from "./types";
const photograph = (id: string, changes: Partial<Photo> = {}): Photo => ({
  id,
  path: `/photos/${id}.jpg`,
  filename: `${id}.jpg`,
  previewPath: "",
  thumbnailPath: "",
  detailPath: null,
  captureTime: "2026-09-01T08:00:00Z",
  width: 6000,
  height: 4000,
  camera: "Nikon Z6",
  groupId: null,
  qualityScore: 70,
  hints: [],
  rating: 0,
  ratingTouched: false,
  decision: "undecided",
  decisionSource: "manual",
  decisionTouched: false,
  suggestedDecision: null,
  suggestionReason: null,
  suggestionConfidence: null,
  reviewed: false,
  tags: [],
  analysisError: null,
  ...changes,
});
const photos = [
  photograph("a", {
    decision: "favorite",
    reviewed: true,
    tags: ["family"],
    rating: 4,
    ratingTouched: true,
  }),
  photograph("b", {
    decision: "pass",
    reviewed: true,
    qualityScore: 98,
    captureTime: "2026-09-01T09:00:00Z",
  }),
  photograph("c", { tags: ["Travel"], captureTime: "2026-09-01T10:00:00Z" }),
];
const project: Project = {
  id: "p",
  name: "Trip",
  sourceDir: "/photos",
  includeSubfolders: true,
  automaticSelectionEnabled: false,
  selectionMode: "cautious",
  firstPassReady: false,
  projectPath: "/p.db",
  createdAt: "",
  updatedAt: "",
  photos,
  groups: [
    {
      id: "moment",
      label: "Moment",
      photoIds: ["a", "b"],
      recommendedPhotoIds: ["b"],
    },
  ],
  collections: [{ id: "book", name: "Photo book", photoIds: ["a", "c"] }],
  importStatus: "completed",
  importError: null,
};
const options: BrowseOptions = {
  filter: "all",
  sort: "time",
  query: "",
  scope: { type: "all" },
};
describe("review navigation", () => {
  it("keeps analysis recommendations out of user favorites", () => {
    expect(
      visiblePhotos(project, { ...options, filter: "favorite" }).map(
        (photo) => photo.id,
      ),
    ).toEqual(["a"]);
  });
  it("combines group, review status, and case-insensitive tag/camera searches", () => {
    expect(
      visiblePhotos(project, {
        ...options,
        scope: { type: "group", id: "moment" },
        query: "family",
      }).map((photo) => photo.id),
    ).toEqual(["a"]);
    expect(
      visiblePhotos(project, {
        ...options,
        scope: { type: "collection", id: "book" },
        filter: "unreviewed",
        query: "TRAVEL",
      }).map((photo) => photo.id),
    ).toEqual(["c"]);
    expect(visiblePhotos(project, { ...options, query: "nikon" })).toHaveLength(
      3,
    );
    expect(
      visiblePhotos(project, {
        ...options,
        scope: { type: "group", id: "missing" },
      }),
    ).toEqual([]);
  });
  it("uses the displayed order for navigation and safely handles a filtered-out active photograph", () => {
    const sorted = visiblePhotos(project, { ...options, sort: "quality" });
    expect(sorted.map((photo) => photo.id)).toEqual(["b", "a", "c"]);
    expect(nextPhotoId(sorted, "b", 1)).toBe("a");
    expect(nextPhotoId(sorted, "b", -1)).toBe("b");
    expect(nextPhotoId(sorted, "c", 1)).toBe("c");
    expect(nextPhotoId(sorted, "removed", 1)).toBe("b");
    expect(nextPhotoId([], null, 1)).toBeNull();
  });
});
describe("manual edits and undo", () => {
  it("restores different prior decisions and ratings per photo, including never-rated state", () => {
    const patches = photos.slice(0, 2).map((photo) =>
      previousPatch(photo, {
        rating: 5,
        decision: "favorite",
        reviewed: true,
      }),
    );
    expect(patches[0]).toEqual({
      rating: 4,
      ratingTouched: true,
      decision: "favorite",
      reviewed: true,
      decisionSource: "manual",
      decisionTouched: false,
    });
    expect(patches[1]).toEqual({
      rating: 0,
      ratingTouched: false,
      decision: "pass",
      reviewed: true,
      decisionSource: "manual",
      decisionTouched: false,
    });
  });
  it("takes a copy of tags for undo and merges saves without overwriting unrelated photos", () => {
    const before = previousPatch(photos[0], { tags: ["print"] });
    expect(before.tags).toEqual(["family"]);
    expect(before.tags).not.toBe(photos[0].tags);
    const saved = { ...photos[1], decision: "favorite" as const };
    const updated = mergePhotos(project, [saved]);
    expect(updated.photos[0]).toBe(photos[0]);
    expect(updated.photos[1]).toBe(saved);
    expect(project.photos[1].decision).toBe("pass");
  });
  it("trims and deduplicates tags without inventing labels", () => {
    expect(parseTags(" family, print , family, , travel ")).toEqual([
      "family",
      "print",
      "travel",
    ]);
  });
});
