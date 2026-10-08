import type { BrowseOptions, Photo, PhotoPatch, Project } from "./types";

export function visiblePhotos(
  project: Project,
  options: BrowseOptions,
): Photo[] {
  const query = options.query.trim().toLocaleLowerCase();
  const scopeId = options.scope.type === "all" ? null : options.scope.id;
  const scopeIds =
    options.scope.type === "group"
      ? project.groups.find((group) => group.id === scopeId)?.photoIds
      : options.scope.type === "collection"
        ? project.collections.find((collection) => collection.id === scopeId)
            ?.photoIds
        : undefined;
  const scope = scopeIds ? new Set(scopeIds) : null;
  return project.photos
    .filter((photo) => {
      if (options.scope.type !== "all" && (!scope || !scope.has(photo.id)))
        return false;
      if (options.filter === "favorite" && photo.decision !== "favorite")
        return false;
      if (options.filter === "pass" && photo.decision !== "pass") return false;
      if (options.filter === "unreviewed" && photo.reviewed) return false;
      return (
        !query ||
        [photo.filename, photo.camera ?? "", ...photo.tags]
          .join(" ")
          .toLocaleLowerCase()
          .includes(query)
      );
    })
    .sort((a, b) => {
      if (options.sort === "quality")
        return (
          b.qualityScore - a.qualityScore ||
          a.captureTime.localeCompare(b.captureTime) ||
          a.filename.localeCompare(b.filename)
        );
      if (options.sort === "rating")
        return (
          b.rating - a.rating ||
          a.captureTime.localeCompare(b.captureTime) ||
          a.filename.localeCompare(b.filename)
        );
      return (
        a.captureTime.localeCompare(b.captureTime) ||
        a.filename.localeCompare(b.filename)
      );
    });
}

export function nextPhotoId(
  photos: Photo[],
  activeId: string | null,
  direction: number,
): string | null {
  if (!photos.length) return null;
  const index = photos.findIndex((photo) => photo.id === activeId);
  if (index < 0) return photos[0].id;
  return photos[Math.max(0, Math.min(photos.length - 1, index + direction))].id;
}

export function comparisonPhotos(
  photos: Photo[],
  selectedIds: string[],
  activeId: string | null,
): Photo[] {
  const selected = new Set(selectedIds);
  const chosen = photos.filter((photo) => selected.has(photo.id));
  if (chosen.length >= 2) return chosen.slice(0, 4);
  const activeIndex = Math.max(
    0,
    photos.findIndex((photo) => photo.id === activeId),
  );
  const start = Math.max(0, Math.min(activeIndex, photos.length - 2));
  return photos.slice(start, start + 2);
}

export function previousPatch(photo: Photo, patch: PhotoPatch): PhotoPatch {
  const result: PhotoPatch = {};
  if (patch.decision !== undefined) result.decision = photo.decision;
  if (patch.rating !== undefined) {
    result.rating = photo.rating;
    result.ratingTouched = photo.ratingTouched;
  }
  if (patch.tags !== undefined) result.tags = [...photo.tags];
  if (patch.reviewed !== undefined) result.reviewed = photo.reviewed;
  return result;
}

export function patchChangesPhoto(photo: Photo, patch: PhotoPatch): boolean {
  if (patch.rating !== undefined && patch.rating !== photo.rating) return true;
  const touched =
    patch.ratingTouched ??
    (patch.rating !== undefined ? true : photo.ratingTouched);
  if (touched !== photo.ratingTouched) return true;
  if (patch.decision !== undefined && patch.decision !== photo.decision)
    return true;
  if (patch.reviewed !== undefined && patch.reviewed !== photo.reviewed)
    return true;
  return (
    patch.tags !== undefined &&
    (patch.tags.length !== photo.tags.length ||
      patch.tags.some((tag, index) => tag !== photo.tags[index]))
  );
}

export function mergePhotos(project: Project, changed: Photo[]): Project {
  const updates = new Map(changed.map((photo) => [photo.id, photo]));
  return {
    ...project,
    photos: project.photos.map((photo) => updates.get(photo.id) ?? photo),
  };
}

export function parseTags(value: string): string[] {
  return [
    ...new Set(
      value
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
    ),
  ];
}
