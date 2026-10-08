import { useCallback, useEffect, useRef, useState } from "react";
import { api, subscribe } from "./api";
import { mergePhotos, previousPatch } from "./domain";
import type {
  Collection,
  ImportProgress,
  Photo,
  PhotoPatch,
  Project,
} from "./types";

type UndoEntry =
  | { label: string; changes: { photoId: string; patch: PhotoPatch }[] }
  | { label: string; collection: Collection };
export function useProject() {
  const [project, setProjectState] = useState<Project | null>(null);
  const projectRef = useRef<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(0);
  const pending = useRef(0);
  const editRevision = useRef(0);
  const queue = useRef(Promise.resolve());
  const undoRef = useRef<UndoEntry[]>([]);
  const [undoLabel, setUndoLabel] = useState<string | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const setProject = useCallback((value: Project | null) => {
    projectRef.current = value;
    setProjectState(value);
  }, []);
  const load = useCallback(
    (value: Project | null) => {
      editRevision.current += 1;
      setProject(value);
      undoRef.current = [];
      setUndoLabel(null);
      setProgress(null);
      setError(null);
    },
    [setProject],
  );
  const refresh = useCallback(async () => {
    const current = projectRef.current;
    if (!current) return;
    if (pending.current) {
      refreshTimer.current = setTimeout(() => {
        refreshTimer.current = undefined;
        void refresh();
      }, 350);
      return;
    }
    const requestedRevision = editRevision.current;
    try {
      const value = await api.project(current.id);
      if (
        projectRef.current?.id === current.id &&
        !pending.current &&
        editRevision.current === requestedRevision
      )
        setProject(value);
    } catch (reason) {
      setError(String(reason));
    }
  }, [setProject]);
  const scheduleRefresh = useCallback(() => {
    if (refreshTimer.current) return;
    refreshTimer.current = setTimeout(() => {
      refreshTimer.current = undefined;
      void refresh();
    }, 700);
  }, [refresh]);
  useEffect(() => {
    let disposed = false;
    const cleanups: (() => void)[] = [];
    void Promise.all([
      subscribe<ImportProgress>("import-progress", (value) => {
        if (value.projectId !== projectRef.current?.id) return;
        setProgress(value);
        if (value.phase === "error" && value.message) setError(value.message);
        scheduleRefresh();
      }),
      subscribe<{ projectId: string }>("project-updated", (value) => {
        if (value.projectId === projectRef.current?.id) scheduleRefresh();
      }),
    ])
      .then((values) => {
        if (disposed) values.forEach((cleanup) => cleanup());
        else cleanups.push(...values);
      })
      .catch((reason) =>
        setError(`Could not subscribe to project updates: ${String(reason)}`),
      );
    return () => {
      disposed = true;
      cleanups.forEach((cleanup) => cleanup());
      if (refreshTimer.current) clearTimeout(refreshTimer.current);
    };
  }, [scheduleRefresh]);
  function enqueue(action: () => Promise<void>): Promise<void> {
    const intendedProjectId = projectRef.current?.id;
    editRevision.current += 1;
    pending.current += 1;
    setSaving(pending.current);
    const task = queue.current
      .then(async () => {
        if (projectRef.current?.id === intendedProjectId) await action();
      })
      .catch((reason) => {
        if (projectRef.current?.id === intendedProjectId)
          setError(String(reason));
      })
      .finally(() => {
        pending.current -= 1;
        setSaving(pending.current);
      });
    queue.current = task;
    return task;
  }
  function pushUndo(entry: UndoEntry) {
    undoRef.current = [...undoRef.current.slice(-49), entry];
    setUndoLabel(entry.label);
  }
  function edit(
    ids: string[],
    patch: PhotoPatch,
    label: string,
  ): Promise<void> {
    return enqueue(async () => {
      const current = projectRef.current;
      if (!current || !ids.length) return;
      const before = current.photos.filter((photo) => ids.includes(photo.id));
      if (!before.length) return;
      const optimistic: Photo[] = before.map((photo) => ({
        ...photo,
        ...patch,
        ratingTouched:
          patch.ratingTouched ??
          (patch.rating !== undefined ? true : photo.ratingTouched),
      }));
      setProject(mergePhotos(current, optimistic));
      try {
        const changed = await api.updatePhotos(
          current.id,
          before.map((photo) => photo.id),
          patch,
        );
        if (projectRef.current?.id !== current.id) return;
        setProject(mergePhotos(projectRef.current, changed));
        pushUndo({
          label,
          changes: before.map((photo) => ({
            photoId: photo.id,
            patch: previousPatch(photo, patch),
          })),
        });
      } catch (reason) {
        if (projectRef.current?.id === current.id)
          setProject(mergePhotos(projectRef.current, before));
        throw reason;
      }
    });
  }
  function undo(): Promise<void> {
    return enqueue(async () => {
      const current = projectRef.current;
      const entry = undoRef.current.at(-1);
      if (!current || !entry) return;
      if ("changes" in entry) {
        const changed: Photo[] = [];
        // Restore independently because batch edits can have different prior ratings and decisions.
        for (const change of entry.changes)
          changed.push(
            await api.updatePhoto(current.id, change.photoId, change.patch),
          );
        if (projectRef.current?.id !== current.id) return;
        setProject(mergePhotos(projectRef.current, changed));
      } else {
        const updated = await api.updateCollection(
          current.id,
          entry.collection.id,
          entry.collection.name,
          entry.collection.photoIds,
        );
        if (projectRef.current?.id !== current.id) return;
        setProject({
          ...projectRef.current,
          collections: projectRef.current!.collections.map((value) =>
            value.id === updated.id ? updated : value,
          ),
        });
      }
      undoRef.current.pop();
      setUndoLabel(undoRef.current.at(-1)?.label ?? null);
    });
  }
  function createCollection(name: string): Promise<void> {
    return enqueue(async () => {
      const current = projectRef.current;
      if (!current) return;
      const value = await api.createCollection(current.id, name);
      if (projectRef.current?.id !== current.id) return;
      setProject({
        ...projectRef.current!,
        collections: [...projectRef.current!.collections, value],
      });
    });
  }
  function collectionEdit(
    id: string,
    ids: string[],
    remove = false,
  ): Promise<void> {
    return enqueue(async () => {
      const current = projectRef.current;
      const collection = current?.collections.find((value) => value.id === id);
      if (!current || !collection) return;
      const photoIds = remove
        ? collection.photoIds.filter((photoId) => !ids.includes(photoId))
        : [...new Set([...collection.photoIds, ...ids])];
      const updated = await api.updateCollection(
        current.id,
        id,
        null,
        photoIds,
      );
      if (projectRef.current?.id !== current.id) return;
      setProject({
        ...projectRef.current!,
        collections: projectRef.current!.collections.map((value) =>
          value.id === id ? updated : value,
        ),
      });
      pushUndo({
        label: remove ? "Remove from collection" : "Add to collection",
        collection: { ...collection, photoIds: [...collection.photoIds] },
      });
    });
  }
  return {
    project,
    load,
    refresh,
    progress,
    error,
    setError,
    saving,
    undoLabel,
    edit,
    undo,
    createCollection,
    collectionEdit,
  };
}
