import { useCallback, useEffect, useRef, useState } from "react";
import { api, subscribe } from "./api";
import { mergePhotos, patchChangesPhoto, previousPatch } from "./domain";
import type {
  Collection,
  ImportProgress,
  Photo,
  PhotoPatch,
  Project,
  SelectionMode,
} from "./types";

type UndoEntry =
  | { label: string; changes: { photoId: string; patch: PhotoPatch }[] }
  | { label: string; collection: Collection };
export function useProject() {
  const [project, setProjectState] = useState<Project | null>(null);
  const projectRef = useRef<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(0);
  const [saveError, setSaveError] = useState<string | null>(null);
  const saveErrorRef = useRef<string | null>(null);
  const projectSession = useRef(0);
  const pending = useRef(0);
  const editRevision = useRef(0);
  const queue = useRef(Promise.resolve());
  const saveFailures = useRef(0);
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
      projectSession.current += 1;
      setSaveError(null);
      saveErrorRef.current = null;
      if (refreshTimer.current) clearTimeout(refreshTimer.current);
      refreshTimer.current = undefined;
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
      if (!refreshTimer.current)
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
      if (
        projectRef.current?.id === current.id &&
        editRevision.current === requestedRevision
      )
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
  function enqueue(action: () => Promise<void>): Promise<boolean> {
    const intendedProjectId = projectRef.current?.id;
    const intendedSession = projectSession.current;
    editRevision.current += 1;
    pending.current += 1;
    setSaving(pending.current);
    const task = queue.current
      .then(async () => {
        if (
          projectSession.current !== intendedSession ||
          projectRef.current?.id !== intendedProjectId
        )
          return false;
        await action();
        return true;
      })
      .catch((reason) => {
        saveFailures.current += 1;
        if (
          projectSession.current === intendedSession &&
          projectRef.current?.id === intendedProjectId
        ) {
          setError(String(reason));
          setSaveError(String(reason));
          saveErrorRef.current = String(reason);
        }
        return false;
      })
      .finally(() => {
        pending.current -= 1;
        setSaving(pending.current);
      });
    queue.current = task.then(() => undefined);
    return task;
  }
  const hasPendingSaves = useCallback(() => pending.current > 0, []);
  const waitForSaves = useCallback(async (): Promise<boolean> => {
    const initialFailures = saveFailures.current;
    // Check the live queue again after each drain in case another edit was added while saving.
    while (pending.current > 0) await queue.current;
    return saveFailures.current === initialFailures;
  }, []);
  function pushUndo(entry: UndoEntry) {
    undoRef.current = [...undoRef.current.slice(-49), entry];
    setUndoLabel(entry.label);
  }
  function savedSuccessfully() {
    const previousError = saveErrorRef.current;
    saveErrorRef.current = null;
    setSaveError(null);
    setError((current) => (current === previousError ? null : current));
  }
  function edit(
    ids: string[],
    patch: PhotoPatch,
    label: string,
  ): Promise<boolean> {
    patch = {
      ...patch,
      decisionSource: patch.decisionSource ?? "manual",
      decisionTouched: patch.decisionTouched ?? true,
    };
    return enqueue(async () => {
      const current = projectRef.current;
      if (!current || !ids.length) return;
      const targets = new Set(ids);
      const before = current.photos.filter(
        (photo) => targets.has(photo.id) && patchChangesPhoto(photo, patch),
      );
      const session = projectSession.current;
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
        if (
          projectRef.current?.id !== current.id ||
          projectSession.current !== session
        )
          return;
        savedSuccessfully();
        setProject(mergePhotos(projectRef.current, changed));
        pushUndo({
          label,
          changes: before.map((photo) => ({
            photoId: photo.id,
            patch: previousPatch(photo, patch),
          })),
        });
      } catch (reason) {
        if (
          projectRef.current?.id === current.id &&
          projectSession.current === session
        )
          setProject(mergePhotos(projectRef.current, before));
        throw reason;
      }
    });
  }
  function automaticFirstPass(
    mode: SelectionMode,
    clear = false,
  ): Promise<boolean> {
    return enqueue(async () => {
      const current = projectRef.current;
      const session = projectSession.current;
      if (!current || current.importStatus === "running") return;
      const updated = clear
        ? await api.clearAutomaticSelection(current.id)
        : await api.automaticFirstPass(current.id, mode);
      if (
        projectRef.current?.id !== current.id ||
        projectSession.current !== session
      )
        return;
      savedSuccessfully();
      setProject(updated);
    });
  }
  function undo(): Promise<boolean> {
    return enqueue(async () => {
      const current = projectRef.current;
      const session = projectSession.current;
      const entry = undoRef.current.at(-1);
      if (!current || !entry) return;
      if ("changes" in entry) {
        const changed = await api.updatePhotoPatches(current.id, entry.changes);
        if (
          projectRef.current?.id !== current.id ||
          projectSession.current !== session
        )
          return;
        savedSuccessfully();
        setProject(mergePhotos(projectRef.current, changed));
      } else {
        const updated = await api.updateCollection(
          current.id,
          entry.collection.id,
          entry.collection.name,
          entry.collection.photoIds,
        );
        if (
          projectRef.current?.id !== current.id ||
          projectSession.current !== session
        )
          return;
        setProject({
          ...projectRef.current,
          collections: projectRef.current!.collections.map((value) =>
            value.id === updated.id ? updated : value,
          ),
        });
      }
      savedSuccessfully();
      undoRef.current.pop();
      setUndoLabel(undoRef.current.at(-1)?.label ?? null);
    });
  }
  function addTags(ids: string[], tags: string[]): Promise<boolean> {
    return enqueue(async () => {
      const current = projectRef.current;
      const session = projectSession.current;
      if (!current || !tags.length) return;
      const targets = new Set(ids);
      const before = current.photos.filter(
        (photo) =>
          targets.has(photo.id) &&
          patchChangesPhoto(photo, {
            tags: [...new Set([...photo.tags, ...tags])],
          }),
      );
      const updates = before.map((photo) => ({
        photoId: photo.id,
        patch: {
          tags: [...new Set([...photo.tags, ...tags])],
          decisionSource: "manual" as const,
          decisionTouched: true,
        },
      }));
      if (!updates.length) return;
      setProject(
        mergePhotos(
          current,
          before.map((photo, index) => ({ ...photo, ...updates[index].patch })),
        ),
      );
      try {
        const changed = await api.updatePhotoPatches(current.id, updates);
        if (
          projectRef.current?.id !== current.id ||
          projectSession.current !== session
        )
          return;
        savedSuccessfully();
        setProject(mergePhotos(projectRef.current, changed));
        pushUndo({
          label: "Add tags",
          changes: before.map((photo) => ({
            photoId: photo.id,
            patch: previousPatch(photo, { tags: [...photo.tags] }),
          })),
        });
      } catch (reason) {
        if (
          projectRef.current?.id === current.id &&
          projectSession.current === session
        )
          setProject(mergePhotos(projectRef.current, before));
        throw reason;
      }
    });
  }
  function createCollection(name: string): Promise<boolean> {
    return enqueue(async () => {
      const current = projectRef.current;
      const session = projectSession.current;
      if (!current) return;
      const value = await api.createCollection(current.id, name);
      if (
        projectRef.current?.id !== current.id ||
        projectSession.current !== session
      )
        return;
      savedSuccessfully();
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
  ): Promise<boolean> {
    return enqueue(async () => {
      const current = projectRef.current;
      const session = projectSession.current;
      const collection = current?.collections.find((value) => value.id === id);
      if (!current || !collection) return;
      const photoIds = remove
        ? collection.photoIds.filter((photoId) => !ids.includes(photoId))
        : [...new Set([...collection.photoIds, ...ids])];
      if (photoIds.length === collection.photoIds.length) return;
      const updated = await api.updateCollection(
        current.id,
        id,
        null,
        photoIds,
      );
      if (
        projectRef.current?.id !== current.id ||
        projectSession.current !== session
      )
        return;
      savedSuccessfully();
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
    saveError,
    hasPendingSaves,
    waitForSaves,
    undoLabel,
    edit,
    automaticFirstPass,
    addTags,
    undo,
    createCollection,
    collectionEdit,
  };
}
