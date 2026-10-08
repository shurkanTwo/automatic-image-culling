import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { demoInvoke } from "./demo";
import type {
  AppState,
  Collection,
  ImportProgress,
  Photo,
  PhotoPatch,
  Project,
} from "./types";

type Bridge = {
  invoke: <T>(command: string, args: Record<string, unknown>) => Promise<T>;
  listen?: (
    event: string,
    callback: (payload: unknown) => void,
  ) => Promise<() => void>;
  chooseFolder?: () => Promise<string | null>;
  chooseProject?: () => Promise<string | null>;
  savePath?: (name: string) => Promise<string | null>;
};
declare global {
  interface Window {
    __PHOTO_SELECT_BRIDGE__?: Bridge;
    __TAURI_INTERNALS__?: unknown;
  }
}
export const isDemo =
  import.meta.env.VITE_DEMO === "true" ||
  new URLSearchParams(location.search).get("demo") === "1";
export const isNative = Boolean(window.__TAURI_INTERNALS__);
export const isAvailable =
  isNative || isDemo || Boolean(window.__PHOTO_SELECT_BRIDGE__);
function call<T>(
  command: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  if (window.__PHOTO_SELECT_BRIDGE__)
    return window.__PHOTO_SELECT_BRIDGE__.invoke<T>(command, args);
  if (isNative) return invoke<T>(command, args);
  if (isDemo) return demoInvoke<T>(command, args);
  return Promise.reject(
    new Error(
      "Open Photo Select in the desktop app to work with your photographs.",
    ),
  );
}
export const api = {
  state: () => call<AppState>("get_app_state"),
  createProject: (name: string, sourceDir: string) =>
    call<Project>("create_project", { name, sourceDir }),
  openProject: (projectPath: string) =>
    call<Project>("open_project", { projectPath }),
  project: (projectId: string) => call<Project>("get_project", { projectId }),
  start: (projectId: string) =>
    call<{ jobId: string }>("start_import", { projectId }),
  cancel: (projectId: string) => call<void>("cancel_import", { projectId }),
  updatePhoto: (projectId: string, photoId: string, patch: PhotoPatch) =>
    call<Photo>("update_photo", { projectId, photoId, patch }),
  updatePhotos: (projectId: string, photoIds: string[], patch: PhotoPatch) =>
    call<Photo[]>("update_photos", { projectId, photoIds, patch }),
  createCollection: (projectId: string, name: string) =>
    call<Collection>("create_collection", { projectId, name }),
  updateCollection: (
    projectId: string,
    collectionId: string,
    name: string | null,
    photoIds: string[] | null,
  ) =>
    call<Collection>("update_collection", {
      projectId,
      collectionId,
      name,
      photoIds,
    }),
  deleteCollection: (projectId: string, collectionId: string) =>
    call<void>("delete_collection", { projectId, collectionId }),
  detail: (projectId: string, photoId: string) =>
    call<{ detailPath: string; width: number; height: number }>(
      "generate_detail",
      { projectId, photoId },
    ),
  export: (
    projectId: string,
    destination: string,
    collectionId: string | null,
    onlyFavorites: boolean,
  ) =>
    call<{ path: string; count: number }>("export_selection", {
      projectId,
      destination,
      collectionId,
      onlyFavorites,
    }),
  plugin: () => call<string>("get_lightroom_plugin_path"),
};
export function photoSrc(path: string | null): string {
  if (!path) return "";
  if (/^(https?:|data:|blob:|\/demo\/)/.test(path)) return path;
  return isNative ? convertFileSrc(path) : path;
}
export async function chooseFolder(): Promise<string | null> {
  if (window.__PHOTO_SELECT_BRIDGE__?.chooseFolder)
    return window.__PHOTO_SELECT_BRIDGE__.chooseFolder();
  if (isDemo) return "/Demo/Alpine weekend";
  const value = await open({
    directory: true,
    multiple: false,
    title: "Choose a photo folder",
  });
  return typeof value === "string" ? value : null;
}
export async function chooseProject(): Promise<string | null> {
  if (window.__PHOTO_SELECT_BRIDGE__?.chooseProject)
    return window.__PHOTO_SELECT_BRIDGE__.chooseProject();
  if (isDemo) return "/Demo/alpine.photoselect";
  const value = await open({
    multiple: false,
    title: "Open a Photo Select project",
    filters: [
      {
        name: "Photo Select project",
        extensions: ["cullproj", "sqlite", "db", "photoselect"],
      },
    ],
  });
  return typeof value === "string" ? value : null;
}
export async function exportPath(name: string): Promise<string | null> {
  if (window.__PHOTO_SELECT_BRIDGE__?.savePath)
    return window.__PHOTO_SELECT_BRIDGE__.savePath(name);
  if (isDemo) return "photo-select-demo.json";
  return save({
    title: "Export selection for Lightroom",
    defaultPath: `${name.replace(/[<>:"/\\|?*]/g, "-")}-selection.json`,
    filters: [{ name: "Lightroom selection", extensions: ["json"] }],
  });
}
export async function revealPath(path: string): Promise<void> {
  if (isNative) return revealItemInDir(path);
  throw new Error("Folder reveal is available in the desktop app.");
}
export async function subscribe<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<() => void> {
  if (window.__PHOTO_SELECT_BRIDGE__?.listen)
    return window.__PHOTO_SELECT_BRIDGE__.listen(event, (value) =>
      handler(value as T),
    );
  if (!isNative) return () => {};
  return listen<T>(event, (value) => handler(value.payload));
}
export function progressLabel(value: ImportProgress): string {
  if (value.phase === "scan") return "Finding photographs";
  if (value.phase === "grouping") return "Organizing moments";
  if (value.phase === "complete") return "Analysis complete";
  if (value.phase === "cancelled") return "Analysis paused";
  if (value.phase === "error") return "Analysis needs attention";
  return "Checking photographs";
}
