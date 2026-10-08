import { useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  Aperture,
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Circle,
  Clock3,
  Download,
  FolderOpen,
  FolderPlus,
  Grid2X2,
  Heart,
  Image,
  Layers,
  LoaderCircle,
  Minus,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  Sparkles,
  Undo2,
  X,
} from "lucide-react";
import {
  api,
  chooseFolder,
  chooseProject,
  exportPath,
  isAvailable,
  isDemo,
  isNative,
  progressLabel,
  revealPath,
} from "./api";
import { comparisonPhotos, nextPhotoId, visiblePhotos } from "./domain";
import { commitDrafts } from "./drafts";
import type {
  AppState,
  BrowseOptions,
  Filter,
  Photo,
  PhotoPatch,
  Scope,
  Sort,
} from "./types";
import { useProject } from "./useProject";
import PhotoCard from "./components/PhotoCard";
import Viewer from "./components/Viewer";
import Inspector from "./components/Inspector";

type View = "grid" | "single" | "compare";
const filters: { id: Filter; label: string; icon: typeof Image }[] = [
  { id: "all", label: "All photographs", icon: Image },
  { id: "favorite", label: "Favorites", icon: Heart },
  { id: "unreviewed", label: "Unreviewed", icon: Circle },
  { id: "pass", label: "Passed", icon: Minus },
];
export default function App() {
  const workspace = useProject();
  const {
    project,
    load,
    progress,
    error,
    setError,
    saving,
    saveError,
    hasPendingSaves,
    waitForSaves,
    undoLabel,
    edit,
    addTags,
    undo,
    createCollection,
    collectionEdit,
  } = workspace;
  useEffect(() => {
    if (!isNative) return;
    let disposed = false;
    let closing = false;
    let unlisten: (() => void) | undefined;
    const window = getCurrentWindow();
    void window
      .onCloseRequested(async (event) => {
        commitDrafts();
        if (!closing && !hasPendingSaves()) return;
        event.preventDefault();
        if (closing) return;
        closing = true;
        try {
          const saved = await waitForSaves();
          closing = false;
          if (saved && !disposed) await window.close();
        } catch (reason) {
          closing = false;
          setError(`Could not finish closing Photo Select: ${String(reason)}`);
        }
      })
      .then((cleanup) => {
        if (disposed) cleanup();
        else unlisten = cleanup;
      })
      .catch((reason) =>
        setError(`Could not protect pending saves on close: ${String(reason)}`),
      );
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [hasPendingSaves, waitForSaves, setError]);
  const [state, setState] = useState<AppState | null>(null);
  const [loading, setLoading] = useState(isAvailable);
  const [options, setOptions] = useState<BrowseOptions>({
    filter: "all",
    sort: "time",
    query: "",
    scope: { type: "all" },
  });
  const [view, setView] = useState<View>("grid");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [renderLimit, setRenderLimit] = useState(180);
  const [collectionName, setCollectionName] = useState<string | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportTarget, setExportTarget] = useState("favorites");
  const [exporting, setExporting] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [pluginPath, setPluginPath] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [importAction, setImportAction] = useState(false);
  const [dismissedImportError, setDismissedImportError] = useState<
    string | null
  >(null);
  useEffect(() => {
    setDismissedImportError(null);
  }, [project?.id]);
  const projectIdRef = useRef(project?.id);
  projectIdRef.current = project?.id;
  const searchRef = useRef<HTMLInputElement | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const filmstripRef = useRef<HTMLDivElement | null>(null);
  const photos = useMemo(
    () => (project ? visiblePhotos(project, options) : []),
    [project, options],
  );
  const active = project?.photos.find((photo) => photo.id === activeId);
  const selectedIds = selected.length ? selected : activeId ? [activeId] : [];
  const similarGroups = useMemo(
    () => project?.groups.filter((group) => group.photoIds.length > 1) ?? [],
    [project?.groups],
  );
  const recommended = useMemo(
    () => new Set(similarGroups.flatMap((group) => group.recommendedPhotoIds)),
    [similarGroups],
  );
  const comparePhotos = comparisonPhotos(photos, selected, activeId);
  const scopeId = options.scope.type === "all" ? null : options.scope.id;
  const scopeCollection =
    options.scope.type === "collection"
      ? project?.collections.find((value) => value.id === scopeId)
      : undefined;
  const scopeGroup =
    options.scope.type === "group"
      ? project?.groups.find((value) => value.id === scopeId)
      : undefined;
  const title =
    scopeCollection?.name ??
    scopeGroup?.label ??
    filters.find((filter) => filter.id === options.filter)?.label ??
    "All photographs";
  const favorites =
    project?.photos.filter((photo) => photo.decision === "favorite").length ??
    0;
  const reviewed =
    project?.photos.filter((photo) => photo.reviewed).length ?? 0;
  const processing = project?.importStatus === "running";

  useEffect(() => {
    if (!isAvailable) return;
    api
      .state()
      .then(setState)
      .catch((reason) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, [setError]);
  useEffect(() => {
    if (!photos.length) {
      setActiveId(null);
      setSelected([]);
      return;
    }
    if (!photos.some((photo) => photo.id === activeId))
      setActiveId(photos[0].id);
    const ids = new Set(photos.map((photo) => photo.id));
    setSelected((previous) =>
      previous.every((id) => ids.has(id))
        ? previous
        : previous.filter((id) => ids.has(id)),
    );
  }, [photos, activeId]);
  useEffect(() => {
    setRenderLimit(180);
  }, [options]);
  useEffect(() => {
    const container = view === "grid" ? gridRef.current : filmstripRef.current;
    container
      ?.querySelector(".photo-card.focused")
      ?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [activeId, view, renderLimit]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(timer);
  }, [toast]);
  function report(reason: unknown) {
    setError(String(reason));
  }
  function changeScope(scope: Scope, filter: Filter = "all") {
    setOptions((previous) => ({ ...previous, scope, filter }));
    setSelected([]);
    setActiveId(null);
  }
  function mark(patch: PhotoPatch, label: string) {
    void edit(selectedIds, patch, label);
  }
  function saveTags(tags: string[], add: boolean) {
    return add
      ? addTags(selectedIds, tags)
      : edit(selectedIds, { tags }, "Edit tags");
  }
  function selectViewerPhoto(id: string) {
    setActiveId(id);
    if (!selected.includes(id)) setSelected([id]);
  }
  async function goToProjects() {
    commitDrafts();
    if (!(await waitForSaves())) return;
    load(null);
    void api.state().then(setState).catch(report);
  }
  function navigate(direction: number) {
    const next = nextPhotoId(photos, activeId, direction);
    setActiveId(next);
    setSelected(next ? [next] : []);
    const index = photos.findIndex((photo) => photo.id === next);
    if (index >= renderLimit)
      setRenderLimit(Math.ceil((index + 1) / 180) * 180);
  }
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!project || exportOpen || collectionName !== null || pluginPath)
        return;
      const target = event.target as HTMLElement;
      if (target.matches("input, textarea, select") || target.isContentEditable)
        return;
      const key = event.key.toLowerCase();
      if ((event.ctrlKey || event.metaKey) && key === "z") {
        event.preventDefault();
        void undo();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && key === "a") {
        event.preventDefault();
        setSelected(photos.map((photo) => photo.id));
        return;
      }
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (key === "arrowright" || key === "arrowdown") {
        event.preventDefault();
        navigate(1);
      } else if (key === "arrowleft" || key === "arrowup") {
        event.preventDefault();
        navigate(-1);
      } else if (key === "f")
        mark({ decision: "favorite", reviewed: true }, "Favorite");
      else if (key === "p") mark({ decision: "pass", reviewed: true }, "Pass");
      else if (key === "u")
        mark({ decision: "undecided", reviewed: false }, "Undecided");
      else if (/^[0-5]$/.test(key))
        mark({ rating: Number(key), reviewed: true }, "Rate photographs");
      else if (
        key === "enter" &&
        (!target.closest("button") || target.closest(".photo-card"))
      )
        setView("single");
      else if (key === "escape" || key === "g") setView("grid");
      else if (key === "c" && photos.length >= 2) setView("compare");
      else if (key === "/") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  useEffect(() => {
    if (!exportOpen && collectionName === null && !pluginPath) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const modal = document.querySelector(
      ".modal-backdrop:last-of-type .modal",
    ) as HTMLElement | null;
    const focusable = () =>
      Array.from(
        modal?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input, select, textarea, [tabindex="0"]',
        ) ?? [],
      );
    (
      modal?.querySelector<HTMLElement>("input, select") ?? focusable()[0]
    )?.focus();
    function handleDialogKey(event: KeyboardEvent) {
      if (event.key === "Escape" && !exporting) {
        if (pluginPath) setPluginPath(null);
        else if (exportOpen) setExportOpen(false);
        else setCollectionName(null);
      }
      if (event.key !== "Tab") return;
      const controls = focusable();
      const first = controls[0];
      const last = controls.at(-1);
      if (!modal?.contains(document.activeElement)) {
        event.preventDefault();
        first?.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
    window.addEventListener("keydown", handleDialogKey);
    return () => {
      window.removeEventListener("keydown", handleDialogKey);
      previousFocus?.focus();
    };
  }, [exportOpen, collectionName !== null, pluginPath, exporting]);
  async function openRecent(path: string) {
    setLoading(true);
    try {
      const value = await api.openProject(path);
      load(value);
      setActiveId(value.photos[0]?.id ?? null);
      setSelected([]);
      setView("grid");
      setOptions({
        filter: "all",
        query: "",
        sort: "time",
        scope: { type: "all" },
      });
    } catch (reason) {
      report(reason);
    } finally {
      setLoading(false);
    }
  }
  async function newProject() {
    try {
      const directory = await chooseFolder();
      if (!directory) return;
      setLoading(true);
      const name =
        directory.split(/[\\/]/).filter(Boolean).at(-1) || "My photographs";
      const value = await api.createProject(name, directory);
      load(value);
      setSelected([]);
      setView("grid");
      setOptions({
        filter: "all",
        query: "",
        sort: "time",
        scope: { type: "all" },
      });
      await api.start(value.id);
      await workspace.refresh();
    } catch (reason) {
      report(reason);
    } finally {
      setLoading(false);
    }
  }
  async function rescan() {
    if (!project || importAction) return;
    setImportAction(true);
    try {
      await api.start(project.id);
      await workspace.refresh();
      setToast("Rescan started. Your choices and collections are preserved.");
    } catch (reason) {
      report(reason);
    } finally {
      setImportAction(false);
    }
  }
  async function detail(values: Photo[]) {
    if (!project) return false;
    setDetailLoading(true);
    try {
      for (const photo of values) {
        if (projectIdRef.current !== project.id) return false;
        await api.detail(project.id, photo.id);
      }
      if (projectIdRef.current !== project.id) return false;
      await workspace.refresh();
      setToast(
        "Full-resolution previews ready. Use 100% to check fine detail.",
      );
      return true;
    } catch (reason) {
      if (projectIdRef.current === project.id) report(reason);
      return false;
    } finally {
      setDetailLoading(false);
    }
  }
  async function exportSelection() {
    if (!project) return;
    commitDrafts();
    if (!(await waitForSaves())) return;
    setExporting(true);
    try {
      const destination = await exportPath(project.name);
      if (!destination) return;
      const result = await api.export(
        project.id,
        destination,
        exportTarget === "favorites" ? null : exportTarget,
        exportTarget === "favorites",
      );
      setToast(`Exported ${result.count} photographs to ${result.path}`);
      setExportOpen(false);
    } catch (reason) {
      report(reason);
    } finally {
      setExporting(false);
    }
  }
  async function showPlugin() {
    try {
      setPluginPath(await api.plugin());
    } catch (reason) {
      report(reason);
    }
  }
  function selectPhoto(id: string, event: React.MouseEvent) {
    if (event.shiftKey && activeId) {
      const first = photos.findIndex((photo) => photo.id === activeId);
      const last = photos.findIndex((photo) => photo.id === id);
      setSelected(
        photos
          .slice(Math.min(first, last), Math.max(first, last) + 1)
          .map((photo) => photo.id),
      );
    } else if (event.ctrlKey || event.metaKey) {
      const next = selected.includes(id)
        ? selected.filter((value) => value !== id)
        : [...selected, id];
      setSelected(next);
      setActiveId(next.includes(id) ? id : (next.at(-1) ?? id));
      return;
    } else setSelected([id]);
    setActiveId(id);
  }
  const visibleError =
    error ||
    (project?.importError !== dismissedImportError
      ? project?.importError
      : null);
  const errorBanner = visibleError && (
    <div className="error-banner" role="alert">
      <span>{visibleError}</span>
      <button
        aria-label="Dismiss error"
        onClick={() => {
          setError(null);
          setDismissedImportError(project?.importError ?? null);
        }}
      >
        <X size={16} />
      </button>
    </div>
  );
  return (
    <div className="app">
      {isDemo && (
        <div className="demo-banner">
          DEMO PREVIEW · Sample photographs and choices. Your photo library is
          not connected.
        </div>
      )}
      {!project ? (
        <>
          <header className="landing-header">
            <div className="brand">
              <Aperture size={28} />
              <span>Photo Select</span>
            </div>
            <span className="local-badge">
              <span />
              Made for your local library
            </span>
          </header>
          {errorBanner}
          <main className="landing">
            <div className="landing-intro">
              <div className="eyebrow">LESS SORTING. MORE PHOTOGRAPHY.</div>
              <h1>
                Find the ones
                <br />
                you want to keep.
              </h1>
              <p>
                A thoughtful first pass through your photographs.
                <br />
                Compare moments, choose favorites, and carry your selection into
                Lightroom.
              </p>
              {!isAvailable ? (
                <div className="desktop-required">
                  <strong>Your photographs live on your computer.</strong>
                  <p>
                    Open the Photo Select desktop app to choose a folder, review
                    originals, and save your selection.
                  </p>
                </div>
              ) : (
                <div className="landing-actions">
                  <button
                    className="primary-button"
                    disabled={loading || state?.engineAvailable === false}
                    onClick={() => void newProject()}
                  >
                    {loading ? (
                      <LoaderCircle className="spin" size={18} />
                    ) : (
                      <FolderPlus size={18} />
                    )}
                    Choose a photo folder
                  </button>
                  <button
                    className="secondary-button"
                    disabled={loading}
                    onClick={() => {
                      void chooseProject()
                        .then((path) => {
                          if (path) return openRecent(path);
                        })
                        .catch(report);
                    }}
                  >
                    <FolderOpen size={17} />
                    Open project
                  </button>
                </div>
              )}
              {state?.engineAvailable === false && (
                <p className="inline-error">
                  The photo analysis engine is unavailable. Repair or reinstall
                  Photo Select to analyze a new folder. Existing projects can
                  still be reviewed.
                </p>
              )}
              <div className="landing-assurances">
                <span>
                  <Check size={14} />
                  Originals stay untouched
                </span>
                <span>
                  <Check size={14} />
                  Your decisions, autosaved
                </span>
                <span>
                  <Check size={14} />
                  Works locally
                </span>
              </div>
            </div>
            <div className="landing-art" aria-hidden="true">
              <div className="art-photo art-one">
                <div className="art-mountain" />
              </div>
              <div className="art-photo art-two">
                <div className="art-sun" />
                <div className="art-mountain" />
              </div>
              <div className="art-photo art-three">
                <div className="art-mountain" />
              </div>
              <div className="art-heart">
                <Heart size={19} fill="currentColor" />
              </div>
            </div>
            {state && state.projects.length > 0 && (
              <section className="recent-projects">
                <div className="section-heading">
                  <h2>Pick up where you left off</h2>
                  <span>Recent projects</span>
                </div>
                <div className="recent-grid">
                  {state.projects.map((recent) => (
                    <button
                      key={recent.id}
                      className="recent-card"
                      disabled={loading}
                      onClick={() => void openRecent(recent.projectPath)}
                    >
                      <div className="recent-icon">
                        <FolderOpen size={26} />
                      </div>
                      <div>
                        <strong>{recent.name}</strong>
                        <span>
                          {recent.photoCount.toLocaleString()} photographs ·{" "}
                          {recent.favoriteCount} favorites
                        </span>
                        <small>{recent.sourceDir}</small>
                      </div>
                      <ArrowRight size={18} />
                    </button>
                  ))}
                </div>
              </section>
            )}
          </main>
          <footer className="landing-footer">
            Photo Select <span>For the photographs that matter.</span>
            <span>v{state?.version ?? "0.2.0"}</span>
          </footer>
        </>
      ) : (
        <>
          <aside className="sidebar">
            <div className="brand">
              <Aperture size={26} />
              <span>Photo Select</span>
            </div>
            <button
              className="back-button"
              disabled={saving > 0}
              onClick={() => void goToProjects()}
            >
              <ArrowLeft size={14} />
              Projects
            </button>
            <div className="project-title">
              <h1>{project.name}</h1>
              <span>{project.photos.length.toLocaleString()} photographs</span>
            </div>
            <nav aria-label="Library filters" className="library-nav">
              {filters.map((filter) => {
                const Icon = filter.icon;
                const count =
                  filter.id === "all"
                    ? project.photos.length
                    : filter.id === "favorite"
                      ? favorites
                      : filter.id === "unreviewed"
                        ? project.photos.length - reviewed
                        : project.photos.filter(
                            (photo) => photo.decision === "pass",
                          ).length;
                return (
                  <button
                    className={
                      options.scope.type === "all" &&
                      options.filter === filter.id
                        ? "active"
                        : ""
                    }
                    key={filter.id}
                    onClick={() => changeScope({ type: "all" }, filter.id)}
                  >
                    <Icon size={17} />
                    <span>{filter.label}</span>
                    <span className="nav-count">{count}</span>
                  </button>
                );
              })}
            </nav>
            <div className="sidebar-section">
              <div className="sidebar-section-title">
                <span>SIMILAR MOMENTS</span>
                <span>{similarGroups.length}</span>
              </div>
              <div className="moment-list">
                {similarGroups.map((group, index) => (
                  <button
                    className={
                      options.scope.type === "group" &&
                      options.scope.id === group.id
                        ? "active"
                        : ""
                    }
                    key={group.id}
                    onClick={() => changeScope({ type: "group", id: group.id })}
                  >
                    <span className="moment-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span>{group.label}</span>
                    <span className="nav-count">{group.photoIds.length}</span>
                  </button>
                ))}
                {!similarGroups.length && (
                  <p className="sidebar-empty">
                    {processing
                      ? "Finding similar photographs…"
                      : "No similar moments found. Your photographs are in All photographs."}
                  </p>
                )}
              </div>
            </div>
            <div className="sidebar-section collections-section">
              <div className="sidebar-section-title">
                <span>COLLECTIONS</span>
                <button
                  className="icon-button"
                  aria-label="Create collection"
                  title="Create collection"
                  onClick={() => setCollectionName("")}
                >
                  <Plus size={15} />
                </button>
              </div>
              {project.collections.map((collection) => (
                <button
                  className={`collection-link ${options.scope.type === "collection" && options.scope.id === collection.id ? "active" : ""}`}
                  key={collection.id}
                  onClick={() =>
                    changeScope({ type: "collection", id: collection.id })
                  }
                >
                  <Layers size={15} />
                  <span>{collection.name}</span>
                  <span className="nav-count">
                    {collection.photoIds.length}
                  </span>
                </button>
              ))}
              {project.collections.length === 0 && (
                <p className="sidebar-empty">
                  Gather photographs for a print, album, or story.
                </p>
              )}
            </div>
            <div className="sidebar-bottom">
              <div className="review-summary">
                <span>Review progress</span>
                <strong>
                  {project.photos.length
                    ? Math.round((reviewed / project.photos.length) * 100)
                    : 0}
                  %
                </strong>
                <div className="review-bar">
                  <span
                    style={{
                      width: `${project.photos.length ? (reviewed / project.photos.length) * 100 : 0}%`,
                    }}
                  />
                </div>
                <small>
                  {reviewed.toLocaleString()} of{" "}
                  {project.photos.length.toLocaleString()} reviewed
                </small>
              </div>
              <button className="text-button" onClick={() => void showPlugin()}>
                <Settings2 size={14} />
                Lightroom connection
              </button>
              <button
                className="text-button"
                disabled={
                  processing || importAction || state?.engineAvailable === false
                }
                onClick={() => void rescan()}
              >
                <RefreshCw size={14} />
                Rescan folder
              </button>
            </div>
          </aside>
          <div className="workspace">
            <header className="workspace-header">
              <div className="breadcrumb">
                <span>{project.name}</span>
                <ChevronRight size={13} />
                <strong>{title}</strong>
              </div>
              <div className="workspace-header-actions">
                <span
                  className={`save-status ${saving ? "saving" : saveError ? "save-failed" : ""}`}
                  title={saveError ?? undefined}
                >
                  {saving ? (
                    <LoaderCircle className="spin" size={13} />
                  ) : saveError ? (
                    <AlertCircle size={13} />
                  ) : (
                    <Check size={13} />
                  )}{" "}
                  {saving
                    ? "Saving…"
                    : saveError
                      ? "Last change failed to save"
                      : "All changes saved"}
                </span>
                <button
                  className="secondary-button small-button"
                  title={
                    undoLabel ? `Undo ${undoLabel} (Ctrl+Z)` : "Nothing to undo"
                  }
                  disabled={!undoLabel || saving > 0}
                  onClick={() => void undo()}
                >
                  <Undo2 size={15} />
                  Undo
                </button>
                <button
                  className="primary-button small-button"
                  disabled={saving > 0}
                  onClick={() => {
                    setExportTarget(scopeCollection?.id ?? "favorites");
                    setExportOpen(true);
                  }}
                >
                  <Download size={15} />
                  Export selection
                </button>
              </div>
            </header>
            {errorBanner}
            {processing && (
              <div className="processing-banner">
                <LoaderCircle className="spin" size={15} />
                <div>
                  <strong>
                    {progress
                      ? progressLabel(progress)
                      : "Preparing your photographs"}
                  </strong>
                  <span>
                    {progress?.total
                      ? `${progress.processed.toLocaleString()} / ${progress.total.toLocaleString()}`
                      : "You can review photos as they arrive."}
                    {progress?.failed
                      ? ` · ${progress.failed} could not be analyzed`
                      : ""}
                  </span>
                </div>
                <div className="processing-track">
                  <span
                    style={{
                      width: `${progress?.total ? Math.min(100, (progress.processed / progress.total) * 100) : 4}%`,
                    }}
                  />
                </div>
                <button
                  className="text-button"
                  onClick={() => {
                    void api
                      .cancel(project.id)
                      .then(workspace.refresh)
                      .catch(report);
                  }}
                >
                  Pause
                </button>
              </div>
            )}
            <div className="browse-toolbar">
              <div>
                <h2>{title}</h2>
                <p>
                  {photos.length.toLocaleString()} photographs
                  {selected.length > 1 ? ` · ${selected.length} selected` : ""}
                  {scopeGroup ? " · Similar moments, your choice" : ""}
                </p>
              </div>
              <div className="browse-controls">
                <label className="search-box">
                  <Search size={16} />
                  <input
                    ref={searchRef}
                    value={options.query}
                    onChange={(event) =>
                      setOptions((previous) => ({
                        ...previous,
                        query: event.target.value,
                      }))
                    }
                    placeholder="Search photos, tags, camera"
                    aria-label="Search photos, tags, or camera"
                  />
                  {options.query && (
                    <button
                      aria-label="Clear search"
                      onClick={() =>
                        setOptions((previous) => ({ ...previous, query: "" }))
                      }
                    >
                      <X size={13} />
                    </button>
                  )}
                </label>
                <label className="sort-select">
                  <Clock3 size={15} />
                  <select
                    value={options.sort}
                    aria-label="Sort photographs"
                    onChange={(event) =>
                      setOptions((previous) => ({
                        ...previous,
                        sort: event.target.value as Sort,
                      }))
                    }
                  >
                    <option value="time">Chronological</option>
                    <option value="quality">Technical quality</option>
                    <option value="rating">My rating</option>
                  </select>
                  <ChevronDown size={12} />
                </label>
                <div className="segmented view-toggle">
                  <button
                    aria-label="Grid view"
                    title="Grid (G)"
                    className={view === "grid" ? "active" : ""}
                    onClick={() => setView("grid")}
                  >
                    <Grid2X2 size={17} />
                  </button>
                  <button
                    aria-label="Single photo view"
                    title="Single photo (Enter)"
                    className={view === "single" ? "active" : ""}
                    onClick={() => setView("single")}
                  >
                    <Image size={17} />
                  </button>
                  <button
                    aria-label="Compare selected photos"
                    title="Compare 2–4 photographs (C)"
                    disabled={photos.length < 2}
                    className={view === "compare" ? "active" : ""}
                    onClick={() => setView("compare")}
                  >
                    <Layers size={17} />
                  </button>
                </div>
              </div>
            </div>
            {(scopeGroup || scopeCollection) && (
              <div className="scope-toolbar">
                <span>
                  {scopeGroup ? (
                    <>
                      <Sparkles size={13} />
                      Suggested representatives have a sparkle. Favorites are
                      always your decision.
                    </>
                  ) : (
                    <>
                      Collection · {scopeCollection?.photoIds.length}{" "}
                      photographs
                    </>
                  )}
                </span>
                {scopeCollection && (
                  <button
                    className="text-button"
                    disabled={!selectedIds.length}
                    onClick={() =>
                      void collectionEdit(scopeCollection.id, selectedIds, true)
                    }
                  >
                    Remove selected from collection
                  </button>
                )}
                {scopeGroup && (
                  <>
                    <button
                      className="icon-button"
                      aria-label="Previous moment"
                      disabled={
                        similarGroups.findIndex(
                          (group) => group.id === scopeGroup.id,
                        ) === 0
                      }
                      onClick={() => {
                        const index = similarGroups.findIndex(
                          (group) => group.id === scopeGroup.id,
                        );
                        if (index > 0)
                          changeScope({
                            type: "group",
                            id: similarGroups[index - 1].id,
                          });
                      }}
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <button
                      className="icon-button"
                      aria-label="Next moment"
                      disabled={
                        similarGroups.findIndex(
                          (group) => group.id === scopeGroup.id,
                        ) ===
                        similarGroups.length - 1
                      }
                      onClick={() => {
                        const index = similarGroups.findIndex(
                          (group) => group.id === scopeGroup.id,
                        );
                        if (index < similarGroups.length - 1)
                          changeScope({
                            type: "group",
                            id: similarGroups[index + 1].id,
                          });
                      }}
                    >
                      <ChevronRight size={16} />
                    </button>
                  </>
                )}
              </div>
            )}
            <div className="review-layout">
              <main className="photo-area">
                {!photos.length ? (
                  <div className="empty-state">
                    <Image size={38} />
                    <h3>
                      {processing
                        ? "Your photographs are on their way."
                        : options.query
                          ? "No photographs match your search."
                          : options.filter === "favorite"
                            ? "Your favorites will live here."
                            : options.filter === "unreviewed"
                              ? "Every photograph has had a look."
                              : scopeCollection
                                ? "A collection waiting for your story."
                                : "No photographs to show yet."}
                    </h3>
                    <p>
                      {processing
                        ? "Analysis runs in the background. You can start reviewing as soon as the first previews arrive."
                        : options.query
                          ? "Try a filename, tag, or camera name."
                          : options.filter === "favorite"
                            ? "Press F on any photograph to make it a favorite."
                            : scopeCollection
                              ? "Select photographs in your library and add them from the details panel."
                              : "Choose another filter or rescan your source folder."}
                    </p>
                    {!processing && !options.query && !scopeCollection && (
                      <button
                        className="secondary-button"
                        onClick={() => void rescan()}
                      >
                        <RefreshCw size={15} />
                        Rescan folder
                      </button>
                    )}
                  </div>
                ) : view === "grid" ? (
                  <div className="grid-scroll" ref={gridRef}>
                    <div className="photo-grid">
                      {photos.slice(0, renderLimit).map((photo) => (
                        <PhotoCard
                          key={photo.id}
                          photo={photo}
                          selected={selected.includes(photo.id)}
                          active={activeId === photo.id}
                          recommended={recommended.has(photo.id)}
                          onClick={(event) => selectPhoto(photo.id, event)}
                          onDoubleClick={() => {
                            setActiveId(photo.id);
                            setView("single");
                          }}
                        />
                      ))}
                    </div>
                    {renderLimit < photos.length && (
                      <button
                        className="load-more secondary-button"
                        onClick={() =>
                          setRenderLimit((previous) => previous + 180)
                        }
                      >
                        Show next {Math.min(180, photos.length - renderLimit)}{" "}
                        photographs <ChevronDown size={15} />
                      </button>
                    )}
                  </div>
                ) : (
                  <>
                    {view === "compare" && selected.length > 4 && (
                      <div className="compare-note">
                        Showing the first 4 of {selected.length} selected
                        photographs. Decisions apply to the full selection.
                      </div>
                    )}
                    <Viewer
                      photos={
                        view === "single"
                          ? active
                            ? [active]
                            : []
                          : comparePhotos
                      }
                      recommended={recommended}
                      onDetail={detail}
                      loadingDetail={detailLoading}
                      onActive={selectViewerPhoto}
                    />
                    <div className="filmstrip-row">
                      <button
                        className="icon-button"
                        aria-label="Previous photo"
                        onClick={() => navigate(-1)}
                        disabled={photos[0]?.id === activeId}
                      >
                        <ChevronLeft size={20} />
                      </button>
                      <div className="filmstrip" ref={filmstripRef}>
                        {photos
                          .slice(
                            Math.max(
                              0,
                              photos.findIndex(
                                (photo) => photo.id === activeId,
                              ) - 20,
                            ),
                            photos.findIndex((photo) => photo.id === activeId) +
                              21,
                          )
                          .map((photo) => (
                            <PhotoCard
                              compact
                              key={photo.id}
                              photo={photo}
                              selected={selected.includes(photo.id)}
                              active={activeId === photo.id}
                              recommended={recommended.has(photo.id)}
                              onClick={(event) => selectPhoto(photo.id, event)}
                              onDoubleClick={() => setView("single")}
                            />
                          ))}
                      </div>
                      <button
                        className="icon-button"
                        aria-label="Next photo"
                        onClick={() => navigate(1)}
                        disabled={photos.at(-1)?.id === activeId}
                      >
                        <ChevronRight size={20} />
                      </button>
                    </div>
                  </>
                )}
              </main>
              <Inspector
                photo={active}
                selectedCount={selectedIds.length}
                selectionKey={selectedIds.join("|")}
                onSaveTags={saveTags}
                collections={project.collections}
                onEdit={mark}
                onCollection={(id) => void collectionEdit(id, selectedIds)}
                onReveal={() => {
                  if (active) void revealPath(active.path).catch(report);
                }}
              />
            </div>
            <footer className="workspace-footer">
              <span>
                <kbd>←</kbd>
                <kbd>→</kbd> Navigate <kbd>F</kbd> Favorite <kbd>P</kbd> Pass{" "}
                <kbd>0–5</kbd> Rate
              </span>
              <span>
                {view === "grid"
                  ? "Double-click to inspect · Ctrl / Shift-click to select"
                  : `${Math.max(0, photos.findIndex((photo) => photo.id === activeId) + 1)} of ${photos.length} · G for grid`}
              </span>
              <span>
                <Heart size={12} />
                {favorites} favorites
              </span>
            </footer>
          </div>
        </>
      )}
      {toast && (
        <div className="toast" role="status">
          <Check size={16} />
          {toast}
          <button
            aria-label="Dismiss notification"
            onClick={() => setToast(null)}
          >
            <X size={14} />
          </button>
        </div>
      )}
      {collectionName !== null && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setCollectionName(null);
          }}
        >
          <form
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="collection-title"
            onSubmit={async (event) => {
              event.preventDefault();
              if (
                collectionName.trim() &&
                !saving &&
                (await createCollection(collectionName.trim()))
              )
                setCollectionName(null);
            }}
          >
            <button
              type="button"
              className="modal-close icon-button"
              aria-label="Close"
              onClick={() => setCollectionName(null)}
            >
              <X size={18} />
            </button>
            <div className="modal-symbol">
              <Layers size={24} />
            </div>
            <h2 id="collection-title">A place for a story.</h2>
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <p>
              Make a collection for a photo book, a set of prints, or someone
              you love.
            </p>
            <label>
              Collection name
              <input
                autoFocus
                value={collectionName}
                onChange={(event) => setCollectionName(event.target.value)}
                placeholder="e.g. Summer photo book"
                maxLength={120}
              />
            </label>
            <button
              className="primary-button"
              disabled={!collectionName.trim() || saving > 0}
              type="submit"
            >
              <Plus size={16} />
              Create collection
            </button>
          </form>
        </div>
      )}
      {exportOpen && project && (
        <div className="modal-backdrop">
          <div
            className="modal export-modal"
            aria-hidden={Boolean(pluginPath)}
            role="dialog"
            aria-modal="true"
            aria-labelledby="export-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close export"
              disabled={exporting}
              onClick={() => setExportOpen(false)}
            >
              <X size={18} />
            </button>
            <div className="modal-symbol">
              <Download size={24} />
            </div>
            <h2 id="export-title">Take your selection to Lightroom.</h2>
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <p>
              Export a selection file with your choices, star ratings, and tags.
              Your original files stay in their folders.
            </p>
            <label>
              What to export
              <select
                value={exportTarget}
                onChange={(event) => setExportTarget(event.target.value)}
              >
                <option value="favorites">Favorites ({favorites})</option>
                {project.collections.map((collection) => (
                  <option key={collection.id} value={collection.id}>
                    {collection.name} ({collection.photoIds.length})
                  </option>
                ))}
              </select>
            </label>
            <div className="export-help">
              <strong>In Lightroom Classic</strong>
              <ol>
                <li>Import your original photos into your catalog.</li>
                <li>
                  Add PhotoSelect.lrplugin through File → Plug-in Manager.
                </li>
                <li>
                  Choose Library → Plug-in Extras → Import Photo Select
                  shortlist… (or File → Plug-in Extras), then open the exported
                  JSON file.
                </li>
              </ol>
              <button className="text-button" onClick={() => void showPlugin()}>
                Locate the Lightroom plugin ↗
              </button>
            </div>
            <button
              className="primary-button"
              disabled={
                exporting ||
                saving > 0 ||
                (exportTarget === "favorites"
                  ? favorites === 0
                  : !project.collections.find(
                      (collection) => collection.id === exportTarget,
                    )?.photoIds.length)
              }
              onClick={() => void exportSelection()}
            >
              {exporting ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Download size={16} />
              )}
              Save selection file
            </button>
          </div>
        </div>
      )}
      {pluginPath && (
        <div className="modal-backdrop">
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="plugin-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close plugin information"
              onClick={() => setPluginPath(null)}
            >
              <X size={18} />
            </button>
            <div className="modal-symbol">
              <Settings2 size={24} />
            </div>
            <h2 id="plugin-title">Connect Lightroom Classic.</h2>
            <p>
              In Lightroom, open File → Plug-in Manager → Add, then select the
              PhotoSelect.lrplugin folder shown below.
            </p>
            <code className="plugin-path">{pluginPath}</code>
            <p>
              The plugin matches photographs by their original file paths and
              applies your exported choices to the catalog.
            </p>
            <button
              className="primary-button"
              onClick={() => void revealPath(pluginPath).catch(report)}
            >
              <FolderOpen size={16} />
              Show plugin in folder
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
