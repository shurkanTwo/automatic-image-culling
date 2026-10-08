import { useCallback, useEffect, useRef, useState } from "react";
import {
  Expand,
  Scan,
  ZoomIn,
  ZoomOut,
  Heart,
  Check,
  LoaderCircle,
} from "lucide-react";
import type { Photo } from "../types";
import PhotoImage from "./PhotoImage";

export default function Viewer({
  photos,
  recommended,
  onDetail,
  loadingDetail,
  onActive,
}: {
  photos: Photo[];
  recommended: Set<string>;
  onDetail: (photos: Photo[]) => Promise<boolean>;
  loadingDetail: boolean;
  onActive: (id: string) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [pixelMode, setPixelMode] = useState(false);
  const [pixelRatio, setPixelRatio] = useState(window.devicePixelRatio || 1);
  const [paneSizes, setPaneSizes] = useState<
    Record<string, { width: number; height: number }>
  >({});
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [wantActualSize, setWantActualSize] = useState(false);
  const [detailRevision, setDetailRevision] = useState(0);
  const [preparingDetail, setPreparingDetail] = useState(false);
  const [loadedImages, setLoadedImages] = useState<
    Record<string, { path: string; width: number; height: number }>
  >({});
  const panes = useRef<Record<string, HTMLDivElement>>({});
  const measurePanes = useCallback(() => {
    const measured = Object.fromEntries(
      Object.entries(panes.current).map(([id, node]) => [
        id,
        { width: node.clientWidth, height: node.clientHeight },
      ]),
    );
    setPaneSizes((current) =>
      JSON.stringify(current) === JSON.stringify(measured) ? current : measured,
    );
    setPixelRatio(window.devicePixelRatio || 1);
  }, []);
  const drag = useRef<{
    x: number;
    y: number;
    panX: number;
    panY: number;
  } | null>(null);
  const identity = photos.map((photo) => photo.id).join(",");
  useEffect(() => {
    setZoom(1);
    setPixelMode(false);
    setPan({ x: 0, y: 0 });
    setWantActualSize(false);
    drag.current = null;
  }, [identity]);
  useEffect(() => {
    measurePanes();
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measurePanes);
    Object.values(panes.current).forEach((node) => observer?.observe(node));
    const display =
      typeof window.matchMedia === "function"
        ? window.matchMedia(`(resolution: ${pixelRatio}dppx)`)
        : null;
    display?.addEventListener("change", measurePanes);
    window.addEventListener("resize", measurePanes);
    return () => {
      observer?.disconnect();
      display?.removeEventListener("change", measurePanes);
      window.removeEventListener("resize", measurePanes);
    };
  }, [identity, measurePanes, pixelRatio]);
  const allDetails = photos.every((photo) => photo.detailPath);
  useEffect(() => {
    if (allDetails) return;
    setPixelMode(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [allDetails]);
  useEffect(() => {
    if (
      !wantActualSize ||
      loadingDetail ||
      preparingDetail ||
      !allDetails ||
      !photos.length
    )
      return;
    const ready = photos.every((photo) => {
      const image = loadedImages[photo.id];
      const size = paneSizes[photo.id];
      return (
        image?.path === photo.detailPath &&
        image.width > 0 &&
        image.height > 0 &&
        size?.width > 0 &&
        size.height > 0
      );
    });
    if (!ready) return;
    setPixelMode(true);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setWantActualSize(false);
  }, [
    wantActualSize,
    loadingDetail,
    preparingDetail,
    loadedImages,
    paneSizes,
    allDetails,
    photos,
  ]);
  function paneScale(id: string) {
    if (!pixelMode) return zoom;
    const image = loadedImages[id];
    const size = paneSizes[id];
    if (!image || !size?.width || !size.height) return zoom;
    const fitRatio = Math.min(
      size.width / image.width,
      size.height / image.height,
    );
    // Each pane has its own fit ratio; 100% maps one source pixel to one physical display pixel.
    return zoom / (fitRatio * pixelRatio);
  }
  function fit() {
    setWantActualSize(false);
    setPixelMode(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }
  function changeZoom(next: number) {
    setWantActualSize(false);
    setZoom(Math.min(64, Math.max(pixelMode ? 0.05 : 1, next)));
    if (next <= 1) setPan({ x: 0, y: 0 });
  }
  async function actualSize() {
    if (!photos.length || loadingDetail || preparingDetail) return;
    measurePanes();
    setWantActualSize(true);
    if (photos.every((photo) => photo.detailPath)) return;
    try {
      if (!(await requestDetail())) setWantActualSize(false);
    } catch {
      setWantActualSize(false);
    }
  }
  async function requestDetail() {
    setPreparingDetail(true);
    try {
      const ready = await onDetail(photos);
      if (ready) {
        setLoadedImages({});
        setDetailRevision((value) => value + 1);
      }
      return ready;
    } catch {
      return false;
    } finally {
      setPreparingDetail(false);
    }
  }
  return (
    <div className="viewer-shell">
      <div className="viewer-tools">
        <div className="segmented compact">
          <button
            onClick={fit}
            className={!pixelMode && zoom === 1 ? "active" : ""}
          >
            <Expand size={15} />
            Fit
          </button>
          <button
            onClick={() => void actualSize()}
            className={pixelMode && zoom === 1 ? "active" : ""}
            disabled={loadingDetail || preparingDetail || wantActualSize}
            title="One image pixel per physical display pixel; prepares full-resolution detail when needed"
          >
            {wantActualSize ? (
              <>
                <LoaderCircle className="spin" size={12} />
                Preparing 100%…
              </>
            ) : (
              "100%"
            )}
          </button>
        </div>
        <button
          className="icon-button"
          title="Zoom out"
          aria-label="Zoom out"
          onClick={() => changeZoom(zoom / 1.5)}
        >
          <ZoomOut size={17} />
        </button>
        <span className="zoom-label">
          {pixelMode
            ? `${Math.round(zoom * 100)}%`
            : zoom === 1
              ? "Fit"
              : `${zoom.toFixed(1)}×`}
        </span>
        <button
          className="icon-button"
          title="Zoom in"
          aria-label="Zoom in"
          onClick={() => changeZoom(zoom * 1.5)}
        >
          <ZoomIn size={17} />
        </button>
        {photos.length > 1 && (
          <span className="sync-note">Pan & zoom linked</span>
        )}
        <button
          className="detail-button"
          disabled={loadingDetail || preparingDetail}
          onClick={() => {
            void requestDetail();
          }}
        >
          {loadingDetail || preparingDetail ? (
            <LoaderCircle className="spin" size={15} />
          ) : (
            <Scan size={15} />
          )}{" "}
          {loadingDetail || preparingDetail
            ? "Preparing detail…"
            : "Full-resolution detail"}
        </button>
      </div>
      <div className={`viewer-panes count-${photos.length}`}>
        {photos.map((photo) => (
          <div className="viewer-pane" key={photo.id}>
            <div
              className={`photo-canvas ${paneScale(photo.id) > 1 ? "zoomed" : ""}`}
              ref={(node) => {
                if (node) panes.current[photo.id] = node;
                else delete panes.current[photo.id];
              }}
              onWheel={(event) => {
                changeZoom(event.deltaY < 0 ? zoom * 1.12 : zoom / 1.12);
              }}
              onPointerDown={(event) => {
                onActive(photo.id);
                if (paneScale(photo.id) <= 1) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                drag.current = {
                  x: event.clientX,
                  y: event.clientY,
                  panX: pan.x,
                  panY: pan.y,
                };
              }}
              onPointerMove={(event) => {
                if (!drag.current) return;
                const bounds = event.currentTarget.getBoundingClientRect();
                const limit = Math.max(0, paneScale(photo.id) - 1) * 50;
                const x =
                  drag.current.panX +
                  ((event.clientX - drag.current.x) / bounds.width) * 100;
                const y =
                  drag.current.panY +
                  ((event.clientY - drag.current.y) / bounds.height) * 100;
                setPan({
                  x: Math.max(-limit, Math.min(limit, x)),
                  y: Math.max(-limit, Math.min(limit, y)),
                });
              }}
              onPointerUp={() => {
                drag.current = null;
              }}
              onPointerCancel={() => {
                drag.current = null;
              }}
              onLostPointerCapture={() => {
                drag.current = null;
              }}
              onDoubleClick={() => {
                if (pixelMode || zoom !== 1) fit();
                else void actualSize();
              }}
            >
              <div
                className="transformed-photo"
                style={{
                  transform: `translate(${pan.x}%, ${pan.y}%) scale(${paneScale(photo.id)})`,
                }}
              >
                <PhotoImage
                  path={photo.detailPath ?? photo.previewPath}
                  retryVersion={photo.detailPath ? detailRevision : 0}
                  alt={photo.filename}
                  onLoad={(image) => {
                    setLoadedImages((current) => ({
                      ...current,
                      [photo.id]: {
                        path: photo.detailPath ?? photo.previewPath,
                        width: image.naturalWidth,
                        height: image.naturalHeight,
                      },
                    }));
                    measurePanes();
                  }}
                  onError={() => setWantActualSize(false)}
                />
              </div>
            </div>
            <div className="pane-caption">
              <span>{photo.filename}</span>
              <span className="pane-badges">
                <span className="resolution-status">
                  {photo.detailPath ? "Full-resolution" : "Preview"}
                </span>
                {recommended.has(photo.id) && (
                  <span
                    className="hint-dot"
                    title="Suggested representative of this moment"
                  >
                    Suggested
                  </span>
                )}
                {photo.decision === "favorite" && (
                  <Heart
                    className="favorite-color"
                    size={15}
                    fill="currentColor"
                  />
                )}
                {photo.reviewed && photo.decision !== "favorite" && (
                  <Check size={15} />
                )}
              </span>
            </div>
          </div>
        ))}
      </div>
      <div className="viewer-footnote">
        {pixelMode || zoom !== 1
          ? "Drag to inspect · double-click to fit"
          : "Scroll to zoom · double-click for 100%"}
        <span>Originals stay untouched</span>
      </div>
    </div>
  );
}
