import { useEffect, useRef, useState } from "react";
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
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [wantActualSize, setWantActualSize] = useState(false);
  const [loadedImage, setLoadedImage] = useState<{
    path: string;
    width: number;
    height: number;
  } | null>(null);
  const drag = useRef<{
    x: number;
    y: number;
    panX: number;
    panY: number;
  } | null>(null);
  const pane = useRef<HTMLDivElement | null>(null);
  const identity = photos.map((photo) => photo.id).join(",");
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setWantActualSize(false);
  }, [identity]);
  const firstPath = photos[0]?.detailPath ?? photos[0]?.previewPath;
  useEffect(() => {
    const node = pane.current;
    if (
      !wantActualSize ||
      loadingDetail ||
      !photos.every((photo) => photo.detailPath) ||
      !node ||
      loadedImage?.path !== firstPath
    )
      return;
    if (
      !loadedImage.width ||
      !loadedImage.height ||
      !node.clientWidth ||
      !node.clientHeight
    )
      return;
    const fitRatio = Math.min(
      node.clientWidth / loadedImage.width,
      node.clientHeight / loadedImage.height,
    );
    setZoom(1 / fitRatio);
    setPan({ x: 0, y: 0 });
    setWantActualSize(false);
  }, [wantActualSize, loadingDetail, loadedImage, firstPath, photos]);
  function changeZoom(next: number) {
    setZoom(Math.min(64, Math.max(1, next)));
    if (next <= 1) setPan({ x: 0, y: 0 });
  }
  async function actualSize() {
    if (!photos.length || loadingDetail) return;
    setWantActualSize(true);
    if (photos.every((photo) => photo.detailPath)) return;
    try {
      if (!(await onDetail(photos))) setWantActualSize(false);
    } catch {
      setWantActualSize(false);
    }
  }
  return (
    <div className="viewer-shell">
      <div className="viewer-tools">
        <div className="segmented compact">
          <button
            onClick={() => {
              setWantActualSize(false);
              changeZoom(1);
            }}
            className={zoom === 1 ? "active" : ""}
          >
            <Expand size={15} />
            Fit
          </button>
          <button
            onClick={() => void actualSize()}
            disabled={loadingDetail || wantActualSize}
            title="Show actual image pixels; prepares full-resolution detail when needed"
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
          {zoom === 1 ? "Fit" : `${zoom.toFixed(1)}×`}
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
          disabled={loadingDetail}
          onClick={() => {
            void onDetail(photos);
          }}
        >
          {loadingDetail ? (
            <LoaderCircle className="spin" size={15} />
          ) : (
            <Scan size={15} />
          )}{" "}
          {loadingDetail ? "Preparing detail…" : "Full-resolution detail"}
        </button>
      </div>
      <div className={`viewer-panes count-${photos.length}`}>
        {photos.map((photo, index) => (
          <div className="viewer-pane" key={photo.id}>
            <div
              className={`photo-canvas ${zoom > 1 ? "zoomed" : ""}`}
              ref={index === 0 ? pane : undefined}
              onWheel={(event) => {
                changeZoom(event.deltaY < 0 ? zoom * 1.12 : zoom / 1.12);
              }}
              onPointerDown={(event) => {
                onActive(photo.id);
                if (zoom <= 1) return;
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
                const limit = (zoom - 1) * 50;
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
              onDoubleClick={() => {
                if (zoom > 1) changeZoom(1);
                else void actualSize();
              }}
            >
              <div
                className="transformed-photo"
                style={{
                  transform: `translate(${pan.x}%, ${pan.y}%) scale(${zoom})`,
                }}
              >
                <PhotoImage
                  path={photo.detailPath ?? photo.previewPath}
                  alt={photo.filename}
                  onLoad={
                    index === 0
                      ? (image) =>
                          setLoadedImage({
                            path: photo.detailPath ?? photo.previewPath,
                            width: image.naturalWidth,
                            height: image.naturalHeight,
                          })
                      : undefined
                  }
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
        {zoom > 1
          ? "Drag to inspect · double-click to fit"
          : "Scroll to zoom · double-click for 100%"}
        <span>Originals stay untouched</span>
      </div>
    </div>
  );
}
