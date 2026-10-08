import { Heart, Minus, Star, Sparkles, AlertCircle } from "lucide-react";
import type { MouseEvent } from "react";
import type { Photo } from "../types";
import PhotoImage from "./PhotoImage";
export default function PhotoCard({
  photo,
  selected,
  active,
  recommended,
  onClick,
  onDoubleClick,
  compact = false,
}: {
  photo: Photo;
  selected: boolean;
  active: boolean;
  recommended: boolean;
  onClick: (event: MouseEvent) => void;
  onDoubleClick: () => void;
  compact?: boolean;
}) {
  return (
    <button
      className={`photo-card ${selected ? "selected" : ""} ${active ? "focused" : ""} ${photo.decision === "pass" ? "passed" : ""} ${compact ? "compact-card" : ""}`}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      aria-label={`${photo.filename}, ${photo.decision}${selected ? ", selected" : ""}`}
      aria-pressed={selected}
    >
      <div className="thumbnail">
        <PhotoImage path={photo.thumbnailPath} alt={photo.filename} thumbnail />
        {selected && <span className="selection-dot" />}
        <span className="image-badges">
          {photo.decision === "favorite" && (
            <span className="favorite-badge">
              <Heart size={12} fill="currentColor" />
            </span>
          )}
          {photo.decision === "pass" && (
            <span className="pass-badge">
              <Minus size={12} />
            </span>
          )}
          {recommended && (
            <span className="recommendation-badge" title="Analysis suggestion">
              <Sparkles size={12} />
            </span>
          )}
          {photo.analysisError && (
            <span className="error-badge" title={photo.analysisError}>
              <AlertCircle size={12} />
            </span>
          )}
        </span>
      </div>
      <div className="card-meta">
        <span>{photo.filename}</span>
        <span className="card-rating">
          {photo.rating > 0 && (
            <>
              <Star size={10} fill="currentColor" />
              {photo.rating}
            </>
          )}
        </span>
      </div>
      {!compact && (
        <div className="card-status">
          <span>
            {photo.reviewed
              ? photo.decision === "undecided"
                ? "Reviewed"
                : photo.decision === "favorite"
                  ? "Favorite"
                  : "Passed"
              : "Unreviewed"}
          </span>
          <span>
            {new Date(photo.captureTime).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
      )}
    </button>
  );
}
