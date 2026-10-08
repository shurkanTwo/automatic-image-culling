import { useEffect, useRef, useState } from "react";
import {
  Heart,
  Minus,
  Circle,
  Star,
  Sparkles,
  Tag,
  FolderPlus,
  Check,
} from "lucide-react";
import { COMMIT_DRAFTS_EVENT } from "../drafts";
import { parseTags } from "../domain";
import type { Collection, Photo, PhotoPatch } from "../types";
export default function Inspector({
  photo,
  selectedCount,
  selectionKey,
  onSaveTags,
  collections,
  onEdit,
  onCollection,
  onReveal,
}: {
  photo: Photo | undefined;
  selectedCount: number;
  selectionKey: string;
  onSaveTags: (tags: string[], add: boolean) => Promise<boolean>;
  collections: Collection[];
  onEdit: (patch: PhotoPatch, label: string) => void;
  onCollection: (id: string) => void;
  onReveal: () => void;
}) {
  const [tags, setTags] = useState("");
  const dirty = useRef(false);
  const draftIdentity = useRef("");
  const revision = useRef(0);
  const committing = useRef<number | null>(null);
  const bulk = selectedCount > 1;
  useEffect(() => {
    const identity = `${photo?.id ?? ""}:${selectionKey}`;
    if (draftIdentity.current !== identity) {
      draftIdentity.current = identity;
      revision.current += 1;
      dirty.current = false;
      setTags(bulk ? "" : (photo?.tags.join(", ") ?? ""));
    } else if (!dirty.current && !bulk) {
      setTags(photo?.tags.join(", ") ?? "");
    }
  }, [photo?.id, photo?.tags.join(", "), selectionKey, bulk]);
  async function saveTags() {
    if (!photo || !dirty.current || committing.current === revision.current)
      return;
    const parsed = parseTags(tags);
    if (
      (bulk && !parsed.length) ||
      (!bulk && parsed.join(",") === photo.tags.join(","))
    ) {
      dirty.current = false;
      return;
    }
    const submittedRevision = revision.current;
    committing.current = submittedRevision;
    try {
      if (
        (await onSaveTags(parsed, bulk)) &&
        revision.current === submittedRevision
      ) {
        dirty.current = false;
        setTags(bulk ? "" : parsed.join(", "));
      }
    } finally {
      if (committing.current === submittedRevision) committing.current = null;
    }
  }
  useEffect(() => {
    const commit = () => {
      void saveTags();
    };
    document.addEventListener(COMMIT_DRAFTS_EVENT, commit);
    return () => document.removeEventListener(COMMIT_DRAFTS_EVENT, commit);
  });
  if (!photo)
    return (
      <aside className="inspector">
        <div className="inspector-empty">
          Select a photograph to see its details.
        </div>
      </aside>
    );
  const decision = photo.decision;
  return (
    <aside className="inspector">
      <div className="inspector-title">
        <span>YOUR SELECTION</span>
        {selectedCount > 1 && (
          <span className="small-pill">{selectedCount} photos</span>
        )}
      </div>
      <div className="decision-buttons">
        <button
          className={decision === "favorite" ? "favorite chosen" : "favorite"}
          title="Favorite (F)"
          onClick={() =>
            onEdit({ decision: "favorite", reviewed: true }, "Favorite")
          }
        >
          <Heart
            size={18}
            fill={decision === "favorite" ? "currentColor" : "none"}
          />
          <span>Favorite</span>
          <kbd>F</kbd>
        </button>
        <button
          className={decision === "pass" ? "chosen" : ""}
          title="Pass (P) — keeps the original"
          onClick={() => onEdit({ decision: "pass", reviewed: true }, "Pass")}
        >
          <Minus size={18} />
          <span>Pass</span>
          <kbd>P</kbd>
        </button>
        <button
          className={decision === "undecided" ? "chosen" : ""}
          title="Undecided (U)"
          onClick={() =>
            onEdit({ decision: "undecided", reviewed: false }, "Undecided")
          }
        >
          <Circle size={17} />
          <span>Undecided</span>
          <kbd>U</kbd>
        </button>
      </div>
      <p className="quiet-note">Passing keeps the photo in your archive.</p>
      <div className="inspector-section">
        <div className="section-label">
          Star rating <span>0–5</span>
        </div>
        <div className="rating-buttons">
          {[1, 2, 3, 4, 5].map((rating) => (
            <button
              key={rating}
              aria-label={`${rating} star${rating > 1 ? "s" : ""}`}
              onClick={() =>
                onEdit({ rating, reviewed: true }, "Rate photographs")
              }
            >
              <Star
                size={22}
                className={photo.rating >= rating ? "filled-star" : ""}
                fill={photo.rating >= rating ? "currentColor" : "none"}
              />
            </button>
          ))}
          <button
            className="clear-rating"
            title="Clear rating (0)"
            onClick={() =>
              onEdit({ rating: 0, reviewed: true }, "Clear rating")
            }
          >
            Clear
          </button>
        </div>
      </div>
      <div className="inspector-section">
        <label className="section-label" htmlFor="photo-tags">
          <span>
            <Tag size={13} />
            Tags
          </span>
        </label>
        <textarea
          id="photo-tags"
          value={tags}
          placeholder={
            bulk ? "Add tags to selected photos…" : "travel, family, print…"
          }
          rows={2}
          onChange={(event) => {
            dirty.current = true;
            revision.current += 1;
            setTags(event.target.value);
          }}
          onBlur={() => void saveTags()}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.blur();
            }
          }}
        />
        <span className="quiet-note">
          {bulk
            ? "Adds tags to selected photos; existing tags are kept. "
            : "Separate tags with commas. "}
          Saves on leaving this field.
        </span>
      </div>
      <div className="inspector-section">
        <div className="section-label">
          <span>
            <FolderPlus size={13} />
            Add to collection
          </span>
        </div>
        <select
          value=""
          aria-label="Add selected photographs to collection"
          onChange={(event) => {
            if (event.target.value) onCollection(event.target.value);
          }}
        >
          <option value="">Choose a collection…</option>
          {collections.map((collection) => (
            <option value={collection.id} key={collection.id}>
              {collection.name}
            </option>
          ))}
        </select>
      </div>
      <div className="inspector-section analysis-section">
        <div className="section-label">
          <span>
            <Sparkles size={13} />
            Analysis notes
          </span>
          <span className="hint-label">SUGGESTIONS</span>
        </div>
        <p className="analysis-intro">
          A second look at technical quality. Your choices always decide what
          stays.
        </p>
        <div className="quality-bar">
          <span style={{ width: `${photo.qualityScore}%` }} />
        </div>
        <div className="quality-caption">
          <span>Technical estimate</span>
          <strong>{Math.round(photo.qualityScore)} / 100</strong>
        </div>
        <ul className="hints">
          {photo.hints.map((hint, index) => (
            <li key={`${hint}-${index}`}>
              <Check size={12} />
              {hint}
            </li>
          ))}
        </ul>
        {photo.analysisError && (
          <p className="inline-error">{photo.analysisError}</p>
        )}
      </div>
      <div className="inspector-section photo-information">
        <div className="section-label">Photo details</div>
        <strong>{photo.filename}</strong>
        <dl>
          <dt>Captured</dt>
          <dd>
            {new Date(photo.captureTime).toLocaleString([], {
              dateStyle: "medium",
              timeStyle: "short",
            })}
          </dd>
          <dt>Camera</dt>
          <dd>{photo.camera || "Not recorded"}</dd>
          <dt>Dimensions</dt>
          <dd>
            {photo.width.toLocaleString()} × {photo.height.toLocaleString()}
          </dd>
        </dl>
        <button className="text-button" onClick={onReveal}>
          Show original in folder ↗
        </button>
      </div>
    </aside>
  );
}
