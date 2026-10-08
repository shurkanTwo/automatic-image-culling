import { useState } from "react";
import { ImageOff } from "lucide-react";
import { photoSrc } from "../api";
export default function PhotoImage({
  path,
  alt,
  thumbnail = false,
  retryVersion = 0,
  onLoad,
  onError,
}: {
  path: string | null;
  alt: string;
  thumbnail?: boolean;
  retryVersion?: number;
  onLoad?: (image: HTMLImageElement) => void;
  onError?: () => void;
}) {
  const [failedSource, setFailedSource] = useState<string | null>(null);
  const base = path ? photoSrc(path) : "";
  const source =
    retryVersion && !/^(data:|blob:)/.test(base)
      ? `${base}${base.includes("?") ? "&" : "?"}detailRevision=${retryVersion}`
      : base;
  if (!path || source === failedSource)
    return (
      <div className="image-fallback">
        <ImageOff size={24} />
        <span>Preview unavailable</span>
      </div>
    );
  return (
    <img
      key={`${source}:${retryVersion}`}
      src={source}
      alt={alt}
      loading={thumbnail ? "lazy" : "eager"}
      decoding="async"
      draggable={false}
      onLoad={(event) => onLoad?.(event.currentTarget)}
      onError={() => {
        setFailedSource(source);
        onError?.();
      }}
    />
  );
}
