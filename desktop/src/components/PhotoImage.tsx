import { useState } from "react";
import { ImageOff } from "lucide-react";
import { photoSrc } from "../api";
export default function PhotoImage({
  path,
  alt,
  thumbnail = false,
}: {
  path: string | null;
  alt: string;
  thumbnail?: boolean;
}) {
  const [failedPath, setFailedPath] = useState<string | null>(null);
  if (!path || path === failedPath)
    return (
      <div className="image-fallback">
        <ImageOff size={24} />
        <span>Preview unavailable</span>
      </div>
    );
  return (
    <img
      src={photoSrc(path)}
      alt={alt}
      loading={thumbnail ? "lazy" : "eager"}
      decoding="async"
      draggable={false}
      onError={() => setFailedPath(path)}
    />
  );
}
