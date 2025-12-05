"""Discovery helpers for locating files and reading EXIF metadata."""

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import exifread
except ImportError:  # pragma: no cover
    exifread = None


ExifData = Dict[str, Any]


def _is_under(child: Path, parent: Path) -> bool:
    """Return True when child is located within parent."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _normalize_exclude_dirs(
    directory: Path, exclude_dirs: Optional[List[str]]
) -> List[Path]:
    """Convert optional exclude directories into absolute resolved Paths."""
    exclude_paths: List[Path] = []
    for candidate in exclude_dirs or []:
        path_candidate = Path(candidate)
        if not path_candidate.is_absolute():
            path_candidate = directory / path_candidate
        exclude_paths.append(path_candidate.resolve())
    return exclude_paths


def find_arw_files(
    directory: str, exclude_dirs: Optional[List[str]] = None
) -> List[Path]:
    """Recursively find .ARW files under directory while respecting optional exclusions."""
    root = Path(directory)
    if not root.exists():
        return []
    exclude_paths = _normalize_exclude_dirs(root, exclude_dirs)

    results: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not any(
                _is_under(current / dirname, excluded) for excluded in exclude_paths
            )
        ]
        for filename in filenames:
            if filename.lower().endswith(".arw"):
                results.append(current / filename)
    return sorted(results)


def read_exif(path: Path) -> ExifData:
    """Read EXIF data from the given image path, returning a plain dictionary."""
    if exifread is None:
        return {}
    with path.open("rb") as file_handle:
        tags = exifread.process_file(file_handle, details=False)
    return {str(key): str(value) for key, value in tags.items()}


def capture_date(
    exif: ExifData, fallback: Optional[_dt.datetime] = None
) -> _dt.datetime:
    """
    Parse the capture datetime from EXIF or fall back to the provided timestamp.
    """
    dt_raw = exif.get("EXIF DateTimeOriginal") or exif.get("Image DateTime")
    if dt_raw:
        try:
            return _dt.datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return fallback or _dt.datetime.fromtimestamp(_safe_mtime(exif))


def _safe_mtime(exif: ExifData) -> float:
    """Return a stored mtime placeholder when EXIF is missing or incomplete."""
    return float(exif.get("_stat_mtime", _dt.datetime.now().timestamp()))


def plan_destination(
    path: Path, exif: ExifData, cfg: Dict[str, Any], output_dir: Path
) -> Path:
    """Return the destination path for a file given EXIF metadata and sort config."""
    mtime = path.stat().st_mtime
    exif["_stat_mtime"] = mtime
    dt = capture_date(exif, fallback=_dt.datetime.fromtimestamp(mtime))
    date_str = dt.strftime("%Y-%m-%d")
    pattern = cfg.get("pattern", "{capture_date}/{basename}")
    subpath = pattern.format(capture_date=date_str, basename=path.name, stem=path.stem)
    return output_dir / subpath


def exif_orientation(exif: ExifData) -> int:
    """Return EXIF orientation (1 normal, 3 flip, 6 rotate 90 CW, 8 rotate 270)."""
    val = exif.get("Image Orientation") or exif.get("EXIF Orientation")
    if not val:
        return 1
    value_str = str(val)
    try:
        return int(value_str.split()[0])
    except Exception:
        pass
    normalized = value_str.lower()
    if "90" in normalized and ("cw" in normalized or "clockwise" in normalized):
        return 6
    if "90" in normalized or "270" in normalized or "ccw" in normalized:
        return 8
    if "180" in normalized:
        return 3
    return 1
