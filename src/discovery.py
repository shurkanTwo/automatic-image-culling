import os
import pathlib
import datetime as _dt
from typing import Dict, List, Optional

try:
    import exifread
except ImportError:  # pragma: no cover
    exifread = None


def _is_under(child: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_arw_files(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[pathlib.Path]:
    root = pathlib.Path(directory)
    if not root.exists():
        return []
    exclude_paths: List[pathlib.Path] = []
    for ex in exclude_dirs or []:
        ex_path = pathlib.Path(ex)
        if not ex_path.is_absolute():
            ex_path = root / ex_path
        exclude_paths.append(ex_path.resolve())

    results: List[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = pathlib.Path(dirpath)
        dirnames[:] = [d for d in dirnames if not any(_is_under(current / d, ex) for ex in exclude_paths)]
        for fname in filenames:
            if fname.lower().endswith(".arw"):
                results.append(current / fname)
    return sorted(results)


def read_exif(path: pathlib.Path) -> Dict:
    if not exifread:
        return {}
    with path.open("rb") as fh:
        tags = exifread.process_file(fh, details=False, stop_tag="DateTimeOriginal")
    return {str(k): str(v) for k, v in tags.items()}


def capture_date(exif: Dict, fallback: Optional[_dt.datetime] = None) -> _dt.datetime:
    dt_raw = exif.get("EXIF DateTimeOriginal") or exif.get("Image DateTime")
    if dt_raw:
        try:
            return _dt.datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return fallback or _dt.datetime.fromtimestamp(_safe_mtime(exif))


def _safe_mtime(exif: Dict) -> float:
    return exif.get("_stat_mtime", _dt.datetime.now().timestamp())


def plan_destination(path: pathlib.Path, exif: Dict, cfg: Dict, output_dir: pathlib.Path) -> pathlib.Path:
    mtime = path.stat().st_mtime
    exif["_stat_mtime"] = mtime
    dt = capture_date(exif, fallback=_dt.datetime.fromtimestamp(mtime))
    date_str = dt.strftime("%Y-%m-%d")
    pattern = cfg.get("pattern", "{capture_date}/{basename}")
    subpath = pattern.format(capture_date=date_str, basename=path.name, stem=path.stem)
    return output_dir / subpath


def exif_orientation(exif: Dict) -> int:
    """
    Return EXIF orientation (1 is normal, 3 upside-down, 6 rotate 90 CW, 8 rotate 270).
    """
    val = exif.get("Image Orientation") or exif.get("EXIF Orientation")
    if not val:
        return 1
    try:
        return int(str(val).split()[0])
    except Exception:
        return 1
