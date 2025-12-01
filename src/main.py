import argparse
import concurrent.futures
import datetime as _dt
import io
import json
import os
import pathlib
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dep
    yaml = None

try:
    import rawpy
except ImportError:  # pragma: no cover - optional dep
    rawpy = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dep
    Image = None

try:
    import exifread
except ImportError:  # pragma: no cover - optional dep
    exifread = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dep
    np = None

try:
    import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dep
    tqdm = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
except Exception:  # pragma: no cover - optional dep
    skimage_ssim = None
    skimage_psnr = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional dep
    FaceAnalysis = None

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dep
    mp = None

DEFAULT_CONFIG = {
    "input_dir": "./input",
    "output_dir": "./output",
    "preview_dir": "./previews",
    "preview": {"long_edge": 2048, "format": "webp", "quality": 85},
    "sort": {"strategy": "flat", "copy": True, "pattern": "{basename}"},
    "analysis": {
        "sharpness_min": 8.0,
        "brightness_min": 0.08,
        "brightness_max": 0.92,
        "duplicate_hamming": 6,
        "duplicate_window_seconds": 8,
        "tenengrad_min": 200.0,
        "motion_ratio_min": 0.02,
        "noise_std_max": 25.0,
        "face": {
            "enabled": True,
            "backend": "mediapipe",  # mediapipe or insightface
            "det_size": 640,
            "ctx_id": 0,
        },
        "report_path": "./report.html",
        "results_path": "./analysis.json",
    },
    "concurrency": 4,
}


def load_config(path: Optional[str]) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if path:
        cfg_path = pathlib.Path(path)
    else:
        cfg_path = pathlib.Path("config.yaml")
    if cfg_path.exists() and yaml:
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            cfg = _deep_update(cfg, loaded)
    return cfg


def _deep_update(base: Dict, override: Dict) -> Dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _is_under(child: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_arw_files(
    directory: str, exclude_dirs: Optional[List[str]] = None
) -> List[pathlib.Path]:
    root = pathlib.Path(directory)
    if not root.exists():
        return []
    exclude_paths = []
    for ex in exclude_dirs or []:
        ex_path = pathlib.Path(ex)
        if not ex_path.is_absolute():
            ex_path = root / ex_path
        exclude_paths.append(ex_path.resolve())
    results: List[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = pathlib.Path(dirpath)
        # prune excluded directories
        dirnames[:] = [
            d
            for d in dirnames
            if not any(_is_under(current / d, ex) for ex in exclude_paths)
        ]
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


def resize_image(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= long_edge:
        return img
    if w >= h:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    else:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def generate_preview(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict
) -> Optional[pathlib.Path]:
    if not rawpy or not Image:
        print("rawpy and Pillow are required for preview generation", file=sys.stderr)
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    target = preview_dir / f"{path.stem}.{cfg['format']}"
    if target.exists():
        return target
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
            else:
                raise RuntimeError("Non-JPEG thumbnail; falling back to demosaic")
        except Exception:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            img = Image.fromarray(rgb)
    img = resize_image(img, cfg["long_edge"])
    img.save(target, cfg["format"].upper(), quality=cfg["quality"])
    return target


def ensure_preview(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict
) -> Optional[pathlib.Path]:
    preview = preview_dir / f"{path.stem}.{cfg['format']}"
    if preview.exists():
        return preview
    return generate_preview(path, preview_dir, cfg)


def _progress_bar(total: int, desc: str):
    if tqdm:
        return tqdm.tqdm(total=total, desc=desc, leave=False)

    class _Dummy:
        def update(self, _n: int = 1) -> None:
            return

        def close(self) -> None:
            return

    return _Dummy()


def plan_destination(
    path: pathlib.Path, exif: Dict, cfg: Dict, output_dir: pathlib.Path
) -> pathlib.Path:
    mtime = path.stat().st_mtime
    exif["_stat_mtime"] = mtime
    dt = capture_date(exif, fallback=_dt.datetime.fromtimestamp(mtime))
    date_str = dt.strftime("%Y-%m-%d")
    pattern = cfg.get("pattern", "{capture_date}/{basename}")
    subpath = pattern.format(capture_date=date_str, basename=path.name, stem=path.stem)
    return output_dir / subpath


def _preview_path_for(
    path: pathlib.Path, preview_dir: pathlib.Path, cfg: Dict
) -> pathlib.Path:
    return preview_dir / f"{path.stem}.{cfg['format']}"


def scan_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    exclude = cfg.get("exclude_dirs", [])
    exclude = list(
        set(
            exclude
            + [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]
        )
    )
    files = find_arw_files(cfg["input_dir"], exclude_dirs=exclude)
    print(f"Found {len(files)} .ARW files in {cfg['input_dir']}")
    bar = _progress_bar(len(files), "Scan")
    if args.json:
        data = []
        for p in files:
            exif = read_exif(p)
            data.append({"path": str(p), "exif": exif})
            bar.update(1)
        print(json.dumps(data, indent=2))
    else:
        for p in files:
            exif = read_exif(p)
            dt = capture_date(
                exif, fallback=_dt.datetime.fromtimestamp(p.stat().st_mtime)
            )
            print(f"{p} | {dt}")
            bar.update(1)
    bar.close()


def previews_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    preview_cfg = cfg.get("preview", {})
    preview_cfg.setdefault("long_edge", 2048)
    preview_cfg.setdefault("format", "webp")
    preview_cfg.setdefault("quality", 85)
    exclude = cfg.get("exclude_dirs", [])
    exclude = list(
        set(
            exclude
            + [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]
        )
    )
    files = find_arw_files(cfg["input_dir"], exclude_dirs=exclude)
    if not files:
        print("No .ARW files found")
        return
    preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
    jobs = []
    bar = _progress_bar(len(files), "Previews")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg.get("concurrency", 4)
    ) as pool:
        for p in files:
            jobs.append(pool.submit(generate_preview, p, preview_dir, preview_cfg))
        for job in concurrent.futures.as_completed(jobs):
            result = job.result()
            if result:
                print(f"Preview written: {result}")
            bar.update(1)
    bar.close()


def sort_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sort_cfg = cfg.get("sort", {})
    exclude = cfg.get("exclude_dirs", [])
    exclude = list(
        set(
            exclude
            + [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]
        )
    )
    files = find_arw_files(cfg["input_dir"], exclude_dirs=exclude)
    output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
    actions: List[str] = []
    bar = _progress_bar(len(files), "Sort")
    for p in files:
        exif = read_exif(p)
        dest = plan_destination(p, exif, sort_cfg, output_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            actions.append(
                f"PLAN {'COPY' if sort_cfg.get('copy', True) else 'MOVE'} {p} -> {dest}"
            )
            bar.update(1)
            continue
        if sort_cfg.get("copy", True):
            shutil.copy2(p, dest)
            actions.append(f"COPIED {p} -> {dest}")
        else:
            shutil.move(p, dest)
            actions.append(f"MOVED {p} -> {dest}")
        bar.update(1)
    for line in actions:
        print(line)
    if args.dry_run:
        print("Dry run complete; use --apply to perform moves/copies.")
    bar.close()


def _open_preview_array(preview_path: pathlib.Path) -> Optional[np.ndarray]:
    if not Image or np is None:
        return None
    img = Image.open(preview_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return arr


def _open_preview_rgb(
    preview_path: pathlib.Path, size: Optional[int] = 256
) -> Optional[np.ndarray]:
    if not Image or np is None:
        return None
    img = Image.open(preview_path).convert("RGB")
    if size:
        w, h = img.size
        scale = size / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def _variance_of_laplacian(arr: np.ndarray) -> float:
    # Simple Laplacian without external deps; measures blur/sharpness.
    core = -4 * arr[1:-1, 1:-1]
    core += arr[:-2, 1:-1]
    core += arr[2:, 1:-1]
    core += arr[1:-1, :-2]
    core += arr[1:-1, 2:]
    return float(core.var())


def _tenengrad(arr: np.ndarray) -> float:
    # Sobel-based Tenengrad (focus/contrast) for more stable magnitudes.
    padded = np.pad(arr, 1, mode="reflect")
    gx = (
        padded[0:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - (padded[0:-2, 0:-2] + 2 * padded[1:-1, 0:-2] + padded[2:, 0:-2])
    )
    gy = (
        padded[2:, 0:-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    )
    return float((gx * gx + gy * gy).mean())


def _structure_tensor_ratio(arr: np.ndarray) -> Dict[str, float]:
    gx, gy = np.gradient(arr)
    gxx = float((gx * gx).mean())
    gyy = float((gy * gy).mean())
    gxy = float((gx * gy).mean())
    trace = gxx + gyy
    tmp = ((gxx - gyy) ** 2 + 4 * (gxy**2)) ** 0.5
    lam1 = 0.5 * (trace + tmp)
    lam2 = 0.5 * (trace - tmp)
    ratio = lam2 / lam1 if lam1 > 1e-9 else 0.0
    return {"lambda_max": lam1, "lambda_min": lam2, "ratio": ratio}


def _noise_estimate(arr: np.ndarray) -> float:
    # Estimate noise std from high-pass residual, focusing on flat regions.
    # Box blur
    padded = np.pad(arr, 1, mode="reflect")
    blur = (
        padded[:-2, :-2]
        + padded[1:-1, :-2]
        + padded[2:, :-2]
        + padded[:-2, 1:-1]
        + padded[1:-1, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, 2:]
        + padded[2:, 2:]
    ) / 9.0
    residual = arr - blur
    gx, gy = np.gradient(arr)
    grad_mag = np.hypot(gx, gy)
    flat_mask = grad_mag < np.percentile(grad_mag, 30)
    if flat_mask.any():
        return float(residual[flat_mask].std())
    return float(residual.std())


_FACE_APP = None
_MP_FACE = None


def _get_face_app(face_cfg: Dict):
    backend = (face_cfg.get("backend") or "mediapipe").lower()
    if backend == "mediapipe":
        return _get_mp_face()
    return _get_insightface(face_cfg)


def _get_insightface(face_cfg: Dict):
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    if not FaceAnalysis:
        return None
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(
            ctx_id=face_cfg.get("ctx_id", 0),
            det_size=(face_cfg.get("det_size", 640), face_cfg.get("det_size", 640)),
        )
        _FACE_APP = app
        return app
    except Exception as exc:  # pragma: no cover - optional init
        print(f"InsightFace unavailable: {exc}", file=sys.stderr)
        return None


def _get_mp_face():
    global _MP_FACE
    if _MP_FACE is not None:
        return _MP_FACE
    if mp is None:
        return None
    try:
        _MP_FACE = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        return _MP_FACE
    except Exception as exc:
        print(f"Mediapipe face detection unavailable: {exc}", file=sys.stderr)
        return None


def _brightness_stats(arr: np.ndarray) -> Dict[str, float]:
    norm = arr / 255.0
    return {
        "mean": float(norm.mean()),
        "shadows": float((norm < 0.1).mean()),
        "highlights": float((norm > 0.9).mean()),
    }


def _composition_score(arr: np.ndarray) -> float:
    # Heuristic: weight intensity near rule-of-thirds intersections.
    h, w = arr.shape
    y, x = np.indices(arr.shape)
    weight = arr + 1e-6
    cx = (x * weight).sum() / (weight.sum() * w)
    cy = (y * weight).sum() / (weight.sum() * h)
    thirds = np.array([1 / 3, 2 / 3])
    dx = float(np.min(np.abs(thirds - cx)))
    dy = float(np.min(np.abs(thirds - cy)))
    score = 1.0 - min(1.0, (dx + dy))  # 0..1 range-ish
    return score


def _phash(preview_path: pathlib.Path, hash_size: int = 8) -> Optional[int]:
    if not Image:
        return None
    img = (
        Image.open(preview_path)
        .convert("L")
        .resize((hash_size, hash_size), Image.LANCZOS)
    )
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _suggest_keep(
    sharpness: float,
    tenengrad: float,
    motion_ratio: float,
    noise: float,
    brightness_mean: float,
    duplicate: bool,
    cfg: Dict,
) -> Tuple[bool, List[str]]:
    reasons = []
    keep = True
    if sharpness < cfg.get("sharpness_min", 12.0):
        keep = False
        reasons.append("soft")
    if tenengrad < cfg.get("tenengrad_min", 30_000.0):
        keep = False
        reasons.append("low contrast focus")
    if motion_ratio < cfg.get("motion_ratio_min", 0.25):
        keep = False
        reasons.append("motion blur")
    if noise > cfg.get("noise_std_max", 12.0):
        keep = False
        reasons.append("noisy")
    if brightness_mean < cfg.get("brightness_min", 0.08):
        keep = False
        reasons.append("too dark")
    if brightness_mean > cfg.get("brightness_max", 0.92):
        keep = False
        reasons.append("too bright")
    if duplicate:
        keep = False
        reasons.append("duplicate")
    return keep, reasons


def analyze_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    preview_cfg = cfg.get("preview", {})
    preview_cfg.setdefault("long_edge", 2048)
    preview_cfg.setdefault("format", "webp")
    preview_cfg.setdefault("quality", 85)
    analysis_cfg = cfg.get("analysis", {})
    exclude = cfg.get("exclude_dirs", [])
    exclude = list(
        set(
            exclude
            + [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]
        )
    )
    files = find_arw_files(cfg["input_dir"], exclude_dirs=exclude)
    if not files:
        print("No .ARW files found")
        return
    if np is None or not Image:
        print("numpy and Pillow are required for analysis", file=sys.stderr)
        return

    preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
    results: List[Dict] = []
    bar = _progress_bar(len(files), "Analyze")

    def worker(path: pathlib.Path) -> Optional[Dict]:
        preview_path = _preview_path_for(path, preview_dir, preview_cfg)
        if not preview_path.exists():
            print(f"Skipping (preview missing) {path}")
            return None
        exif = read_exif(path)
        dt = capture_date(
            exif, fallback=_dt.datetime.fromtimestamp(path.stat().st_mtime)
        )
        arr = _open_preview_array(preview_path)
        if arr is None:
            return None
        sharp = _variance_of_laplacian(arr)
        teneng = _tenengrad(arr)
        tensor = _structure_tensor_ratio(arr)
        noise = _noise_estimate(arr)
        bright = _brightness_stats(arr)
        comp = _composition_score(arr)
        phash = _phash(preview_path)
        face_info = None
        face_cfg = analysis_cfg.get("face", {})
        if face_cfg.get("enabled", False):
            app = _get_face_app(face_cfg)
            if app:
                rgb_full = _open_preview_rgb(preview_path, size=None)
                if rgb_full is not None:
                    faces = []
                    if mp and isinstance(app, mp.solutions.face_detection.FaceDetection):
                        results_mp = app.process(rgb_full.astype(np.uint8))
                        if results_mp.detections:
                            h, w, _ = rgb_full.shape
                            for det in results_mp.detections:
                                bbox = det.location_data.relative_bounding_box
                                x1 = int(bbox.xmin * w)
                                y1 = int(bbox.ymin * h)
                                x2 = int((bbox.xmin + bbox.width) * w)
                                y2 = int((bbox.ymin + bbox.height) * h)
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x2)
                                y2 = min(h, y2)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                face_gray = arr[y1:y2, x1:x2]
                                face_sharp = _variance_of_laplacian(face_gray) if face_gray.size else 0.0
                                faces.append(
                                    {
                                        "bbox": [x1, y1, x2, y2],
                                        "score": float(det.score[0]) if det.score else 0.0,
                                        "sharpness": face_sharp,
                                    }
                                )
                    elif FaceAnalysis and hasattr(app, "get"):
                        bgr = rgb_full[:, :, ::-1].astype(np.uint8)
                        dets = app.get(bgr)
                        for f in dets:
                            box = [float(x) for x in f.bbox.tolist()]
                            x1, y1, x2, y2 = map(int, box)
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(bgr.shape[1], x2)
                            y2 = min(bgr.shape[0], y2)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            face_gray = arr[y1:y2, x1:x2]
                            face_sharp = _variance_of_laplacian(face_gray) if face_gray.size else 0.0
                            faces.append({"bbox": box, "score": float(f.det_score), "sharpness": face_sharp})
                    if faces:
                        best = max(faces, key=lambda d: d["sharpness"])
                        face_info = {
                            "count": len(faces),
                            "best_sharpness": best["sharpness"],
                            "best_score": best["score"],
                            "faces": faces,
                        }
        return {
            "path": str(path),
            "preview": str(preview_path),
            "capture_time": dt.isoformat(),
            "capture_ts": dt.timestamp(),
            "sharpness": sharp,
            "tenengrad": teneng,
            "motion_ratio": tensor["ratio"],
            "noise": noise,
            "brightness": bright,
            "composition": comp,
            "phash": phash,
            "exif": exif,
            "faces": face_info,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg.get("concurrency", 4)
    ) as pool:
        futures = {pool.submit(worker, p): p for p in files}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
            bar.update(1)
    bar.close()
    # Ensure deterministic ordering by filename
    results.sort(key=lambda r: r["path"])
    # Duplicate detection: group by near-identical phash within a time window.
    dup_threshold = int(analysis_cfg.get("duplicate_hamming", 6))
    window_sec = float(analysis_cfg.get("duplicate_window_seconds", 8))
    entries: List[Tuple[int, int, float]] = []  # (index, phash, ts)
    for idx, r in enumerate(results):
        if r["phash"] is not None and r.get("capture_ts") is not None:
            entries.append((idx, int(r["phash"]), float(r["capture_ts"])))
    # sort by capture time so we only compare to prior shots in the window
    entries.sort(key=lambda t: t[2])

    parent = {idx: idx for idx, _, _ in entries}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(entries)):
        idx_i, h_i, ts_i = entries[i]
        # look backwards only while within window
        j = i - 1
        while j >= 0:
            idx_j, h_j, ts_j = entries[j]
            if ts_i - ts_j > window_sec:
                break
            if _hamming(h_i, h_j) <= dup_threshold:
                union(idx_i, idx_j)
            j -= 1

    groups: Dict[int, List[int]] = {}
    for idx, _, _ in entries:
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    # Map duplicates and ensure at least one keep per group
    duplicate_indexes = set()
    ssim_scores: Dict[Tuple[int, int], float] = {}
    psnr_scores: Dict[Tuple[int, int], float] = {}
    rgb_cache: Dict[int, np.ndarray] = {}
    use_similarity = skimage_ssim is not None and skimage_psnr is not None

    for members in groups.values():
        if len(members) < 2:
            continue
        # Sort members by sharpness descending; keep first by default
        members_sorted = sorted(
            members, key=lambda i: results[i]["sharpness"], reverse=True
        )
        keeper = members_sorted[0]
        for m in members:
            results[m]["duplicate_group"] = int(keeper)
            if m != keeper:
                duplicate_indexes.add(m)
                results[m]["duplicate_of"] = results[keeper]["path"]
                k = results[keeper]
                candidate = results[m]
                reason_parts = []
                if candidate["sharpness"] < k["sharpness"]:
                    reason_parts.append(
                        f"sharpness {candidate['sharpness']:.1f} < {k['sharpness']:.1f}"
                    )
                if (
                    candidate.get("tenengrad") is not None
                    and k.get("tenengrad") is not None
                ):
                    if candidate["tenengrad"] < k["tenengrad"]:
                        reason_parts.append(
                            f"contrast {candidate['tenengrad']:.0f} < {k['tenengrad']:.0f}"
                        )
                if (
                    candidate.get("motion_ratio") is not None
                    and k.get("motion_ratio") is not None
                ):
                    if candidate["motion_ratio"] < k["motion_ratio"]:
                        reason_parts.append(
                            f"motion ratio {candidate['motion_ratio']:.2f} < {k['motion_ratio']:.2f}"
                        )
                if candidate.get("noise") is not None and k.get("noise") is not None:
                    if candidate["noise"] > k["noise"]:
                        reason_parts.append(
                            f"noise {candidate['noise']:.1f} > {k['noise']:.1f}"
                        )

                if use_similarity:
                    # compute SSIM/PSNR on downscaled RGB previews
                    if keeper not in rgb_cache:
                        rgb_cache[keeper] = _open_preview_rgb(
                            pathlib.Path(k["preview"])
                        ) or np.zeros((1, 1, 3), dtype=np.float32)
                    if m not in rgb_cache:
                        rgb_cache[m] = _open_preview_rgb(
                            pathlib.Path(candidate["preview"])
                        ) or np.zeros((1, 1, 3), dtype=np.float32)
                    a = rgb_cache[keeper]
                    b = rgb_cache[m]
                    try:
                        ssim_val = float(
                            skimage_ssim(a, b, channel_axis=2, data_range=255)
                        )
                        psnr_val = float(skimage_psnr(a, b, data_range=255))
                        ssim_scores[(keeper, m)] = ssim_val
                        psnr_scores[(keeper, m)] = psnr_val
                        reason_parts.append(f"ssim={ssim_val:.3f}, psnr={psnr_val:.1f}")
                    except Exception:
                        pass

                results[m]["duplicate_reason"] = (
                    "; ".join(reason_parts) if reason_parts else "lower ranked in set"
                )

    # Suggestions
    for idx, r in enumerate(results):
        dup = idx in duplicate_indexes
        keep, reasons = _suggest_keep(
            r["sharpness"],
            r.get("tenengrad", 0.0),
            r.get("motion_ratio", 1.0),
            r.get("noise", 0.0),
            r["brightness"]["mean"],
            dup,
            analysis_cfg,
        )
        if dup and r.get("duplicate_reason"):
            reasons = [
                "duplicate: " + r["duplicate_reason"] if x == "duplicate" else x
                for x in reasons
            ]
        r["suggest_keep"] = keep
        r["decision"] = "keep" if keep else "discard"
        r["reasons"] = reasons

    # Enforce: each duplicate group must have at least one keep
    groups_by_root: Dict[int, List[int]] = {}
    for idx, r in enumerate(results):
        if "duplicate_group" in r:
            groups_by_root.setdefault(int(r["duplicate_group"]), []).append(idx)
    for root, members in groups_by_root.items():
        keeps = [i for i in members if results[i]["decision"] == "keep"]
        if keeps:
            continue
        # Pick sharpest as keeper
        keeper = max(members, key=lambda i: results[i]["sharpness"])
        results[keeper]["decision"] = "keep"
        results[keeper]["reasons"].append("kept to avoid discarding all duplicates")

    output_json = pathlib.Path(analysis_cfg.get("results_path", "./analysis.json"))
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = pathlib.Path(analysis_cfg.get("report_path", "./report.html"))
    _write_html_report(results, report_path)
    print(f"Analysis written to {output_json}")
    print(f"HTML report written to {report_path}")


def _write_html_report(results: List[Dict], path: pathlib.Path) -> None:
    data_json = json.dumps(results)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ARW Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 1rem; }}
    h1 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 0.5rem; border-bottom: 1px solid #1e293b; vertical-align: top; }}
    img.preview {{ max-width: 200px; height: auto; border-radius: 4px; cursor: zoom-in; }}
    .lightbox {{
      position: fixed; inset: 0; background: rgba(0,0,0,0.8);
      display: flex; align-items: center; justify-content: center;
      opacity: 0; pointer-events: none; transition: opacity 0.2s ease;
      z-index: 1000;
    }}
    .lightbox.open {{ opacity: 1; pointer-events: all; }}
    .lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 6px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    .controls button {{ margin-right: 0.25rem; }}
    .keep {{ color: #22c55e; font-weight: 600; }}
    .discard {{ color: #ef4444; font-weight: 600; }}
    .badge {{ padding: 0.1rem 0.4rem; border-radius: 4px; background: #1f2937; margin-right: 0.3rem; }}
    .row {{ background: #0b1223; }}
    .row:nth-child(odd) {{ background: #0c162b; }}
    .reasons {{ margin-top: 0.35rem; color: #94a3b8; font-size: 0.9rem; }}
    .summary {{ margin: 0.25rem 0 0.75rem 0; font-weight: 600; }}
    #loading {{
      position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.7); color: #e2e8f0; z-index: 2000;
      font-size: 1.2rem; gap: 0.5rem;
    }}
    .spinner {{
      width: 24px; height: 24px; border-radius: 999px; border: 3px solid #1e293b; border-top-color: #38bdf8;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <h1>ARW Analysis Report</h1>
  <p>Toggle decisions per photo and export as JSON.</p>
  <div id="summary" class="summary"></div>
  <div style="margin-bottom: 1rem;">
    <button id="export">Download decisions JSON</button>
  </div>
  <div id="loading"><div class="spinner"></div>Loading…</div>
  <div class="lightbox" id="lightbox">
    <img id="lightbox-img" src="" alt="Preview" />
  </div>
  <table>
    <thead>
      <tr>
        <th>Preview</th>
        <th>File</th>
        <th>Scores</th>
        <th>Decision</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="stats" style="margin-top: 1rem;"></div>
  <script id="data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("data").textContent);
    const tbody = document.getElementById("rows");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    lightbox.onclick = () => {{ lightbox.classList.remove("open"); }};
    const groups = {{}};
    data.forEach((item, idx) => {{
      if (item.duplicate_group !== undefined) {{
        const g = item.duplicate_group;
        if (!groups[g]) groups[g] = [];
        groups[g].push(idx);
      }}
    }});

    function hasOtherKeep(groupId, excludeIdx) {{
      if (!groups[groupId]) return false;
      return groups[groupId].some(i => i !== excludeIdx && data[i].decision === "keep");
    }}

    function render() {{
      tbody.innerHTML = "";
      const stats = {{ total: data.length, keep: 0, discard: 0 }};
      data.forEach((item, idx) => {{
        const tr = document.createElement("tr");
        tr.className = "row";

        const tdImg = document.createElement("td");
        const img = document.createElement("img");
        img.className = "preview";
        img.src = item.preview.replace(/\\\\/g, "/");
        img.loading = "lazy";
        img.onclick = () => {{
          lightboxImg.src = img.src;
          lightbox.classList.add("open");
        }};
        tdImg.appendChild(img);

        const tdFile = document.createElement("td");
        tdFile.innerHTML = `<div>{'{'}${{item.path}}{'}'}</div>`;
        if (item.duplicate_of) {{
          const badge = document.createElement("div");
          badge.className = "badge";
          badge.textContent = "Duplicate of: " + item.duplicate_of;
          tdFile.appendChild(badge);
        }}

        const tdScores = document.createElement("td");
        const faceBadges = item.faces
          ? `<div class="badge">Faces: ${{item.faces.count}}</div><div class="badge">Face sharpness: ${{item.faces.best_sharpness?.toFixed(1) ?? 'n/a'}}</div>`
          : "";
        tdScores.innerHTML = `
          <div class="badge">Sharpness: ${{item.sharpness.toFixed(1)}}</div>
          <div class="badge">Tenengrad: ${{item.tenengrad?.toFixed(0) ?? 'n/a'}}</div>
          <div class="badge">Motion ratio: ${{item.motion_ratio?.toFixed(2) ?? 'n/a'}}</div>
          <div class="badge">Noise std: ${{item.noise?.toFixed(1) ?? 'n/a'}}</div>
          ${{faceBadges}}
          <div class="badge">Brightness: ${{(item.brightness.mean * 100).toFixed(0)}}%</div>
          <div class="badge">Shadows: ${{(item.brightness.shadows * 100).toFixed(0)}}%</div>
          <div class="badge">Highlights: ${{(item.brightness.highlights * 100).toFixed(0)}}%</div>
          <div class="badge">Composition: ${{item.composition.toFixed(2)}}</div>
        `;

        const tdDecision = document.createElement("td");
        const status = document.createElement("div");
        status.className = item.decision === "keep" ? "keep" : "discard";
        status.textContent = item.decision.toUpperCase();
        const controls = document.createElement("div");
        controls.className = "controls";
        const btnKeep = document.createElement("button");
        btnKeep.textContent = "Keep";
        btnKeep.onclick = () => {{ item.decision = "keep"; render(); }};
        const btnDrop = document.createElement("button");
        btnDrop.textContent = "Discard";
        btnDrop.onclick = () => {{
          if (item.duplicate_group !== undefined && !hasOtherKeep(item.duplicate_group, idx)) {{
            alert("At least one photo in a duplicate set must be kept.");
            return;
          }}
          item.decision = "discard";
          render();
        }};
        controls.appendChild(btnKeep);
        controls.appendChild(btnDrop);
        tdDecision.appendChild(status);
        tdDecision.appendChild(controls);
        if (item.reasons && item.reasons.length) {{
          const reasons = document.createElement("div");
          reasons.className = "reasons";
          reasons.textContent = "Reasons: " + item.reasons.join(", ");
          tdDecision.appendChild(reasons);
        }}

        tr.appendChild(tdImg);
        tr.appendChild(tdFile);
        tr.appendChild(tdScores);
        tr.appendChild(tdDecision);
        tbody.appendChild(tr);

        if (item.decision === "keep") stats.keep += 1; else stats.discard += 1;
      }});

      const statsDiv = document.getElementById("stats");
      const keepPct = stats.total ? Math.round((stats.keep / stats.total) * 100) : 0;
      const discardPct = stats.total ? Math.round((stats.discard / stats.total) * 100) : 0;
      statsDiv.innerHTML = `Keeps: ${{keepPct}}% (${{stats.keep}}/${{stats.total}}) &nbsp; Discards: ${{discardPct}}% (${{stats.discard}}/${{stats.total}})`;
      const summaryDiv = document.getElementById("summary");
      summaryDiv.textContent = `Discarded: ${{discardPct}}% (${{stats.discard}} of ${{stats.total}})`;

      document.getElementById("loading").style.display = "none";
    }}

    render();

    document.getElementById("export").onclick = () => {{
      const blob = new Blob([JSON.stringify(data, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "decisions.json";
      a.click();
      URL.revokeObjectURL(url);
    }};
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sony .ARW preprocessing and sorting")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="Path to YAML config file", default=None)

    scan = sub.add_parser("scan", help="List .ARW files and basic EXIF info")
    add_common(scan)
    scan.add_argument("--json", action="store_true", help="Output JSON metadata")
    scan.set_defaults(func=scan_command)

    prev = sub.add_parser("previews", help="Generate quick previews")
    add_common(prev)
    prev.set_defaults(func=previews_command)

    sort = sub.add_parser("sort", help="Copy/move files into structured folders")
    add_common(sort)
    sort.add_argument(
        "--apply",
        action="store_true",
        help="Perform the operations (default is dry run)",
    )
    sort.set_defaults(func=sort_command)

    analyze = sub.add_parser("analyze", help="Score images and emit JSON + HTML report")
    add_common(analyze)
    analyze.set_defaults(func=analyze_command)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "sort" and not args.apply:
        args.dry_run = True
    elif args.command == "sort":
        args.dry_run = False
    args.func(args)


if __name__ == "__main__":
    main()
