"""Analysis pipeline for scoring and reporting RAW images."""

import concurrent.futures
import datetime as _dt
import json
import pathlib
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, TypedDict, cast

import numpy as np
from PIL import Image

from .config import AnalysisConfig, AppConfig, FaceConfig, PreviewConfig
from .discovery import (
    ExifData,
    capture_date,
    exif_orientation,
    read_exif,
)
from .faces import FaceSummary, detect_faces
from .metrics import (
    BrightnessStats,
    brightness_stats,
    composition_score,
    hamming,
    noise_estimate,
    phash,
    structure_tensor_ratio,
    tenengrad,
    variance_of_laplacian,
)
from .preview import open_preview_rgb, preview_path_for
from .report import write_html_report

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
except Exception:  # pragma: no cover
    skimage_ssim = None
    skimage_psnr = None

ProgressCallback = Optional[Callable[[int], None]]


class AnalysisResult(TypedDict, total=False):
    """Single frame analysis result."""

    path: str
    preview: str
    capture_time: str
    capture_ts: float
    sharpness: float
    sharpness_center: float
    tenengrad: float
    motion_ratio: float
    noise: float
    brightness: BrightnessStats
    composition: float
    phash: Optional[int]
    exif: ExifData
    faces: FaceSummary
    duplicate_group: int
    duplicate_of: str
    duplicate_reason: str
    quality_score: float
    suggest_keep: bool
    decision: str
    reasons: List[str]


@dataclass(frozen=True)
class FrameMetrics:
    """Container for the measurements calculated for a frame."""

    sharpness: float
    sharpness_center: float
    tenengrad: float
    motion_ratio: float
    noise: float
    brightness_mean: float
    shadows: float
    highlights: float
    composition: float


@dataclass(frozen=True)
class Thresholds:
    """Aggregated thresholds derived from configuration and shooting context."""

    sharpness_min: float
    center_sharpness_min: float
    tenengrad_min: float
    motion_ratio_min: float
    noise_std_max: float
    brightness_min: float
    brightness_max: float
    shadows_min: float
    shadows_max: float
    highlights_min: float
    highlights_max: float

    @classmethod
    def from_config(cls, cfg: AnalysisConfig, exif: ExifData) -> "Thresholds":
        iso = _parse_iso(exif) or 0.0
        shutter = _parse_shutter_seconds(exif) or 0.0

        sharp_min = float(cfg.get("sharpness_min", 12.0))
        teneng_min = float(cfg.get("tenengrad_min", 30_000.0))
        motion_min = float(cfg.get("motion_ratio_min", 0.25))

        sharp_factor = 1.0
        teneng_factor = 1.0
        motion_factor = 1.0

        low_iso, fast_shutter = 400.0, 1 / 500.0
        high_iso, slow_shutter = 3200.0, 1 / 50.0

        if iso and shutter:
            if iso <= low_iso and shutter <= fast_shutter:
                sharp_factor = 1.2
                teneng_factor = 1.15
                motion_factor = 1.1
            elif iso >= high_iso and shutter >= slow_shutter:
                sharp_factor = 0.8
                teneng_factor = 0.8
                motion_factor = 0.7

        sharpness_min = sharp_min * sharp_factor
        return cls(
            sharpness_min=sharpness_min,
            center_sharpness_min=float(
                cfg.get("center_sharpness_min", sharpness_min * 1.2)
            ),
            tenengrad_min=teneng_min * teneng_factor,
            motion_ratio_min=motion_min * motion_factor,
            noise_std_max=float(cfg.get("noise_std_max", 12.0)),
            brightness_min=float(cfg.get("brightness_min", 0.08)),
            brightness_max=float(cfg.get("brightness_max", 0.92)),
            shadows_min=float(cfg.get("shadows_min", 0.0)),
            shadows_max=float(cfg.get("shadows_max", 0.5)),
            highlights_min=float(cfg.get("highlights_min", 0.0)),
            highlights_max=float(cfg.get("highlights_max", 0.1)),
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    """Normalized score per metric and overall quality score."""

    scores: Dict[str, float]
    quality_score: float


def _parse_iso(exif: ExifData) -> Optional[float]:
    """Extract ISO value from EXIF dictionary."""
    raw = exif.get("EXIF ISOSpeedRatings") or exif.get("EXIF PhotographicSensitivity")
    if raw is None:
        return None
    try:
        return float(str(raw).split()[0])
    except Exception:
        return None


def _parse_shutter_seconds(exif: ExifData) -> Optional[float]:
    """Extract shutter speed in seconds from EXIF dictionary."""
    raw = exif.get("EXIF ExposureTime")
    if raw is None:
        return None
    s = str(raw).strip()
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return None


def _clamp01(v: float) -> float:
    """Clamp a float into the inclusive [0, 1] range."""
    return max(0.0, min(1.0, v))


def _hard_fail_thresholds(cfg: AnalysisConfig) -> Dict[str, float]:
    """Return configured hard-fail ratios for each metric."""
    return {
        "sharp": float(cfg.get("hard_fail_sharp_ratio", 0.55)),
        "sharp_center": float(cfg.get("hard_fail_sharp_center_ratio", 0.55)),
        "teneng": float(cfg.get("hard_fail_teneng_ratio", 0.55)),
        "motion": float(cfg.get("hard_fail_motion_ratio", 0.55)),
        "brightness": float(cfg.get("hard_fail_brightness_ratio", 0.5)),
        "noise": float(cfg.get("hard_fail_noise_ratio", 0.45)),
        "shadows": float(cfg.get("hard_fail_shadows_ratio", 0.5)),
        "highlights": float(cfg.get("hard_fail_highlights_ratio", 0.5)),
        "composition": float(cfg.get("hard_fail_composition_ratio", 0.4)),
    }


def _score_metrics(metrics: FrameMetrics, thresholds: Thresholds) -> ScoreBreakdown:
    """Normalize each metric and compute the combined quality score."""
    sharp_score = _clamp01(metrics.sharpness / max(thresholds.sharpness_min, 1e-6))
    sharp_center_score = _clamp01(
        metrics.sharpness_center / max(thresholds.center_sharpness_min, 1e-6)
    )
    teneng_score = _clamp01(metrics.tenengrad / max(thresholds.tenengrad_min, 1e-6))
    motion_score = _clamp01(
        metrics.motion_ratio / max(thresholds.motion_ratio_min, 1e-6)
    )
    noise_score = _clamp01(
        thresholds.noise_std_max / (thresholds.noise_std_max + max(metrics.noise, 1e-6))
    )
    brightness_mid = (thresholds.brightness_min + thresholds.brightness_max) / 2
    brightness_half = (thresholds.brightness_max - thresholds.brightness_min) / 2
    brightness_score = _clamp01(
        1 - abs(metrics.brightness_mean - brightness_mid) / max(brightness_half, 1e-6)
    )

    def range_score(val: float, lo: float, hi: float) -> float:
        if val < lo:
            return _clamp01(1 - (lo - val) / max(lo, 1e-6))
        if val > hi:
            return _clamp01(1 - (val - hi) / max(1 - hi, 1e-6))
        return 1.0

    shadows_score = range_score(
        metrics.shadows, thresholds.shadows_min, thresholds.shadows_max
    )
    highlights_score = range_score(
        metrics.highlights, thresholds.highlights_min, thresholds.highlights_max
    )
    comp_score = _clamp01(metrics.composition)

    weights = {
        "sharp": 1.5,
        "sharp_center": 1.0,
        "teneng": 1.0,
        "motion": 1.0,
        "noise": 0.7,
        "brightness": 0.8,
        "shadows": 0.5,
        "highlights": 0.5,
        "composition": 0.4,
    }

    scores = {
        "sharp": sharp_score,
        "sharp_center": sharp_center_score,
        "teneng": teneng_score,
        "motion": motion_score,
        "brightness": brightness_score,
        "noise": noise_score,
        "shadows": shadows_score,
        "highlights": highlights_score,
        "composition": comp_score,
    }

    weighted_sum = sum(scores[key] * weights[key] for key in scores)
    total_weights = sum(weights.values())
    quality_score = weighted_sum / total_weights if total_weights else 0.0
    return ScoreBreakdown(scores=scores, quality_score=quality_score)


def _quality_reasons(
    metrics: FrameMetrics,
    thresholds: Thresholds,
    scores: Dict[str, float],
    hard_cfg: Dict[str, float],
    quality_score: float,
    cutoff: float,
) -> List[str]:
    """Compile human-readable reasons for discarding a frame."""
    reasons: List[str] = []
    if quality_score < cutoff:
        reasons.append(f"quality score {quality_score:.2f} below {cutoff:.2f}")

    def add_reason(condition: bool, text: str) -> None:
        if condition:
            reasons.append(text)

    add_reason(
        scores["sharp"] < hard_cfg["sharp"]
        or metrics.sharpness < thresholds.sharpness_min,
        f"sharpness {metrics.sharpness:.1f} < min {thresholds.sharpness_min:.1f}",
    )
    add_reason(
        scores["sharp_center"] < hard_cfg["sharp_center"]
        or metrics.sharpness_center < thresholds.center_sharpness_min,
        (
            "center sharpness "
            f"{metrics.sharpness_center:.1f} < "
            f"min {thresholds.center_sharpness_min:.1f}"
        ),
    )
    add_reason(
        scores["teneng"] < hard_cfg["teneng"]
        or metrics.tenengrad < thresholds.tenengrad_min,
        f"contrast {metrics.tenengrad:.0f} < min {thresholds.tenengrad_min:.0f}",
    )
    add_reason(
        scores["motion"] < hard_cfg["motion"]
        or metrics.motion_ratio < thresholds.motion_ratio_min,
        f"motion ratio {metrics.motion_ratio:.2f} < min {thresholds.motion_ratio_min:.2f}",
    )
    add_reason(
        scores["noise"] < hard_cfg["noise"] or metrics.noise > thresholds.noise_std_max,
        f"noise {metrics.noise:.1f} > max {thresholds.noise_std_max:.1f}",
    )

    if scores["brightness"] < hard_cfg["brightness"]:
        if metrics.brightness_mean < thresholds.brightness_min:
            reasons.append(
                f"brightness {metrics.brightness_mean:.2f} < min {thresholds.brightness_min:.2f}"
            )
        elif metrics.brightness_mean > thresholds.brightness_max:
            reasons.append(
                f"brightness {metrics.brightness_mean:.2f} > max {thresholds.brightness_max:.2f}"
            )
        else:
            reasons.append("poor exposure")
    if scores["shadows"] < hard_cfg["shadows"]:
        if metrics.shadows > thresholds.shadows_max:
            reasons.append(
                f"shadows {metrics.shadows:.2f} > max {thresholds.shadows_max:.2f}"
            )
        elif metrics.shadows < thresholds.shadows_min:
            reasons.append(
                f"shadows {metrics.shadows:.2f} < min {thresholds.shadows_min:.2f}"
            )
    if scores["highlights"] < hard_cfg["highlights"]:
        if metrics.highlights > thresholds.highlights_max:
            reasons.append(
                f"highlights {metrics.highlights:.2f} > max {thresholds.highlights_max:.2f}"
            )
        elif metrics.highlights < thresholds.highlights_min:
            reasons.append(
                f"highlights {metrics.highlights:.2f} < min {thresholds.highlights_min:.2f}"
            )
    if scores["composition"] < hard_cfg["composition"]:
        reasons.append(
            f"composition {metrics.composition:.2f} < min {hard_cfg['composition']:.2f}"
        )
    return reasons


def _frame_metrics_from_result(result: AnalysisResult) -> FrameMetrics:
    """Build a FrameMetrics instance from the raw analysis dictionary."""
    brightness: BrightnessStats = result["brightness"]
    return FrameMetrics(
        sharpness=result["sharpness"],
        sharpness_center=result.get("sharpness_center", result["sharpness"]),
        tenengrad=result.get("tenengrad", 0.0),
        motion_ratio=result.get("motion_ratio", 1.0),
        noise=result.get("noise", 0.0),
        brightness_mean=brightness["mean"],
        shadows=brightness.get("shadows", 0.0),
        highlights=brightness.get("highlights", 0.0),
        composition=result.get("composition", 0.0) or 0.0,
    )


def _suggest_keep(
    metrics: FrameMetrics, duplicate: bool, cfg: AnalysisConfig, exif: ExifData
) -> Tuple[bool, List[str], float]:
    """Return keep/discard suggestion, reasons, and quality score for a frame."""
    thresholds = Thresholds.from_config(cfg, exif)
    hard_cfg = _hard_fail_thresholds(cfg)
    cutoff = float(cfg.get("quality_score_min", 0.75))
    score_breakdown = _score_metrics(metrics, thresholds)
    hard_fail = any(
        score_breakdown.scores[key] < hard_cfg[key] for key in score_breakdown.scores
    )
    threshold_fail = any(
        (
            metrics.sharpness < thresholds.sharpness_min,
            metrics.sharpness_center < thresholds.center_sharpness_min,
            metrics.tenengrad < thresholds.tenengrad_min,
            metrics.motion_ratio < thresholds.motion_ratio_min,
            metrics.noise > thresholds.noise_std_max,
            metrics.brightness_mean < thresholds.brightness_min,
            metrics.brightness_mean > thresholds.brightness_max,
            metrics.shadows < thresholds.shadows_min,
            metrics.shadows > thresholds.shadows_max,
            metrics.highlights < thresholds.highlights_min,
            metrics.highlights > thresholds.highlights_max,
        )
    )
    keep = score_breakdown.quality_score >= cutoff and not hard_fail and not threshold_fail
    reasons = _quality_reasons(
        metrics,
        thresholds,
        score_breakdown.scores,
        hard_cfg,
        score_breakdown.quality_score,
        cutoff,
    )
    if duplicate:
        reasons.append("duplicate")
        keep = False
    return keep, reasons, score_breakdown.quality_score


def _load_preview_arrays(
    preview_path: pathlib.Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[int]]]:
    """Load preview once and return RGB uint8 array, grayscale float32 array, and pHash."""
    if Image is None:
        return None
    try:
        with Image.open(preview_path) as img_handle:
            img_rgb = img_handle.convert("RGB")
            rgb_arr = np.array(img_rgb, dtype=np.uint8)
            # Fast luminance approximation without an extra PIL convert call.
            gray_arr = (
                0.299 * rgb_arr[:, :, 0]
                + 0.587 * rgb_arr[:, :, 1]
                + 0.114 * rgb_arr[:, :, 2]
            ).astype(np.float32)
            hash_val = phash(preview_path, gray_array=gray_arr)
    except Exception:
        return None
    return rgb_arr, gray_arr, hash_val


def _analyze_single_file(
    path: pathlib.Path,
    preview_dir: pathlib.Path,
    preview_cfg: PreviewConfig,
    analysis_cfg: AnalysisConfig,
) -> Optional[AnalysisResult]:
    """Analyze a single RAW file using its generated preview."""
    preview_path = preview_path_for(path, preview_dir, preview_cfg)
    if not preview_path.exists():
        return None

    exif = read_exif(path)
    dt = capture_date(exif, fallback=_dt.datetime.fromtimestamp(path.stat().st_mtime))
    preview_data = _load_preview_arrays(preview_path)
    if preview_data is None:
        return None
    rgb_full, gray_arr, hash_val = preview_data

    sharpness = variance_of_laplacian(gray_arr)
    center_slice = gray_arr[
        int(gray_arr.shape[0] * 0.25) : int(gray_arr.shape[0] * 0.75),
        int(gray_arr.shape[1] * 0.25) : int(gray_arr.shape[1] * 0.75),
    ]
    face_cfg = cast(FaceConfig, analysis_cfg.get("face") or {})
    face_info: Optional[FaceSummary] = None
    if face_cfg.get("enabled", False):
        face_info = detect_faces(
            preview_path,
            gray_arr,
            face_cfg,
            orientation=exif_orientation(exif),
            rgb_arr=rgb_full,
        )

    return cast(
        AnalysisResult,
        {
            "path": str(path),
            "preview": str(preview_path),
            "capture_time": dt.isoformat(),
            "capture_ts": dt.timestamp(),
            "sharpness": sharpness,
            "sharpness_center": variance_of_laplacian(center_slice),
            "tenengrad": tenengrad(gray_arr),
            "motion_ratio": structure_tensor_ratio(gray_arr)["ratio"],
            "noise": noise_estimate(gray_arr),
            "brightness": brightness_stats(gray_arr),
            "composition": composition_score(gray_arr),
            "phash": hash_val,
            "exif": exif,
            "faces": face_info,
        },
    )


def analyze_files(
    cfg: AppConfig,
    files: List[pathlib.Path],
    preview_dir: pathlib.Path,
    preview_cfg: PreviewConfig,
    progress_cb: ProgressCallback = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[AnalysisResult]:
    """Analyze a collection of RAW files and return structured metrics."""
    analysis_cfg = cast(AnalysisConfig, cfg.get("analysis") or {})
    use_similarity = skimage_ssim is not None and skimage_psnr is not None
    concurrency = int(cfg.get("concurrency", 4) or 4)
    results = _run_analysis_workers(
        files,
        preview_dir,
        preview_cfg,
        analysis_cfg,
        concurrency=concurrency,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    duplicate_indexes = _label_duplicates(results, analysis_cfg, use_similarity)
    _apply_quality_decisions(results, analysis_cfg, duplicate_indexes)
    return results


def _run_analysis_workers(
    files: List[pathlib.Path],
    preview_dir: pathlib.Path,
    preview_cfg: PreviewConfig,
    analysis_cfg: AnalysisConfig,
    concurrency: int,
    progress_cb: ProgressCallback,
    cancel_event: Optional[threading.Event],
) -> List[AnalysisResult]:
    """Analyze files concurrently and return sorted results."""
    results: List[AnalysisResult] = []
    worker_count = max(1, concurrency)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
    futures = [
        pool.submit(_analyze_single_file, path, preview_dir, preview_cfg, analysis_cfg)
        for path in files
    ]
    cancelled = False
    try:
        for future in concurrent.futures.as_completed(futures):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                for job in futures:
                    job.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                break
            result = future.result()
            if result is not None:
                results.append(result)
            if progress_cb is not None:
                progress_cb(1)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not cancelled:
            pool.shutdown(wait=True)
    results.sort(key=lambda res: res["path"])
    return results


def _label_duplicates(
    results: List[AnalysisResult],
    analysis_cfg: AnalysisConfig,
    use_similarity: bool,
) -> Set[int]:
    """Detect duplicate frames and annotate results with grouping metadata."""
    dup_threshold = int(analysis_cfg.get("duplicate_hamming", 6))
    window_sec = float(analysis_cfg.get("duplicate_window_seconds", 8))
    bucket_bits = max(0, int(analysis_cfg.get("duplicate_bucket_bits", 8)))
    bucket_shift = max(0, 64 - bucket_bits) if bucket_bits else 0

    def bucket_key(hash_val: int) -> int:
        return hash_val >> bucket_shift if bucket_bits else 0

    entries: List[Tuple[int, int, float]] = []
    for idx, result in enumerate(results):
        if result["phash"] is not None and result.get("capture_ts") is not None:
            entries.append((idx, int(result["phash"]), float(result["capture_ts"])))
    entries.sort(key=lambda item: item[2])
    parent = {idx: idx for idx, __, ___ in entries}
    hash_by_idx = {idx: hash_val for idx, hash_val, __ in entries}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    active_window = deque()  # (result_idx, bucket, timestamp)
    bucket_index: Dict[int, List[int]] = defaultdict(list)

    for idx_i, hash_i, ts_i in entries:
        while active_window and ts_i - active_window[0][2] > window_sec:
            expired_idx, expired_bucket, __ = active_window.popleft()
            bucket_list = bucket_index.get(expired_bucket)
            if bucket_list:
                try:
                    bucket_list.remove(expired_idx)
                except ValueError:
                    pass
                if not bucket_list:
                    bucket_index.pop(expired_bucket, None)

        key = bucket_key(hash_i)
        for candidate_idx in list(bucket_index.get(key, [])):
            if hamming(hash_i, hash_by_idx[candidate_idx]) <= dup_threshold:
                union(idx_i, candidate_idx)

        active_window.append((idx_i, key, ts_i))
        bucket_index[key].append(idx_i)

    groups: Dict[int, List[int]] = {}
    for idx, __, ___ in entries:
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    duplicate_indexes = set()
    rgb_cache: Dict[int, np.ndarray] = {}

    def _similarity_reason(keeper_idx: int, candidate_idx: int) -> str:
        if not use_similarity:
            return ""
        keeper = results[keeper_idx]
        candidate = results[candidate_idx]
        if keeper_idx not in rgb_cache:
            arr_keep = open_preview_rgb(pathlib.Path(keeper["preview"]))
            rgb_cache[keeper_idx] = (
                arr_keep
                if arr_keep is not None
                else np.zeros((1, 1, 3), dtype=np.float32)
            )
        if candidate_idx not in rgb_cache:
            arr_dup = open_preview_rgb(pathlib.Path(candidate["preview"]))
            rgb_cache[candidate_idx] = (
                arr_dup
                if arr_dup is not None
                else np.zeros((1, 1, 3), dtype=np.float32)
            )
        try:
            ssim_val = float(
                skimage_ssim(
                    rgb_cache[keeper_idx],
                    rgb_cache[candidate_idx],
                    channel_axis=2,
                    data_range=255,
                )
            )
            psnr_val = float(
                skimage_psnr(
                    rgb_cache[keeper_idx], rgb_cache[candidate_idx], data_range=255
                )
            )
            return f"ssim={ssim_val:.3f}, psnr={psnr_val:.1f}"
        except Exception:
            return ""

    for members in groups.values():
        if len(members) < 2:
            continue
        members_sorted = sorted(
            members,
            key=lambda i: (
                results[i].get("sharpness", 0.0),
                results[i].get("tenengrad", 0.0) or 0.0,
                results[i].get("motion_ratio", 0.0) or 0.0,
            ),
            reverse=True,
        )
        keeper = members_sorted[0]
        keeper_result = results[keeper]
        for member in members:
            results[member]["duplicate_group"] = int(keeper)
            if member == keeper:
                continue
            duplicate_indexes.add(member)
            candidate = results[member]
            candidate["duplicate_of"] = keeper_result["path"]
            reason_parts: List[str] = []
            if candidate["sharpness"] < keeper_result["sharpness"]:
                reason_parts.append(
                    f"sharpness {candidate['sharpness']:.1f} < {keeper_result['sharpness']:.1f}"
                )
            if (
                candidate.get("tenengrad") is not None
                and keeper_result.get("tenengrad") is not None
            ):
                if candidate["tenengrad"] < keeper_result["tenengrad"]:
                    reason_parts.append(
                        f"contrast {candidate['tenengrad']:.0f} < {keeper_result['tenengrad']:.0f}"
                    )
            if (
                candidate.get("motion_ratio") is not None
                and keeper_result.get("motion_ratio") is not None
            ):
                if candidate["motion_ratio"] < keeper_result["motion_ratio"]:
                    reason_parts.append(
                        (
                            "motion ratio "
                            f"{candidate['motion_ratio']:.2f} < "
                            f"{keeper_result['motion_ratio']:.2f}"
                        )
                    )
            if (
                candidate.get("noise") is not None
                and keeper_result.get("noise") is not None
            ):
                if candidate["noise"] > keeper_result["noise"]:
                    reason_parts.append(
                        f"noise {candidate['noise']:.1f} > {keeper_result['noise']:.1f}"
                    )
            sim_reason = _similarity_reason(keeper, member)
            if sim_reason:
                reason_parts.append(sim_reason)

            candidate["duplicate_reason"] = (
                "; ".join(reason_parts) if reason_parts else "lower ranked in set"
            )

    return duplicate_indexes


def _to_windows_path(path_value: str) -> str:
    """Convert WSL-style /mnt/<drive>/ paths to Windows paths for host consumption."""
    if not isinstance(path_value, str):
        return path_value
    if len(path_value) >= 3 and path_value[1:3] == ":\\":
        return path_value
    parts = path_value.split("/")
    is_wsl_root = len(parts) > 3 and parts[0] == "" and parts[1] == "mnt"
    if is_wsl_root and len(parts[2]) == 1:
        drive = f"{parts[2].upper()}:"
        rest = [part for part in parts[3:] if part]
        return str(pathlib.PureWindowsPath(drive, *rest))
    return path_value


def _apply_quality_decisions(
    results: List[AnalysisResult],
    analysis_cfg: AnalysisConfig,
    duplicate_indexes: Set[int],
) -> None:
    """Apply quality scoring and final keep/discard decisions."""
    for idx, result in enumerate(results):
        is_duplicate = idx in duplicate_indexes
        metrics = _frame_metrics_from_result(result)
        exif = result.get("exif", {})
        keep, reasons, quality_score = _suggest_keep(
            metrics,
            is_duplicate,
            analysis_cfg,
            exif,
        )
        result["quality_score"] = quality_score
        if is_duplicate and result.get("duplicate_reason"):
            reasons = [
                (
                    "duplicate: " + result["duplicate_reason"]
                    if reason == "duplicate"
                    else reason
                )
                for reason in reasons
            ]
        result["suggest_keep"] = keep
        result["decision"] = "keep" if keep else "discard"
        result["reasons"] = reasons

    groups_by_root: Dict[int, List[int]] = {}
    for idx, result in enumerate(results):
        if "duplicate_group" in result:
            groups_by_root.setdefault(int(result["duplicate_group"]), []).append(idx)
    for members in groups_by_root.values():
        keepers = [idx for idx in members if results[idx]["decision"] == "keep"]
        if keepers:
            continue
        fallback = max(members, key=lambda idx: results[idx]["sharpness"])
        results[fallback]["decision"] = "keep"
        results[fallback]["reasons"].append("kept to avoid discarding all duplicates")


def _convert_paths_for_host(results: List[AnalysisResult]) -> List[AnalysisResult]:
    """Return a copy of results with paths converted to Windows-style when applicable."""
    converted: List[AnalysisResult] = []
    for entry in results:
        item = dict(entry)
        item["path"] = _to_windows_path(item["path"])
        item["preview"] = _to_windows_path(item["preview"])
        if "duplicate_of" in item:
            item["duplicate_of"] = _to_windows_path(item["duplicate_of"])
        converted.append(cast(AnalysisResult, item))
    return converted


def write_outputs(results: List[AnalysisResult], analysis_cfg: AnalysisConfig) -> None:
    """Persist analysis results to JSON and render the HTML report."""
    results_for_host = _convert_paths_for_host(results)
    output_json = pathlib.Path(analysis_cfg.get("results_path", "./analysis.json"))
    output_json.write_text(json.dumps(results_for_host, indent=2), encoding="utf-8")
    report_path = pathlib.Path(analysis_cfg.get("report_path", "./report.html"))
    write_html_report(results_for_host, report_path, {"analysis": analysis_cfg})
