import concurrent.futures
import datetime as _dt
import json
import pathlib
from typing import Dict, Iterable, List, Optional, Tuple


import numpy as np
from PIL import Image

from .discovery import capture_date, exif_orientation, find_arw_files, read_exif
from .faces import detect_faces
from .metrics import (
    brightness_stats,
    composition_score,
    hamming,
    noise_estimate,
    phash,
    structure_tensor_ratio,
    tenengrad,
    variance_of_laplacian,
)
from .preview import open_preview_gray, open_preview_rgb, preview_path_for
from .report import write_html_report

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
except Exception:  # pragma: no cover
    skimage_ssim = None
    skimage_psnr = None


def _parse_iso(exif: Dict) -> Optional[float]:
    raw = exif.get("EXIF ISOSpeedRatings") or exif.get("EXIF PhotographicSensitivity")
    if not raw:
        return None
    try:
        return float(str(raw).split()[0])
    except Exception:
        return None


def _parse_shutter_seconds(exif: Dict) -> Optional[float]:
    raw = exif.get("EXIF ExposureTime")
    if not raw:
        return None
    s = str(raw).strip()
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return None


def _adjust_thresholds_for_exposure(cfg: Dict, exif: Dict) -> Dict[str, float]:
    """
    Tighten or relax thresholds based on shooting conditions:
    - Low ISO + fast shutter -> demand more (tighten).
    - High ISO + slow shutter -> demand less (relax).
    """
    iso = _parse_iso(exif) or 0.0
    shutter = _parse_shutter_seconds(exif) or 0.0

    sharp_min = cfg.get("sharpness_min", 12.0)
    teneng_min = cfg.get("tenengrad_min", 30_000.0)
    motion_min = cfg.get("motion_ratio_min", 0.25)

    # Defaults
    sharp_factor = 1.0
    teneng_factor = 1.0
    motion_factor = 1.0

    low_iso, fast_shutter = 400.0, 1 / 500.0
    high_iso, slow_shutter = 3200.0, 1 / 50.0

    if iso and shutter:
        if iso <= low_iso and shutter <= fast_shutter:
            # Easy light: be stricter
            sharp_factor = 1.2
            teneng_factor = 1.15
            motion_factor = 1.1
        elif iso >= high_iso and shutter >= slow_shutter:
            # Tough light: be more forgiving
            sharp_factor = 0.8
            teneng_factor = 0.8
            motion_factor = 0.7

    return {
        "sharpness_min": sharp_min * sharp_factor,
        "tenengrad_min": teneng_min * teneng_factor,
        "motion_ratio_min": motion_min * motion_factor,
        "noise_std_max": cfg.get("noise_std_max", 12.0),
        "brightness_min": cfg.get("brightness_min", 0.08),
        "brightness_max": cfg.get("brightness_max", 0.92),
    }


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _suggest_keep(
    sharpness: float,
    sharpness_center: float,
    teneng: float,
    motion_ratio: float,
    noise: float,
    brightness_mean: float,
    shadows: float,
    highlights: float,
    composition: float,
    duplicate: bool,
    cfg: Dict,
    exif: Dict,
) -> Tuple[bool, List[str], float]:
    reasons: List[str] = []
    thresholds = _adjust_thresholds_for_exposure(cfg, exif)

    sharp_score = _clamp01(sharpness / thresholds["sharpness_min"])
    center_min = cfg.get(
        "center_sharpness_min", thresholds["sharpness_min"] * 1.2
    )
    sharp_center_score = _clamp01(sharpness_center / max(center_min, 1e-6))
    teneng_score = _clamp01(teneng / thresholds["tenengrad_min"])
    motion_score = _clamp01(motion_ratio / thresholds["motion_ratio_min"])
    noise_score = _clamp01(thresholds["noise_std_max"] / (thresholds["noise_std_max"] + max(noise, 1e-6)))
    brightness_mid = (thresholds["brightness_min"] + thresholds["brightness_max"]) / 2
    brightness_half = (thresholds["brightness_max"] - thresholds["brightness_min"]) / 2
    brightness_score = _clamp01(1 - abs(brightness_mean - brightness_mid) / max(brightness_half, 1e-6))
    shadows_min = cfg.get("shadows_min", 0.0)
    shadows_max = cfg.get("shadows_max", 0.5)
    highlights_min = cfg.get("highlights_min", 0.0)
    highlights_max = cfg.get("highlights_max", 0.1)

    def range_score(val: float, lo: float, hi: float) -> float:
        if val < lo:
            return _clamp01(1 - (lo - val) / max(lo, 1e-6))
        if val > hi:
            return _clamp01(1 - (val - hi) / max(1 - hi, 1e-6))
        return 1.0

    shadows_score = range_score(shadows, shadows_min, shadows_max)
    highlights_score = range_score(highlights, highlights_min, highlights_max)
    comp_score = _clamp01(composition if composition is not None else 0.0)

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

    weighted_sum = (
        sharp_score * weights["sharp"]
        + sharp_center_score * weights["sharp_center"]
        + teneng_score * weights["teneng"]
        + motion_score * weights["motion"]
        + noise_score * weights["noise"]
        + brightness_score * weights["brightness"]
        + shadows_score * weights["shadows"]
        + highlights_score * weights["highlights"]
        + comp_score * weights["composition"]
    )
    total_weights = sum(weights.values())
    quality_score = weighted_sum / total_weights if total_weights else 0.0

    cutoff = cfg.get("quality_score_min", 0.75)

    # Hard fails: configurable floor ratios to avoid keeping obviously bad frames.
    hard_cfg = {
        "sharp": cfg.get("hard_fail_sharp_ratio", 0.55),
        "sharp_center": cfg.get("hard_fail_sharp_center_ratio", 0.55),
        "teneng": cfg.get("hard_fail_teneng_ratio", 0.55),
        "motion": cfg.get("hard_fail_motion_ratio", 0.55),
        "brightness": cfg.get("hard_fail_brightness_ratio", 0.5),
        "noise": cfg.get("hard_fail_noise_ratio", 0.45),
        "shadows": cfg.get("hard_fail_shadows_ratio", 0.5),
        "highlights": cfg.get("hard_fail_highlights_ratio", 0.5),
        "composition": cfg.get("hard_fail_composition_ratio", 0.4),
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

    hard_fail = any(scores[k] < hard_cfg[k] for k in scores)

    keep = (quality_score >= cutoff) and not hard_fail

    if not keep:
        if quality_score < cutoff:
            reasons.append(f"quality score {quality_score:.2f} below {cutoff:.2f}")

        def add_reason(cond: bool, text: str) -> None:
            if cond:
                reasons.append(text)

        add_reason(
            scores["sharp"] < hard_cfg["sharp"]
            or sharpness < thresholds["sharpness_min"],
            f"sharpness {sharpness:.1f} < min {thresholds['sharpness_min']:.1f}",
        )
        add_reason(
            scores["sharp_center"] < hard_cfg["sharp_center"]
            or sharpness_center < center_min,
            f"center sharpness {sharpness_center:.1f} < min {center_min:.1f}",
        )
        add_reason(
            scores["teneng"] < hard_cfg["teneng"]
            or teneng < thresholds["tenengrad_min"],
            f"contrast {teneng:.0f} < min {thresholds['tenengrad_min']:.0f}",
        )
        add_reason(
            scores["motion"] < hard_cfg["motion"]
            or motion_ratio < thresholds["motion_ratio_min"],
            f"motion ratio {motion_ratio:.2f} < min {thresholds['motion_ratio_min']:.2f}",
        )
        add_reason(
            scores["noise"] < hard_cfg["noise"] or noise > thresholds["noise_std_max"],
            f"noise {noise:.1f} > max {thresholds['noise_std_max']:.1f}",
        )
        if scores["brightness"] < hard_cfg["brightness"]:
            if brightness_mean < thresholds["brightness_min"]:
                reasons.append(
                    f"brightness {brightness_mean:.2f} < min {thresholds['brightness_min']:.2f}"
                )
            elif brightness_mean > thresholds["brightness_max"]:
                reasons.append(
                    f"brightness {brightness_mean:.2f} > max {thresholds['brightness_max']:.2f}"
                )
            else:
                reasons.append("poor exposure")
        if scores["shadows"] < hard_cfg["shadows"]:
            if shadows > shadows_max:
                reasons.append(f"shadows {shadows:.2f} > max {shadows_max:.2f}")
            elif shadows < shadows_min:
                reasons.append(f"shadows {shadows:.2f} < min {shadows_min:.2f}")
        if scores["highlights"] < hard_cfg["highlights"]:
            if highlights > highlights_max:
                reasons.append(
                    f"highlights {highlights:.2f} > max {highlights_max:.2f}"
                )
            elif highlights < highlights_min:
                reasons.append(
                    f"highlights {highlights:.2f} < min {highlights_min:.2f}"
                )
        if scores["composition"] < hard_cfg["composition"]:
            reasons.append(
                f"composition {composition:.2f} < min {hard_cfg['composition']:.2f}"
            )
    if duplicate:
        reasons.append("duplicate")
        keep = False

    return keep, reasons, quality_score


def analyze_files(
    cfg: Dict,
    files: List[pathlib.Path],
    preview_dir: pathlib.Path,
    preview_cfg: Dict,
    progress_cb=None,
) -> List[Dict]:
    analysis_cfg = cfg.get("analysis", {})
    use_similarity = skimage_ssim is not None and skimage_psnr is not None
    results: List[Dict] = []

    def worker(path: pathlib.Path) -> Optional[Dict]:
        preview_path = preview_path_for(path, preview_dir, preview_cfg)
        if not preview_path.exists():
            return None
        exif = read_exif(path)
        dt = capture_date(
            exif, fallback=_dt.datetime.fromtimestamp(path.stat().st_mtime)
        )
        arr = open_preview_gray(preview_path)
        if arr is None:
            return None
        sharp = variance_of_laplacian(arr)
        teneng = tenengrad(arr)
        tensor = structure_tensor_ratio(arr)
        noise = noise_estimate(arr)
        bright = brightness_stats(arr)
        comp = composition_score(arr)
        face_info = None
        face_cfg = analysis_cfg.get("face", {})
        if face_cfg.get("enabled", False):
            rgb_preview = open_preview_rgb(preview_path, size=None)
            face_info = detect_faces(
                preview_path,
                arr,
                face_cfg,
                orientation=exif_orientation(exif),
                rgb_arr=rgb_preview,
            )
        ph = phash(preview_path, Image=Image)

        return {
            "path": str(path),
            "preview": str(preview_path),
            "capture_time": dt.isoformat(),
            "capture_ts": dt.timestamp(),
            "sharpness": sharp,
            "sharpness_center": variance_of_laplacian(
                arr[
                    int(arr.shape[0] * 0.25) : int(arr.shape[0] * 0.75),
                    int(arr.shape[1] * 0.25) : int(arr.shape[1] * 0.75),
                ]
            ),
            "tenengrad": teneng,
            "motion_ratio": tensor["ratio"],
            "noise": noise,
            "brightness": bright,
            "composition": comp,
            "phash": ph,
            "exif": exif,
            "faces": face_info,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg.get("concurrency", 4)
    ) as pool:
        futures = {pool.submit(worker, p): p for p in files}
        try:
            pending = set(futures.keys())
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    res = fut.result()
                    if res:
                        results.append(res)
                    if progress_cb:
                        progress_cb(1)
        except KeyboardInterrupt:
            for f in pending:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    results.sort(key=lambda r: r["path"])

    # Duplicate detection
    dup_threshold = int(analysis_cfg.get("duplicate_hamming", 6))
    window_sec = float(analysis_cfg.get("duplicate_window_seconds", 8))
    entries: List[Tuple[int, int, float]] = []
    for idx, r in enumerate(results):
        if r["phash"] is not None and r.get("capture_ts") is not None:
            entries.append((idx, int(r["phash"]), float(r["capture_ts"])))
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
        j = i - 1
        while j >= 0:
            idx_j, h_j, ts_j = entries[j]
            if ts_i - ts_j > window_sec:
                break
            if hamming(h_i, h_j) <= dup_threshold:
                union(idx_i, idx_j)
            j -= 1

    groups: Dict[int, List[int]] = {}
    for idx, _, _ in entries:
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    duplicate_indexes = set()
    rgb_cache: Dict[int, np.ndarray] = {}

    for members in groups.values():
        if len(members) < 2:
            continue
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
                reason_parts: List[str] = []
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
                    if keeper not in rgb_cache:
                        arr_keep = open_preview_rgb(pathlib.Path(k["preview"]))
                        if arr_keep is None:
                            arr_keep = np.zeros((1, 1, 3), dtype=np.float32)
                        rgb_cache[keeper] = arr_keep
                    if m not in rgb_cache:
                        arr_dup = open_preview_rgb(pathlib.Path(candidate["preview"]))
                        if arr_dup is None:
                            arr_dup = np.zeros((1, 1, 3), dtype=np.float32)
                        rgb_cache[m] = arr_dup
                    a = rgb_cache[keeper]
                    b = rgb_cache[m]
                    try:
                        ssim_val = float(
                            skimage_ssim(a, b, channel_axis=2, data_range=255)
                        )
                        psnr_val = float(skimage_psnr(a, b, data_range=255))
                        reason_parts.append(f"ssim={ssim_val:.3f}, psnr={psnr_val:.1f}")
                    except Exception:
                        pass

                results[m]["duplicate_reason"] = (
                    "; ".join(reason_parts) if reason_parts else "lower ranked in set"
                )

    for idx, r in enumerate(results):
        dup = idx in duplicate_indexes
        keep, reasons, quality_score = _suggest_keep(
            r["sharpness"],
            r.get("sharpness_center", r["sharpness"]),
            r.get("tenengrad", 0.0),
            r.get("motion_ratio", 1.0),
            r.get("noise", 0.0),
            r["brightness"]["mean"],
            r["brightness"].get("shadows", 0.0),
            r["brightness"].get("highlights", 0.0),
            r.get("composition", 0.0),
            dup,
            analysis_cfg,
            r.get("exif", {}),
        )
        r["quality_score"] = quality_score
        if dup and r.get("duplicate_reason"):
            reasons = [
                "duplicate: " + r["duplicate_reason"] if x == "duplicate" else x
                for x in reasons
            ]
        r["suggest_keep"] = keep
        r["decision"] = "keep" if keep else "discard"
        r["reasons"] = reasons

    groups_by_root: Dict[int, List[int]] = {}
    for idx, r in enumerate(results):
        if "duplicate_group" in r:
            groups_by_root.setdefault(int(r["duplicate_group"]), []).append(idx)
    for _, members in groups_by_root.items():
        keeps = [i for i in members if results[i]["decision"] == "keep"]
        if keeps:
            continue
        keeper = max(members, key=lambda i: results[i]["sharpness"])
        results[keeper]["decision"] = "keep"
        results[keeper]["reasons"].append("kept to avoid discarding all duplicates")

    return results


def write_outputs(results: List[Dict], analysis_cfg: Dict) -> None:
    output_json = pathlib.Path(analysis_cfg.get("results_path", "./analysis.json"))
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = pathlib.Path(analysis_cfg.get("report_path", "./report.html"))
    write_html_report(results, report_path, {"analysis": analysis_cfg})
