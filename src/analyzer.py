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


def _suggest_keep(
    sharpness: float,
    teneng: float,
    motion_ratio: float,
    noise: float,
    brightness_mean: float,
    duplicate: bool,
    cfg: Dict,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    keep = True
    if sharpness < cfg.get("sharpness_min", 12.0):
        keep = False
        reasons.append("soft")
    if teneng < cfg.get("tenengrad_min", 30_000.0):
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
        dt = capture_date(exif, fallback=_dt.datetime.fromtimestamp(path.stat().st_mtime))
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
            face_info = detect_faces(preview_path, arr, face_cfg, orientation=exif_orientation(exif))
        ph = phash(preview_path, Image=Image)

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
            "phash": ph,
            "exif": exif,
            "faces": face_info,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.get("concurrency", 4)) as pool:
        futures = {pool.submit(worker, p): p for p in files}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
            if progress_cb:
                progress_cb(1)

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
        members_sorted = sorted(members, key=lambda i: results[i]["sharpness"], reverse=True)
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
                    reason_parts.append(f"sharpness {candidate['sharpness']:.1f} < {k['sharpness']:.1f}")
                if candidate.get("tenengrad") is not None and k.get("tenengrad") is not None:
                    if candidate["tenengrad"] < k["tenengrad"]:
                        reason_parts.append(f"contrast {candidate['tenengrad']:.0f} < {k['tenengrad']:.0f}")
                if candidate.get("motion_ratio") is not None and k.get("motion_ratio") is not None:
                    if candidate["motion_ratio"] < k["motion_ratio"]:
                        reason_parts.append(f"motion ratio {candidate['motion_ratio']:.2f} < {k['motion_ratio']:.2f}")
                if candidate.get("noise") is not None and k.get("noise") is not None:
                    if candidate["noise"] > k["noise"]:
                        reason_parts.append(f"noise {candidate['noise']:.1f} > {k['noise']:.1f}")

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
                        ssim_val = float(skimage_ssim(a, b, channel_axis=2, data_range=255))
                        psnr_val = float(skimage_psnr(a, b, data_range=255))
                        reason_parts.append(f"ssim={ssim_val:.3f}, psnr={psnr_val:.1f}")
                    except Exception:
                        pass

                results[m]["duplicate_reason"] = "; ".join(reason_parts) if reason_parts else "lower ranked in set"

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
            reasons = ["duplicate: " + r["duplicate_reason"] if x == "duplicate" else x for x in reasons]
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
    write_html_report(results, report_path)
