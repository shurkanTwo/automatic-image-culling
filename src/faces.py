"""Face detection helpers for preview images."""

import pathlib
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import variance_of_laplacian
from .preview import open_preview_rgb

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None

_FACE_APP: Optional[Any] = None
_THREAD_LOCAL = threading.local()


def _get_face_detector(face_cfg: Dict[str, Any]) -> Any:
    """Return a face detector instance based on configuration."""
    backend = (face_cfg.get("backend") or "mediapipe").lower()
    if backend == "mediapipe":
        return _get_mp_face()
    return _get_insightface(face_cfg)


def _get_insightface(face_cfg: Dict[str, Any]) -> Any:
    """Construct or reuse a global InsightFace detector."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    if FaceAnalysis is None:
        return None

    providers_cfg = face_cfg.get("providers")
    if isinstance(providers_cfg, str):
        providers = [providers_cfg]
    elif isinstance(providers_cfg, list) and providers_cfg:
        providers = providers_cfg
    else:
        providers = ["CPUExecutionProvider"]

    allowed_modules = face_cfg.get("allowed_modules", ["detection", "recognition"])

    def prepare_detector(provider_list: List[str]) -> Optional[Any]:
        try:
            app = FaceAnalysis(
                name="buffalo_l", providers=provider_list, allowed_modules=allowed_modules
            )
            app.prepare(
                ctx_id=face_cfg.get("ctx_id", 0),
                det_size=(face_cfg.get("det_size", 640), face_cfg.get("det_size", 640)),
            )
            return app
        except Exception:
            return None

    detector = prepare_detector(providers) or prepare_detector(["CPUExecutionProvider"])
    _FACE_APP = detector
    return detector


def _get_mp_face() -> Any:
    """Return a thread-local Mediapipe detector."""
    detector = getattr(_THREAD_LOCAL, "mp_face", None)
    if detector is not None:
        return detector
    if mp is None:
        return None
    try:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3
        )
        _THREAD_LOCAL.mp_face = detector
        return detector
    except Exception:
        return None


def _rotate(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate the array based on EXIF orientation code."""
    if orientation == 3:
        return np.rot90(arr, 2)
    if orientation == 6:
        return np.rot90(arr, -1)
    if orientation == 8:
        return np.rot90(arr, 1)
    return arr


def detect_faces(
    preview_path: pathlib.Path,
    gray_arr: np.ndarray,
    face_cfg: Dict[str, Any],
    orientation: int = 1,
    rgb_arr: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect faces in a preview image and return summary statistics.

    Returns a dictionary with counts, best scores, and per-face data, or None when
    detection is unavailable or no faces are present.
    """
    detector = _get_face_detector(face_cfg)
    if detector is None:
        return None

    rgb_full = rgb_arr if rgb_arr is not None else open_preview_rgb(preview_path, size=None)
    if rgb_full is None:
        return None
    rgb_full = _rotate(rgb_full, orientation)
    if rgb_full.dtype != np.uint8:
        rgb_full = np.clip(rgb_full, 0, 255).astype(np.uint8)
    gray_arr = _rotate(gray_arr, orientation)

    faces: List[Dict[str, Any]] = []
    if mp and isinstance(detector, mp.solutions.face_detection.FaceDetection):
        results_mp = detector.process(rgb_full.astype(np.uint8))
        if results_mp.detections:
            height, width, _ = rgb_full.shape
            for det in results_mp.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                x2 = int((bbox.xmin + bbox.width) * width)
                y2 = int((bbox.ymin + bbox.height) * height)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_gray = gray_arr[y1:y2, x1:x2]
                face_sharp = variance_of_laplacian(face_gray) if face_gray.size else 0.0
                faces.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "score": float(det.score[0]) if det.score else 0.0,
                        "sharpness": face_sharp,
                    }
                )

    elif FaceAnalysis and hasattr(detector, "get"):
        bgr = rgb_full[:, :, ::-1].astype(np.uint8)
        detections = detector.get(bgr)
        for face in detections:
            box = [float(value) for value in face.bbox.tolist()]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_gray = gray_arr[y1:y2, x1:x2]
            face_sharp = variance_of_laplacian(face_gray) if face_gray.size else 0.0
            face_entry: Dict[str, Any] = {
                "bbox": box,
                "score": float(face.det_score),
                "sharpness": face_sharp,
            }
            if hasattr(face, "embedding") and face.embedding is not None:
                try:
                    face_entry["embedding"] = face.embedding.tolist()
                except Exception:
                    pass
            elif hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                try:
                    face_entry["embedding"] = face.normed_embedding.tolist()
                except Exception:
                    pass
            faces.append(face_entry)

    if not faces:
        return None

    best = max(faces, key=lambda data: data["sharpness"])
    return {
        "count": len(faces),
        "best_sharpness": best["sharpness"],
        "best_score": best["score"],
        "faces": faces,
    }
