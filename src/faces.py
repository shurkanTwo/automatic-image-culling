"""Face detection helpers for preview images."""

import pathlib
import threading
from typing import Any, List, Optional, TypedDict

import numpy as np

from .config import FaceConfig
from .metrics import variance_of_laplacian
from .preview import open_preview_rgb
from .orientation import rotate_array

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


class FaceDetection(TypedDict, total=False):
    """Single face detection data."""

    bbox: List[int]
    score: float
    sharpness: float
    embedding: List[float]


class FaceSummary(TypedDict):
    """Aggregate results for a frame."""

    count: int
    best_sharpness: float
    best_score: float
    faces: List[FaceDetection]


def _get_face_detector(face_cfg: FaceConfig) -> Optional[Any]:
    """Return a face detector instance based on configuration."""
    backend = (face_cfg.get("backend") or "mediapipe").lower()
    if backend == "mediapipe":
        return _get_mp_face()
    return _get_insightface(face_cfg)


def _get_insightface(face_cfg: FaceConfig) -> Optional[Any]:
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
                name="buffalo_l",
                providers=provider_list,
                allowed_modules=allowed_modules,
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


def _get_mp_face() -> Optional[Any]:
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


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Clamp array values and cast to uint8."""
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _detect_with_mediapipe(
    detector: Any, rgb_full: np.ndarray, gray_arr: np.ndarray
) -> List[FaceDetection]:
    """Run Mediapipe detection and return face data."""
    detections: List[FaceDetection] = []
    results_mp = detector.process(rgb_full.astype(np.uint8))
    if not results_mp.detections:
        return detections
    height, width, __ = rgb_full.shape
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
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": float(det.score[0]) if det.score else 0.0,
                "sharpness": face_sharp,
            }
        )
    return detections


def _detect_with_insightface(
    detector: Any, rgb_full: np.ndarray, gray_arr: np.ndarray
) -> List[FaceDetection]:
    """Run InsightFace detection and return face data."""
    detections: List[FaceDetection] = []
    bgr = _ensure_uint8(rgb_full)[:, :, ::-1]
    detector_results = detector.get(bgr)
    if detector_results is None:
        return detections
    for face in detector_results:
        box = [float(value) for value in face.bbox.tolist()]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        face_gray = gray_arr[y1:y2, x1:x2]
        face_sharp = variance_of_laplacian(face_gray) if face_gray.size else 0.0
        face_entry: FaceDetection = {
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
        detections.append(face_entry)
    return detections


def detect_faces(
    preview_path: pathlib.Path,
    gray_arr: np.ndarray,
    face_cfg: FaceConfig,
    orientation: int = 1,
    rgb_arr: Optional[np.ndarray] = None,
) -> Optional[FaceSummary]:
    """
    Detect faces in a preview image and return summary statistics.

    Returns a dictionary with counts, best scores, and per-face data, or None when
    detection is unavailable or no faces are present.
    """
    detector = _get_face_detector(face_cfg)
    if detector is None:
        return None

    rgb_full = (
        rgb_arr if rgb_arr is not None else open_preview_rgb(preview_path, size=None)
    )
    if rgb_full is None:
        return None

    oriented_rgb = _ensure_uint8(rotate_array(rgb_full, orientation))
    oriented_gray = rotate_array(gray_arr, orientation)

    faces: List[FaceDetection]
    if mp is not None and isinstance(
        detector, mp.solutions.face_detection.FaceDetection
    ):
        faces = _detect_with_mediapipe(detector, oriented_rgb, oriented_gray)
    elif FaceAnalysis is not None and hasattr(detector, "get"):
        faces = _detect_with_insightface(detector, oriented_rgb, oriented_gray)
    else:
        faces = []

    if not faces:
        return None

    best = max(faces, key=lambda data: data["sharpness"])
    return {
        "count": len(faces),
        "best_sharpness": best["sharpness"],
        "best_score": best["score"],
        "faces": faces,
    }
