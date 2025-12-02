import pathlib
from typing import Dict, List, Optional

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


_FACE_APP = None
_MP_FACE = None


def _get_face_detector(face_cfg: Dict):
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
    except Exception:
        return None


def _get_mp_face():
    global _MP_FACE
    if _MP_FACE is not None:
        return _MP_FACE
    if mp is None:
        return None
    try:
        _MP_FACE = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3
        )
        return _MP_FACE
    except Exception:
        return None


def _rotate(arr: np.ndarray, orientation: int) -> np.ndarray:
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
    face_cfg: Dict,
    orientation: int = 1,
) -> Optional[Dict]:
    detector = _get_face_detector(face_cfg)
    if detector is None:
        return None

    faces: List[Dict] = []
    rgb_full = open_preview_rgb(preview_path, size=None)
    if rgb_full is None:
        return None
    rgb_full = _rotate(rgb_full, orientation)
    gray_arr = _rotate(gray_arr, orientation)

    if mp and isinstance(detector, mp.solutions.face_detection.FaceDetection):
        results_mp = detector.process(rgb_full.astype(np.uint8))
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
        dets = detector.get(bgr)
        for f in dets:
            box = [float(x) for x in f.bbox.tolist()]
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(bgr.shape[1], x2)
            y2 = min(bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_gray = gray_arr[y1:y2, x1:x2]
            face_sharp = variance_of_laplacian(face_gray) if face_gray.size else 0.0
            faces.append(
                {"bbox": box, "score": float(f.det_score), "sharpness": face_sharp}
            )

    if not faces:
        return None
    best = max(faces, key=lambda d: d["sharpness"])
    return {
        "count": len(faces),
        "best_sharpness": best["sharpness"],
        "best_score": best["score"],
        "faces": faces,
    }
