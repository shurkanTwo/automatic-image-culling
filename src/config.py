import pathlib
from typing import Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict = {
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


def _deep_update(base: Dict, override: Dict) -> Dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str]) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg["analysis"] = dict(DEFAULT_CONFIG["analysis"])  # shallow clone nested
    cfg["preview"] = dict(DEFAULT_CONFIG["preview"])
    cfg["sort"] = dict(DEFAULT_CONFIG["sort"])
    cfg["analysis"]["face"] = dict(DEFAULT_CONFIG["analysis"]["face"])

    cfg_path = pathlib.Path(path) if path else pathlib.Path("config.yaml")
    if cfg_path.exists() and yaml:
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            cfg = _deep_update(cfg, loaded)
    return cfg
