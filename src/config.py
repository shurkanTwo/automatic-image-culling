"""Configuration loading and defaults."""

import copy
import pathlib
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
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
            "backend": "mediapipe",
            "det_size": 640,
            "ctx_id": 0,
        },
        "report_path": "./report.html",
        "results_path": "./analysis.json",
    },
    "concurrency": 4,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into base and return the updated mapping."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clone_defaults() -> Dict[str, Any]:
    """Create a safe deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML if present; otherwise return defaults.

    Path resolution prefers the provided path and falls back to ``config.yaml``.
    """
    cfg = _clone_defaults()

    cfg_path = pathlib.Path(path) if path else pathlib.Path("config.yaml")
    if not cfg_path.exists() or yaml is None:
        return cfg

    with cfg_path.open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle) or {}
        return _deep_update(cfg, loaded)
