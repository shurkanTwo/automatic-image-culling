"""Configuration loading and defaults."""

import copy
import pathlib
from typing import Any, Dict, List, Optional, TypedDict, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class PreviewConfig(TypedDict, total=False):
    """Preview generation options."""

    long_edge: int
    format: str
    quality: int


class FaceConfig(TypedDict, total=False):
    """Face detection backend configuration."""

    enabled: bool
    backend: str
    providers: Union[str, List[str]]
    allowed_modules: List[str]
    det_size: int
    ctx_id: int


class AnalysisConfig(TypedDict, total=False):
    """Image analysis scoring thresholds and output paths."""

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
    duplicate_hamming: int
    duplicate_window_seconds: int
    duplicate_bucket_bits: int
    quality_score_min: float
    hard_fail_sharp_ratio: float
    hard_fail_sharp_center_ratio: float
    hard_fail_teneng_ratio: float
    hard_fail_motion_ratio: float
    hard_fail_brightness_ratio: float
    hard_fail_noise_ratio: float
    hard_fail_shadows_ratio: float
    hard_fail_highlights_ratio: float
    hard_fail_composition_ratio: float
    face: FaceConfig
    report_path: str
    results_path: str


class AppConfig(TypedDict, total=False):
    """Top-level application configuration."""

    input_dir: str
    preview: PreviewConfig
    analysis: AnalysisConfig
    concurrency: int
    exclude_dirs: List[str]


DEFAULT_CONFIG: AppConfig = {
    "input_dir": "./input",
    "preview": {"long_edge": 2048, "format": "webp", "quality": 85},
    "exclude_dirs": ["analysis", "output", "previews"],
    "analysis": {
        "sharpness_min": 8.0,
        "brightness_min": 0.08,
        "brightness_max": 0.92,
        "shadows_min": 0.0,
        "shadows_max": 0.5,
        "highlights_min": 0.0,
        "highlights_max": 0.1,
        "duplicate_hamming": 6,
        "duplicate_window_seconds": 8,
        "duplicate_bucket_bits": 8,
        "tenengrad_min": 200.0,
        "motion_ratio_min": 0.02,
        "noise_std_max": 25.0,
        "quality_score_min": 0.75,
        "hard_fail_sharp_ratio": 0.55,
        "hard_fail_sharp_center_ratio": 0.55,
        "hard_fail_teneng_ratio": 0.55,
        "hard_fail_motion_ratio": 0.55,
        "hard_fail_brightness_ratio": 0.5,
        "hard_fail_noise_ratio": 0.45,
        "hard_fail_shadows_ratio": 0.5,
        "hard_fail_highlights_ratio": 0.5,
        "hard_fail_composition_ratio": 0.4,
        "face": {
            "enabled": False,
            "backend": "mediapipe",
            "allowed_modules": ["detection", "recognition"],
            "det_size": 640,
            "ctx_id": 0,
        },
        "report_path": "./report.html",
        "results_path": "./analysis.json",
    },
    "concurrency": 4,
}


def _deep_update(base: AppConfig, override: Dict[str, Any]) -> AppConfig:
    """Recursively merge override values into base and return the updated mapping."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clone_defaults() -> AppConfig:
    """Create a safe deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def default_config() -> AppConfig:
    """Return a deep copy of the default configuration."""
    return _clone_defaults()


def load_config(path: Optional[str]) -> AppConfig:
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


def save_config(path: str, cfg: AppConfig) -> None:
    """Persist configuration to a YAML file."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to save configuration.")
    cfg_path = pathlib.Path(path)
    with cfg_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(cfg, file_handle, sort_keys=False)
