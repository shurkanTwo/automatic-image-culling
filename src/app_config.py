"""Shared helpers for derived application configuration."""

from __future__ import annotations

import pathlib
from typing import List, cast

from .config import DEFAULT_CONFIG, AnalysisConfig, AppConfig, PreviewConfig
from .paths import (
    analysis_dir_for_input,
    input_dir_from_cfg,
    output_dir_for_input,
    preview_dir_for_input,
)


def exclude_list(cfg: AppConfig) -> List[str]:
    """Return a de-duplicated, deterministic list of directories to ignore."""
    exclude_dirs = list(cfg.get("exclude_dirs", DEFAULT_CONFIG["exclude_dirs"]))
    input_dir = input_dir_from_cfg(cfg)
    exclude_dirs.extend(
        [
            str(analysis_dir_for_input(input_dir)),
            str(output_dir_for_input(input_dir)),
            str(preview_dir_for_input(input_dir)),
        ]
    )
    return list(dict.fromkeys(exclude_dirs))


def preview_config(cfg: AppConfig) -> PreviewConfig:
    """Return a defensive copy of the preview configuration."""
    preview_cfg = dict(DEFAULT_CONFIG["preview"])
    preview_cfg.update(cfg.get("preview") or {})
    return cast(PreviewConfig, preview_cfg)


def prepare_analysis_config(
    cfg: AppConfig, analysis_dir: pathlib.Path
) -> AnalysisConfig:
    """Return analysis config with resolved output paths."""
    analysis_cfg = cast(AnalysisConfig, dict(cfg.get("analysis") or {}))

    results_path = pathlib.Path(
        analysis_cfg.get("results_path", DEFAULT_CONFIG["analysis"]["results_path"])
    )
    if not results_path.is_absolute():
        results_path = analysis_dir / results_path
    analysis_cfg["results_path"] = str(results_path)

    report_path = pathlib.Path(
        analysis_cfg.get("report_path", DEFAULT_CONFIG["analysis"]["report_path"])
    )
    if not report_path.is_absolute():
        report_path = analysis_dir / report_path
    analysis_cfg["report_path"] = str(report_path)

    return analysis_cfg
