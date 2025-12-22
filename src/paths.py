"""Path constants and helpers derived from the input directory."""

from __future__ import annotations

import pathlib
from typing import Any, Mapping, MutableMapping

OUTPUT_SUBDIR = "output"
PREVIEW_SUBDIR = "previews"
ANALYSIS_SUBDIR = "analysis"
DECISIONS_FILENAME = "decisions.json"


def input_dir_from_cfg(cfg: Mapping[str, Any]) -> pathlib.Path:
    """Return the input directory from config."""
    return pathlib.Path(cfg.get("input_dir", "./input")).expanduser()


def output_dir_for_input(input_dir: pathlib.Path) -> pathlib.Path:
    """Return the output directory under the input directory."""
    return input_dir / OUTPUT_SUBDIR


def preview_dir_for_input(input_dir: pathlib.Path) -> pathlib.Path:
    """Return the previews directory under the input directory."""
    return input_dir / PREVIEW_SUBDIR


def analysis_dir_for_input(input_dir: pathlib.Path) -> pathlib.Path:
    """Return the analysis directory under the input directory."""
    return input_dir / ANALYSIS_SUBDIR


def decisions_path_for_input(input_dir: pathlib.Path) -> pathlib.Path:
    """Return the decisions.json path under the analysis directory."""
    return analysis_dir_for_input(input_dir) / DECISIONS_FILENAME


def drop_path_config(cfg: MutableMapping[str, Any]) -> None:
    """Remove legacy path keys from config in-place."""
    cfg.pop("output_dir", None)
    cfg.pop("preview_dir", None)
