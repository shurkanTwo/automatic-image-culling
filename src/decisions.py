"""Apply decisions.json outputs to move/copy source images."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, TypedDict, cast

from .config import AppConfig
from .paths import input_dir_from_cfg, output_dir_for_input


class DecisionEntry(TypedDict, total=False):
    """Minimal decision payload for a single file."""

    path: str
    decision: str


@dataclass(frozen=True)
class DecisionsSummary:
    """Aggregate counts for applied decisions."""

    total: int
    keep: int
    discard: int
    moved: int
    copied: int
    skipped: int
    missing: int


ProgressCallback = Optional[Callable[[int, int], None]]
LogCallback = Optional[Callable[[str], None]]


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _normalize_decision_path(path_value: str) -> pathlib.Path:
    """Normalize a decision path string to a local filesystem Path."""
    raw = str(path_value).strip()
    return pathlib.Path(raw).expanduser()


def _load_decisions(path: pathlib.Path) -> List[DecisionEntry]:
    """Load decisions from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if not isinstance(payload, list):
        raise ValueError("decisions.json must contain a list of results")
    return cast(List[DecisionEntry], payload)


def _destination_for(
    source: pathlib.Path,
    input_dir: Optional[pathlib.Path],
    dest_root: pathlib.Path,
) -> pathlib.Path:
    """Plan destination, preserving relative structure when possible."""
    if input_dir is not None:
        try:
            rel = source.resolve().relative_to(input_dir.resolve())
            return dest_root / rel
        except Exception:
            pass
    return dest_root / source.name


def _dedupe_destination(dest: pathlib.Path) -> pathlib.Path:
    """Avoid overwriting existing files by adding a numeric suffix."""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    for idx in range(1, 10_000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find a free destination for {dest}")


def apply_decisions(
    decisions_path: pathlib.Path,
    cfg: AppConfig,
    *,
    apply: bool = False,
    copy_files: bool = False,
    keep_subdir: str = "keep",
    discard_subdir: str = "discard",
    progress_cb: ProgressCallback = None,
    log_cb: LogCallback = None,
) -> DecisionsSummary:
    """
    Apply decisions.json to move/copy files into keep/discard subfolders.
    """
    entries = _load_decisions(decisions_path)
    total = len(entries)
    input_dir = input_dir_from_cfg(cfg)
    output_dir = output_dir_for_input(input_dir)
    keep_root = output_dir / keep_subdir
    discard_root = output_dir / discard_subdir
    if apply:
        keep_root.mkdir(parents=True, exist_ok=True)
        discard_root.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": total,
        "keep": 0,
        "discard": 0,
        "moved": 0,
        "copied": 0,
        "skipped": 0,
        "missing": 0,
    }
    log_limit = 25
    logged = 0

    for idx, entry in enumerate(entries, 1):
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"keep", "discard"}:
            counts["skipped"] += 1
            if progress_cb is not None:
                progress_cb(idx, total)
            continue
        raw_path = entry.get("path")
        if not raw_path:
            counts["skipped"] += 1
            if progress_cb is not None:
                progress_cb(idx, total)
            continue
        source = _normalize_decision_path(raw_path)
        if not source.exists():
            counts["missing"] += 1
            if log_cb is not None and logged < log_limit:
                log_cb(f"Missing: {source}")
                logged += 1
            if progress_cb is not None:
                progress_cb(idx, total)
            continue

        dest_root = keep_root if decision == "keep" else discard_root
        dest = _destination_for(source, input_dir, dest_root)
        try:
            if dest.resolve() == source.resolve():
                counts["skipped"] += 1
                if progress_cb is not None:
                    progress_cb(idx, total)
                continue
        except Exception:
            pass
        dest = _dedupe_destination(dest)

        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if copy_files:
                shutil.copy2(source, dest)
                counts["copied"] += 1
                action = "COPIED"
            else:
                shutil.move(source, dest)
                counts["moved"] += 1
                action = "MOVED"
            if log_cb is not None and logged < log_limit:
                log_cb(f"{action} {source} -> {dest}")
                logged += 1
        else:
            if log_cb is not None and logged < log_limit:
                verb = "COPY" if copy_files else "MOVE"
                log_cb(f"PLAN {verb} {source} -> {dest}")
                logged += 1

        if decision == "keep":
            counts["keep"] += 1
        else:
            counts["discard"] += 1

        if progress_cb is not None:
            progress_cb(idx, total)

    return DecisionsSummary(**counts)
