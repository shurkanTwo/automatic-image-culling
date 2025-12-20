"""CLI entrypoint for preprocessing and analyzing .ARW files."""

import argparse
import concurrent.futures
import datetime as _dt
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from .analyzer import analyze_files, write_outputs
from .config import AnalysisConfig, AppConfig, PreviewConfig, load_config
from .decisions import apply_decisions
from .discovery import (
    ExifData,
    capture_date,
    find_arw_files,
    read_exif,
)
from .preview import ensure_preview, generate_preview

try:
    import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


class _Progress:
    """Minimal progress interface compatible with tqdm update/close."""

    def __init__(self, total: int, desc: str, batch_size: int = 16):
        self._impl: Optional[Any] = None
        self._pending = 0
        self._batch_size = max(1, batch_size)
        if tqdm is not None:
            self._impl = tqdm.tqdm(total=total, desc=desc, leave=False)

    def update(self, amount: int = 1) -> None:
        self._pending += amount
        if self._impl is not None and self._pending >= self._batch_size:
            self._impl.update(self._pending)
            self._pending = 0

    def close(self) -> None:
        if self._impl is not None:
            if self._pending:
                self._impl.update(self._pending)
                self._pending = 0
            self._impl.close()


def _exclude_list(cfg: AppConfig) -> List[str]:
    """Return a de-duplicated, deterministic list of directories to ignore."""
    exclude_dirs = list(cfg.get("exclude_dirs", []))
    exclude_dirs.extend(
        [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]
    )
    return list(dict.fromkeys(exclude_dirs))


def _preview_config(cfg: AppConfig) -> PreviewConfig:
    """Return a defensive copy of the preview configuration."""
    preview_cfg = cast(PreviewConfig, dict(cfg.get("preview") or {}))
    preview_cfg.setdefault("long_edge", 2048)
    preview_cfg.setdefault("format", "webp")
    preview_cfg.setdefault("quality", 85)
    return preview_cfg


def _preview_dir(cfg: AppConfig) -> pathlib.Path:
    """Return the configured preview directory path."""
    return pathlib.Path(cfg.get("preview_dir", "./previews"))


def _prepare_analysis_config(
    cfg: AppConfig, analysis_dir: pathlib.Path
) -> AnalysisConfig:
    """Return analysis config with resolved output paths."""
    analysis_cfg = cast(AnalysisConfig, dict(cfg.get("analysis") or {}))

    results_path = pathlib.Path(analysis_cfg.get("results_path", "analysis.json"))
    if not results_path.is_absolute():
        results_path = analysis_dir / results_path
    analysis_cfg["results_path"] = str(results_path)

    report_path = pathlib.Path(analysis_cfg.get("report_path", "report.html"))
    if not report_path.is_absolute():
        report_path = analysis_dir / report_path
    analysis_cfg["report_path"] = str(report_path)

    return analysis_cfg


def scan_command(args: argparse.Namespace) -> None:
    """List .ARW files with basic metadata."""
    cfg = load_config(args.config)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    print(f"Found {len(files)} .ARW files in {cfg['input_dir']}")
    bar = _Progress(len(files), "Scan")

    def _exif_with_fallback(path: pathlib.Path) -> Tuple[ExifData, _dt.datetime]:
        exif = read_exif(path)
        fallback_dt = _dt.datetime.fromtimestamp(path.stat().st_mtime)
        return exif, capture_date(exif, fallback=fallback_dt)

    if args.json:
        import json
    data: List[Dict[str, Any]] = []

    for file_path in files:
        exif, dt = _exif_with_fallback(file_path)
        if args.json:
            data.append({"path": str(file_path), "exif": exif})
        else:
            print(f"{file_path} | {dt}")
        bar.update(1)
    bar.close()
    if args.json:
        print(json.dumps(data, indent=2))


def previews_command(args: argparse.Namespace) -> None:
    """Generate preview images for each discovered RAW file."""
    cfg = load_config(args.config)
    preview_cfg = _preview_config(cfg)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    if not files:
        print("No .ARW files found")
        return
    preview_dir = _preview_dir(cfg)
    bar = _Progress(len(files), "Previews")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg.get("concurrency", 4)
    ) as pool:
        jobs = [
            pool.submit(generate_preview, path, preview_dir, preview_cfg)
            for path in files
        ]
        for job in concurrent.futures.as_completed(jobs):
            result = job.result()
            if result is not None:
                print(f"Preview written: {result}")
            bar.update(1)
    bar.close()


def analyze_command(args: argparse.Namespace) -> None:
    """Run analysis pipeline and emit JSON and HTML outputs."""
    cfg = load_config(args.config)
    preview_cfg = _preview_config(cfg)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    if not files:
        print("No .ARW files found")
        return

    preview_dir = _preview_dir(cfg)
    for path in files:
        ensure_preview(path, preview_dir, preview_cfg)

    analysis_dir = pathlib.Path(cfg.get("output_dir", "./output")) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_cfg = _prepare_analysis_config(cfg, analysis_dir)
    cfg["analysis"] = analysis_cfg

    bar = _Progress(len(files), "Analyze")
    try:
        results = analyze_files(
            cfg, files, preview_dir, preview_cfg, progress_cb=bar.update
        )
    except KeyboardInterrupt:
        bar.close()
        print("Analysis cancelled by user.")
        return
    bar.close()
    write_outputs(results, analysis_cfg)
    print(f"Analysis written to {analysis_cfg['results_path']}")
    print(f"HTML report written to {analysis_cfg['report_path']}")


def decisions_command(args: argparse.Namespace) -> None:
    """Apply decisions.json and move/copy files into keep/discard folders."""
    cfg = load_config(args.config)
    decisions_path = pathlib.Path(args.decisions)
    if not decisions_path.exists():
        print(f"Decisions file not found: {decisions_path}")
        return
    bar: Optional[_Progress] = None

    def _progress(current: int, total: int) -> None:
        nonlocal bar
        if total <= 0:
            return
        if bar is None:
            bar = _Progress(total, "Decisions")
        bar.update(1)
        if current == total:
            bar.close()

    summary = apply_decisions(
        decisions_path,
        cfg,
        apply=args.apply,
        copy_files=args.copy,
        keep_subdir=args.keep_subdir,
        discard_subdir=args.discard_subdir,
        progress_cb=_progress,
        log_cb=print,
    )
    if not args.apply:
        print("Dry run complete; use --apply to perform moves/copies.")
    print(
        "Decisions summary: "
        f"total={summary.total} keep={summary.keep} discard={summary.discard} "
        f"moved={summary.moved} copied={summary.copied} "
        f"missing={summary.missing} skipped={summary.skipped}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Sony .ARW preprocessing and analysis")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="List .ARW files and basic EXIF info")
    scan.add_argument("--json", action="store_true", help="Output JSON metadata")
    scan.set_defaults(func=scan_command)

    prev = sub.add_parser("previews", help="Generate quick previews")
    prev.set_defaults(func=previews_command)

    analyze = sub.add_parser("analyze", help="Score images and emit JSON + HTML report")
    analyze.set_defaults(func=analyze_command)

    decisions = sub.add_parser(
        "decisions", help="Apply decisions.json to move/copy files"
    )
    decisions.add_argument(
        "--decisions", required=True, help="Path to decisions.json"
    )
    decisions.add_argument(
        "--apply",
        action="store_true",
        help="Perform moves/copies (default is dry run)",
    )
    decisions.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving"
    )
    decisions.add_argument(
        "--keep-subdir", default="keep", help="Subfolder for keepers"
    )
    decisions.add_argument(
        "--discard-subdir", default="discard", help="Subfolder for discards"
    )
    decisions.set_defaults(func=decisions_command)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Parse CLI arguments and dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    start = time.perf_counter()
    try:
        args.func(args)
    finally:
        elapsed = time.perf_counter() - start
        print(f"{args.command} completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
