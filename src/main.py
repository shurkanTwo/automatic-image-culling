import argparse
import concurrent.futures
import pathlib
import shutil
from typing import Iterable, List, Optional

import numpy as np

from .analyzer import analyze_files, write_outputs
from .config import load_config
from .discovery import capture_date, find_arw_files, plan_destination, read_exif
from .preview import ensure_preview, generate_preview

try:
    import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


def _progress_bar(total: int, desc: str):
    if tqdm:
        return tqdm.tqdm(total=total, desc=desc, leave=False)

    class _Dummy:
        def update(self, _n: int = 1) -> None:
            return

        def close(self) -> None:
            return

    return _Dummy()


def _exclude_list(cfg: dict) -> List[str]:
    exclude = cfg.get("exclude_dirs", [])
    exclude = list(set(exclude + [cfg.get("output_dir", "./output"), cfg.get("preview_dir", "./previews")]))
    return exclude


def scan_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    print(f"Found {len(files)} .ARW files in {cfg['input_dir']}")
    bar = _progress_bar(len(files), "Scan")
    if args.json:
        data = []
        for p in files:
            exif = read_exif(p)
            data.append({"path": str(p), "exif": exif})
            bar.update(1)
        import json

        print(json.dumps(data, indent=2))
    else:
        for p in files:
            exif = read_exif(p)
            dt = capture_date(exif, fallback=None)
            print(f"{p} | {dt}")
            bar.update(1)
    bar.close()


def previews_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    preview_cfg = cfg.get("preview", {})
    preview_cfg.setdefault("long_edge", 2048)
    preview_cfg.setdefault("format", "webp")
    preview_cfg.setdefault("quality", 85)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    if not files:
        print("No .ARW files found")
        return
    preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
    jobs = []
    bar = _progress_bar(len(files), "Previews")
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.get("concurrency", 4)) as pool:
        for p in files:
            jobs.append(pool.submit(generate_preview, p, preview_dir, preview_cfg))
        for job in concurrent.futures.as_completed(jobs):
            result = job.result()
            if result:
                print(f"Preview written: {result}")
            bar.update(1)
    bar.close()


def sort_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sort_cfg = cfg.get("sort", {})
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
    actions: List[str] = []
    bar = _progress_bar(len(files), "Sort")
    for p in files:
        exif = read_exif(p)
        dest = plan_destination(p, exif, sort_cfg, output_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            actions.append(f"PLAN {'COPY' if sort_cfg.get('copy', True) else 'MOVE'} {p} -> {dest}")
            bar.update(1)
            continue
        if sort_cfg.get("copy", True):
            shutil.copy2(p, dest)
            actions.append(f"COPIED {p} -> {dest}")
        else:
            shutil.move(p, dest)
            actions.append(f"MOVED {p} -> {dest}")
        bar.update(1)
    for line in actions:
        print(line)
    if args.dry_run:
        print("Dry run complete; use --apply to perform moves/copies.")
    bar.close()


def analyze_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    preview_cfg = cfg.get("preview", {})
    preview_cfg.setdefault("long_edge", 2048)
    preview_cfg.setdefault("format", "webp")
    preview_cfg.setdefault("quality", 85)
    files = find_arw_files(cfg["input_dir"], exclude_dirs=_exclude_list(cfg))
    if not files:
        print("No .ARW files found")
        return

    preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
    # Ensure previews exist; skip generating in parallel here to keep analyze simple
    for p in files:
        ensure_preview(p, preview_dir, preview_cfg)

    bar = _progress_bar(len(files), "Analyze")
    results = analyze_files(cfg, files, preview_dir, preview_cfg, progress_cb=bar.update)
    bar.close()
    write_outputs(results, cfg.get("analysis", {}))
    print(f"Analysis written to {cfg['analysis']['results_path']}")
    print(f"HTML report written to {cfg['analysis']['report_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sony .ARW preprocessing and sorting")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="Path to YAML config file", default=None)

    scan = sub.add_parser("scan", help="List .ARW files and basic EXIF info")
    add_common(scan)
    scan.add_argument("--json", action="store_true", help="Output JSON metadata")
    scan.set_defaults(func=scan_command)

    prev = sub.add_parser("previews", help="Generate quick previews")
    add_common(prev)
    prev.set_defaults(func=previews_command)

    sort = sub.add_parser("sort", help="Copy/move files into structured folders")
    add_common(sort)
    sort.add_argument("--apply", action="store_true", help="Perform the operations (default is dry run)")
    sort.set_defaults(func=sort_command)

    analyze = sub.add_parser("analyze", help="Score images and emit JSON + HTML report")
    add_common(analyze)
    analyze.set_defaults(func=analyze_command)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "sort" and not args.apply:
        args.dry_run = True
    elif args.command == "sort":
        args.dry_run = False
    args.func(args)


if __name__ == "__main__":
    main()
