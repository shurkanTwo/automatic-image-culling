"""Minimal Tkinter GUI wrapper for the analysis pipeline."""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
import queue
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple, cast

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    _TK_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TK_IMPORT_ERROR = None

from .analyzer import analyze_files, write_outputs
from .config import AnalysisConfig, AppConfig, PreviewConfig, load_config
from .decisions import apply_decisions
from .discovery import capture_date, find_arw_files, read_exif
from .preview import generate_preview

QueueItem = Tuple[str, object]


def _exclude_list(cfg: AppConfig) -> list[str]:
    """Return a de-duplicated list of directories to ignore."""
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


def _clean_path(value: str) -> str:
    """Normalize user-provided path strings."""
    return str(pathlib.Path(value).expanduser())


class GuiApp:
    """Simple desktop UI for launching the analysis pipeline."""

    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.queue: "queue.Queue[QueueItem]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.last_report_path: Optional[pathlib.Path] = None
        self.running = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.preview_var = tk.StringVar()
        self.decisions_var = tk.StringVar()
        self.config_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.phase_var = tk.StringVar(value="")
        self._refresh_job: Optional[str] = None

        self._build_ui()
        self._apply_startup_config()
        self._bind_state_refresh()
        self._refresh_action_states()
        self.root.after(120, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("Automatic Image Culling")
        self.root.geometry("860x620")
        self.root.minsize(760, 540)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(4, weight=1)

        title = ttk.Label(
            container, text="Automatic Image Culling", font=("TkDefaultFont", 15, "bold")
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        form = ttk.Frame(container)
        form.grid(row=1, column=0, sticky="ew")
        form.grid_columnconfigure(1, weight=1)

        self._add_path_row(
            form, 0, "Input folder", self.input_var, self._browse_input
        )
        self._add_path_row(
            form, 1, "Output folder", self.output_var, self._browse_output
        )
        self._add_path_row(
            form, 2, "Preview folder", self.preview_var, self._browse_preview
        )

        decisions_label = ttk.Label(form, text="Decisions file")
        decisions_label.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.decisions_entry = ttk.Entry(form, textvariable=self.decisions_var)
        self.decisions_entry.grid(row=3, column=1, sticky="ew", pady=4)
        self.decisions_browse_btn = ttk.Button(
            form, text="Browse", command=self._browse_decisions
        )
        self.decisions_browse_btn.grid(
            row=3, column=2, sticky="ew", pady=4, padx=(4, 0)
        )

        config_label = ttk.Label(form, text="Config file (optional)")
        config_label.grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        self.config_entry = ttk.Entry(form, textvariable=self.config_var)
        self.config_entry.grid(row=4, column=1, sticky="ew", pady=4)
        self.config_browse_btn = ttk.Button(
            form, text="Browse", command=self._browse_config
        )
        self.config_browse_btn.grid(row=4, column=2, sticky="ew", pady=4, padx=(4, 0))
        self.config_load_btn = ttk.Button(
            form, text="Load", command=self._load_config
        )
        self.config_load_btn.grid(row=4, column=3, sticky="ew", pady=4, padx=(4, 0))

        actions = ttk.Frame(container)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        actions.grid_columnconfigure(2, weight=1)

        self.scan_btn = ttk.Button(actions, text="Discover", command=self._start_scan)
        self.scan_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.previews_btn = ttk.Button(
            actions, text="Generate previews", command=self._start_previews
        )
        self.previews_btn.grid(row=0, column=1, sticky="ew", padx=6, pady=(0, 6))
        self.analyze_btn = ttk.Button(
            actions, text="Analyze + report", command=self._start_analyze
        )
        self.analyze_btn.grid(row=0, column=2, sticky="ew", padx=(6, 0), pady=(0, 6))

        self.decisions_dry_btn = ttk.Button(
            actions,
            text="Apply decisions (dry run)",
            command=lambda: self._start_decisions(False),
        )
        self.decisions_dry_btn.grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0)
        )
        self.decisions_apply_btn = ttk.Button(
            actions,
            text="Apply decisions (move)",
            command=lambda: self._start_decisions(True),
        )
        self.decisions_apply_btn.grid(
            row=1, column=1, sticky="ew", padx=6, pady=(6, 0)
        )
        self.open_btn = ttk.Button(
            actions, text="Open report", command=self._open_report, state="disabled"
        )
        self.open_btn.grid(row=1, column=2, sticky="ew", padx=(6, 0), pady=(6, 0))

        status_frame = ttk.Frame(container)
        status_frame.grid(row=3, column=0, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_frame, textvariable=self.phase_var).grid(
            row=0, column=1, sticky="e"
        )

        self.progress = ttk.Progressbar(
            container, mode="determinate", maximum=1, value=0
        )
        self.progress.grid(row=4, column=0, sticky="ew", pady=(4, 8))

        log_frame = ttk.Frame(container)
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        self.input_entry = self._entry_by_var(self.input_var, form)
        self.output_entry = self._entry_by_var(self.output_var, form)
        self.preview_entry = self._entry_by_var(self.preview_var, form)
        self._controls = [
            self.input_entry,
            self.output_entry,
            self.preview_entry,
            self.decisions_entry,
            self.config_entry,
            self.decisions_browse_btn,
            self.config_browse_btn,
            self.config_load_btn,
            self.scan_btn,
            self.previews_btn,
            self.analyze_btn,
            self.decisions_dry_btn,
            self.decisions_apply_btn,
            self.open_btn,
        ]
        self._action_buttons = [
            self.scan_btn,
            self.previews_btn,
            self.analyze_btn,
            self.decisions_dry_btn,
            self.decisions_apply_btn,
            self.open_btn,
        ]

    def _apply_startup_config(self) -> None:
        """Populate fields from config.yaml when available; otherwise defaults."""
        defaults = load_config(None)
        cfg_path = pathlib.Path("config.yaml")
        if cfg_path.exists() and not self.config_var.get():
            self.config_var.set(str(cfg_path))
        if not self.input_var.get():
            self.input_var.set(defaults.get("input_dir", ""))
        if not self.output_var.get():
            self.output_var.set(defaults.get("output_dir", ""))
        if not self.preview_var.get():
            self.preview_var.set(defaults.get("preview_dir", ""))
        self._maybe_set_decisions_path(defaults)

    def _bind_state_refresh(self) -> None:
        for var in (
            self.input_var,
            self.output_var,
            self.preview_var,
            self.decisions_var,
            self.config_var,
        ):
            var.trace_add("write", lambda *_: self._schedule_state_refresh())

    def _schedule_state_refresh(self) -> None:
        if self._refresh_job is not None:
            return
        self._refresh_job = self.root.after(250, self._refresh_action_states)

    def _maybe_set_decisions_path(self, cfg: AppConfig) -> None:
        """Use output_dir/analysis/decisions.json when it exists."""
        if self.decisions_var.get():
            return
        output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
        candidate = output_dir / "analysis" / "decisions.json"
        if candidate.exists():
            self.decisions_var.set(str(candidate))

    @staticmethod
    def _entry_by_var(var: "tk.StringVar", parent: "tk.Widget") -> "ttk.Entry":
        for child in parent.winfo_children():
            if isinstance(child, ttk.Entry) and child.cget("textvariable") == str(var):
                return child
        raise RuntimeError("Entry widget not found")

    def _add_path_row(
        self,
        parent: "tk.Widget",
        row: int,
        label: str,
        variable: "tk.StringVar",
        browse_cmd: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(parent, text="Browse", command=browse_cmd)
        button.grid(row=row, column=2, sticky="ew", pady=4, padx=(4, 0))

    def _browse_input(self) -> None:
        self._browse_directory(self.input_var, "Select input folder")

    def _browse_output(self) -> None:
        self._browse_directory(self.output_var, "Select output folder")

    def _browse_preview(self) -> None:
        self._browse_directory(self.preview_var, "Select preview folder")

    def _browse_decisions(self) -> None:
        if filedialog is None:
            return
        selected = filedialog.askopenfilename(
            title="Select decisions.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.decisions_var.set(selected)

    def _browse_directory(self, var: "tk.StringVar", title: str) -> None:
        if filedialog is None:
            return
        selected = filedialog.askdirectory(title=title)
        if selected:
            var.set(selected)

    def _browse_config(self) -> None:
        if filedialog is None:
            return
        selected = filedialog.askopenfilename(
            title="Select config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.config_var.set(selected)

    def _load_config(self) -> None:
        path = self.config_var.get().strip()
        if not path:
            self._show_message("Select a config file to load.")
            return
        cfg_path = pathlib.Path(_clean_path(path))
        if not cfg_path.exists():
            self._show_message(f"Config not found: {cfg_path}")
            return
        cfg = load_config(str(cfg_path))
        self.input_var.set(cfg.get("input_dir", ""))
        self.output_var.set(cfg.get("output_dir", ""))
        self.preview_var.set(cfg.get("preview_dir", ""))
        self._maybe_set_decisions_path(cfg)
        self._append_log(f"Loaded config from {cfg_path}")

    def _start_task(
        self,
        action_label: str,
        target: Callable[..., None],
        *,
        clear_report: bool = False,
        target_args: Tuple[object, ...] = (),
    ) -> None:
        if self.running:
            return
        try:
            cfg = self._build_config()
        except ValueError as exc:
            self._show_message(str(exc))
            return
        self._append_log(f"Starting {action_label}")
        self._set_running(True, action_label)
        if clear_report:
            self.last_report_path = None
            self.open_btn.configure(state="disabled")
        self.worker = threading.Thread(
            target=target, args=(cfg, *target_args), daemon=True
        )
        self.worker.start()

    def _start_scan(self) -> None:
        self._start_task("discover", self._run_scan)

    def _start_previews(self) -> None:
        self._start_task("preview generation", self._run_previews)

    def _start_analyze(self) -> None:
        self._start_task("analysis", self._run_analysis, clear_report=True)

    def _start_decisions(self, apply_changes: bool) -> None:
        if apply_changes and messagebox is not None:
            confirm = messagebox.askyesno(
                "Automatic Image Culling",
                "This will move files based on decisions.json. Continue?",
            )
            if not confirm:
                return
        label = "decisions (apply)" if apply_changes else "decisions (dry run)"
        self._start_task(label, self._run_decisions, target_args=(apply_changes,))

    def _build_config(self) -> AppConfig:
        cfg_path = self.config_var.get().strip()
        cfg = load_config(cfg_path or None)

        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        preview_dir = self.preview_var.get().strip()

        if input_dir:
            cfg["input_dir"] = _clean_path(input_dir)
        if output_dir:
            cfg["output_dir"] = _clean_path(output_dir)
        if preview_dir:
            cfg["preview_dir"] = _clean_path(preview_dir)

        input_value = cfg.get("input_dir")
        if not input_value:
            raise ValueError("Input folder is required.")
        if not pathlib.Path(input_value).exists():
            raise ValueError(f"Input folder not found: {input_value}")
        return cfg

    def _read_config_for_state(self) -> AppConfig:
        cfg_path = self.config_var.get().strip()
        cfg = load_config(cfg_path or None)

        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        preview_dir = self.preview_var.get().strip()

        if input_dir:
            cfg["input_dir"] = _clean_path(input_dir)
        if output_dir:
            cfg["output_dir"] = _clean_path(output_dir)
        if preview_dir:
            cfg["preview_dir"] = _clean_path(preview_dir)

        return cfg

    @staticmethod
    def _resolve_analysis_paths(cfg: AppConfig) -> Tuple[pathlib.Path, pathlib.Path]:
        analysis_cfg = cast(AnalysisConfig, dict(cfg.get("analysis") or {}))
        output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
        analysis_dir = output_dir / "analysis"

        report_path = pathlib.Path(analysis_cfg.get("report_path", "report.html"))
        if not report_path.is_absolute():
            report_path = analysis_dir / report_path

        results_path = pathlib.Path(analysis_cfg.get("results_path", "analysis.json"))
        if not results_path.is_absolute():
            results_path = analysis_dir / results_path

        return report_path, results_path

    def _resolve_decisions_path(self, cfg: AppConfig) -> pathlib.Path:
        raw = self.decisions_var.get().strip()
        if raw:
            return pathlib.Path(_clean_path(raw))
        output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
        return output_dir / "analysis" / "decisions.json"

    @staticmethod
    def _has_arw_files(directory: pathlib.Path) -> bool:
        if not directory.exists():
            return False
        for _, __, filenames in os.walk(directory):
            for name in filenames:
                if name.lower().endswith(".arw"):
                    return True
        return False

    @staticmethod
    def _has_preview_files(directory: pathlib.Path, fmt: str) -> bool:
        if not directory.exists():
            return False
        suffix = f".{fmt.lower().lstrip('.')}" if fmt else ""
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            if not suffix or entry.suffix.lower() == suffix:
                return True
        return False

    def _refresh_action_states(self) -> None:
        self._refresh_job = None
        if self.running:
            for button in self._action_buttons:
                button.configure(state="disabled")
            return

        cfg = self._read_config_for_state()
        input_dir = pathlib.Path(cfg.get("input_dir", ""))
        output_dir = pathlib.Path(cfg.get("output_dir", "./output"))
        preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
        preview_cfg = cast(PreviewConfig, cfg.get("preview") or {})
        preview_fmt = str(preview_cfg.get("format", "webp"))

        input_exists = input_dir.exists()
        arw_exists = input_exists and self._has_arw_files(input_dir)
        previews_exist = self._has_preview_files(preview_dir, preview_fmt)

        report_path, results_path = self._resolve_analysis_paths(cfg)
        analysis_done = report_path.exists() or results_path.exists()

        decisions_path = self._resolve_decisions_path(cfg)
        if decisions_path.exists() and not self.decisions_var.get():
            self.decisions_var.set(str(decisions_path))
        decisions_exist = decisions_path.exists()

        self.scan_btn.configure(state="normal" if input_exists else "disabled")
        self.previews_btn.configure(state="normal" if arw_exists else "disabled")
        self.analyze_btn.configure(state="normal" if previews_exist else "disabled")

        decisions_state = "normal" if analysis_done and decisions_exist else "disabled"
        self.decisions_dry_btn.configure(state=decisions_state)
        self.decisions_apply_btn.configure(state=decisions_state)

        self.open_btn.configure(state="normal" if analysis_done else "disabled")

    def _run_scan(self, cfg: AppConfig) -> None:
        try:
            input_dir = cfg.get("input_dir", "./input")
            files = find_arw_files(input_dir, exclude_dirs=_exclude_list(cfg))
            if not files:
                self._send("log", "No .ARW files found.")
                self._send("done", None)
                self._send("log", "Done with discover")
                return
            total = len(files)
            self._send("log", f"Found {total} .ARW files.")
            self._send("progress", ("Discover", 0, total))

            max_log = 25
            for idx, path in enumerate(files, 1):
                if idx <= max_log:
                    exif = read_exif(path)
                    fallback_dt = _dt.datetime.fromtimestamp(path.stat().st_mtime)
                    capture = capture_date(exif, fallback=fallback_dt)
                    self._send("log", f"{path} | {capture}")
                if idx % 25 == 0 or idx == total:
                    self._send("progress", ("Discover", idx, total))
            if total > max_log:
                self._send("log", f"... ({total - max_log} more files)")
            self._send("done", None)
            self._send("log", "Done with discover")
        except Exception as exc:
            self._send("error", str(exc))

    def _generate_previews(
        self,
        files: list[pathlib.Path],
        preview_dir: pathlib.Path,
        preview_cfg: PreviewConfig,
        concurrency: int,
        label: str,
    ) -> bool:
        missing_dep = False
        total = len(files)
        self._send("progress", (label, 0, total))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {
                pool.submit(generate_preview, path, preview_dir, preview_cfg): path
                for path in files
            }
            done = 0
            for future in as_completed(future_map):
                path = future_map[future]
                try:
                    result = future.result()
                    if result is None:
                        missing_dep = True
                except Exception as exc:
                    self._send("log", f"Preview failed for {path}: {exc}")
                done += 1
                if done % 10 == 0 or done == total:
                    self._send("progress", (label, done, total))
        return missing_dep

    def _run_previews(self, cfg: AppConfig) -> None:
        try:
            input_dir = cfg.get("input_dir", "./input")
            files = find_arw_files(input_dir, exclude_dirs=_exclude_list(cfg))
            if not files:
                self._send("log", "No .ARW files found.")
                self._send("done", None)
                self._send("log", "Done with previews")
                return
            total = len(files)
            self._send("log", f"Found {total} .ARW files.")

            preview_cfg = _preview_config(cfg)
            preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
            preview_dir.mkdir(parents=True, exist_ok=True)
            workers = max(1, int(cfg.get("concurrency", 4) or 4))
            missing_dep = self._generate_previews(
                files, preview_dir, preview_cfg, workers, "Previews"
            )
            if missing_dep:
                self._send(
                    "log",
                    "Preview generation unavailable; check rawpy/Pillow installation.",
                )
            self._send("log", f"Previews processed: {total}")
            self._send("done", None)
            self._send("log", "Done with previews")
        except Exception as exc:
            self._send("error", str(exc))

    def _run_decisions(self, cfg: AppConfig, apply_changes: bool) -> None:
        try:
            decisions_path = self.decisions_var.get().strip()
            if not decisions_path:
                self._send("error", "Decisions file is required.")
                return
            decisions_file = pathlib.Path(_clean_path(decisions_path))
            if not decisions_file.exists():
                self._send("error", f"Decisions file not found: {decisions_file}")
                return
            self._send(
                "log",
                f"Using decisions file: {decisions_file}",
            )

            def progress_cb(current: int, total: int) -> None:
                self._send("progress", ("Decisions", current, total))

            summary = apply_decisions(
                decisions_file,
                cfg,
                apply=apply_changes,
                copy_files=False,
                progress_cb=progress_cb,
                log_cb=lambda msg: self._send("log", msg),
            )
            if not apply_changes:
                self._send(
                    "log",
                    "Dry run complete; use Apply decisions (move) to move files.",
                )
            self._send(
                "log",
                "Decisions summary: "
                f"total={summary.total} keep={summary.keep} discard={summary.discard} "
                f"moved={summary.moved} copied={summary.copied} "
                f"missing={summary.missing} skipped={summary.skipped}",
            )
            self._send("done", None)
            label = "decisions (apply)" if apply_changes else "decisions (dry run)"
            self._send("log", f"Done with {label}")
        except Exception as exc:
            self._send("error", str(exc))

    def _run_analysis(self, cfg: AppConfig) -> None:
        try:
            input_dir = cfg.get("input_dir", "./input")
            files = find_arw_files(input_dir, exclude_dirs=_exclude_list(cfg))
            if not files:
                self._send("log", "No .ARW files found.")
                self._send("done", None)
                self._send("log", "Done with analysis")
                return
            total = len(files)
            self._send("log", f"Found {total} .ARW files.")

            preview_cfg = _preview_config(cfg)
            preview_dir = pathlib.Path(cfg.get("preview_dir", "./previews"))
            preview_dir.mkdir(parents=True, exist_ok=True)
            workers = max(1, int(cfg.get("concurrency", 4) or 4))
            missing_dep = self._generate_previews(
                files, preview_dir, preview_cfg, workers, "Previews"
            )
            if missing_dep:
                self._send(
                    "log",
                    "Preview generation unavailable; check rawpy/Pillow installation.",
                )

            analysis_dir = pathlib.Path(cfg.get("output_dir", "./output")) / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_cfg = _prepare_analysis_config(cfg, analysis_dir)
            cfg["analysis"] = analysis_cfg

            self._send("progress", ("Analyze", 0, total))
            analyzed = 0

            def progress_cb(amount: int) -> None:
                nonlocal analyzed
                analyzed += amount
                self._send("progress", ("Analyze", analyzed, total))

            results = analyze_files(
                cfg, files, preview_dir, preview_cfg, progress_cb=progress_cb
            )
            write_outputs(results, analysis_cfg)
            report_path = pathlib.Path(analysis_cfg["report_path"])
            self._send("log", f"Analysis written to {analysis_cfg['results_path']}")
            self._send("log", f"HTML report written to {analysis_cfg['report_path']}")
            self._send("done", report_path)
            self._send("log", "Done with analysis")
        except Exception as exc:
            self._send("error", str(exc))

    def _send(self, kind: str, payload: object) -> None:
        self.queue.put((kind, payload))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    phase, current, total = cast(Tuple[str, int, int], payload)
                    self._update_progress(phase, current, total)
                elif kind == "done":
                    report = payload if isinstance(payload, pathlib.Path) else None
                    if report is not None:
                        self.last_report_path = report
                        self.open_btn.configure(state="normal")
                    self._set_running(False)
                elif kind == "error":
                    self._append_log(f"Error: {payload}")
                    self._show_message(str(payload))
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _update_progress(self, phase: str, current: int, total: int) -> None:
        if total <= 0:
            self.progress.configure(maximum=1, value=0)
            self.phase_var.set("")
            return
        self.progress.configure(maximum=total, value=current)
        self.phase_var.set(f"{phase}: {current}/{total}")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running: bool, action: str = "") -> None:
        self.running = running
        state = "disabled" if running else "normal"
        for widget in self._controls:
            widget.configure(state=state)
        if running:
            label = f"Running: {action}" if action else "Running"
            self.status_var.set(label)
        else:
            self.status_var.set("Idle")
        if not running:
            self.phase_var.set("")
            if self.progress["value"] != self.progress["maximum"]:
                self.progress.configure(value=0)
        self._schedule_state_refresh()

    def _open_report(self) -> None:
        if not self.last_report_path:
            self._show_message("No report available yet.")
            return
        report_path = self.last_report_path
        if not report_path.exists():
            self._show_message(f"Report not found: {report_path}")
            return
        webbrowser.open(report_path.resolve().as_uri())

    def _show_message(self, message: str) -> None:
        if messagebox is None:
            return
        messagebox.showinfo("Automatic Image Culling", message)

    def _on_close(self) -> None:
        if self.running:
            if messagebox is None:
                return
            confirm = messagebox.askyesno(
                "Automatic Image Culling",
                "Analysis is still running. Quit anyway?",
            )
            if not confirm:
                return
        self.root.destroy()


def main() -> None:
    """Entry point for launching the GUI."""
    if tk is None:  # pragma: no cover
        raise SystemExit(
            "Tkinter is not available. Install a Python build with Tk support."
        ) from _TK_IMPORT_ERROR
    root = tk.Tk()
    app = GuiApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
