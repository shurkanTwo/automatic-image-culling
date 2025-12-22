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
from .config import (
    DEFAULT_CONFIG,
    AnalysisConfig,
    AppConfig,
    FaceConfig,
    PreviewConfig,
    default_config,
    load_config,
    save_config,
)
from .paths import (
    analysis_dir_for_input,
    decisions_path_for_input,
    drop_path_config,
    input_dir_from_cfg,
    output_dir_for_input,
    preview_dir_for_input,
)
from .decisions import apply_decisions
from .discovery import capture_date, find_arw_files, read_exif
from .preview import generate_preview

QueueItem = Tuple[str, object]


class _CancelError(Exception):
    pass


class _Tooltip:
    """Lightweight tooltip helper for Tk widgets."""

    def __init__(
        self,
        widget: "tk.Widget",
        text: str,
        *,
        delay_ms: int = 400,
        wraplength: int = 360,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._job: Optional[str] = None
        self._tip_window: Optional["tk.Toplevel"] = None
        self._last_xy: Optional[Tuple[int, int]] = None
        self.widget.bind("<Enter>", self._schedule, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<Motion>", self._track_motion, add="+")

    def _schedule(self, event: "tk.Event") -> None:
        self._cancel()
        self._last_xy = (event.x_root, event.y_root)
        self._job = self.widget.after(self.delay_ms, self._show)

    def _track_motion(self, event: "tk.Event") -> None:
        self._last_xy = (event.x_root, event.y_root)
        if self._tip_window is not None:
            self._position_tip()

    def _cancel(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self) -> None:
        if self._tip_window is not None or not self.text:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
        )
        label.pack(ipadx=6, ipady=3)
        self._tip_window = tip
        self._position_tip()

    def _position_tip(self) -> None:
        if self._tip_window is None:
            return
        x, y = self._last_xy or (self.widget.winfo_rootx(), self.widget.winfo_rooty())
        x += 16
        y += 16
        self._tip_window.update_idletasks()
        width = self._tip_window.winfo_width()
        height = self._tip_window.winfo_height()
        screen_w = self.widget.winfo_screenwidth()
        screen_h = self.widget.winfo_screenheight()
        if x + width > screen_w:
            x = max(0, screen_w - width - 8)
        if y + height > screen_h:
            y = max(0, screen_h - height - 8)
        self._tip_window.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event: Optional["tk.Event"] = None) -> None:
        self._cancel()
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None


def _exclude_list(cfg: AppConfig) -> list[str]:
    """Return a de-duplicated list of directories to ignore."""
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


def _preview_config(cfg: AppConfig) -> PreviewConfig:
    """Return a defensive copy of the preview configuration."""
    preview_cfg = dict(DEFAULT_CONFIG["preview"])
    preview_cfg.update(cfg.get("preview") or {})
    return cast(PreviewConfig, preview_cfg)


def _prepare_analysis_config(
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
        self.config_var = tk.StringVar()
        self.preview_long_edge_var = tk.StringVar()
        self.preview_format_var = tk.StringVar()
        self.preview_quality_var = tk.StringVar()
        self.exclude_dirs_var = tk.StringVar()
        self.concurrency_var = tk.StringVar()
        self.analysis_sharpness_min_var = tk.StringVar()
        self.analysis_center_sharpness_min_var = tk.StringVar()
        self.analysis_tenengrad_min_var = tk.StringVar()
        self.analysis_motion_ratio_min_var = tk.StringVar()
        self.analysis_noise_std_max_var = tk.StringVar()
        self.analysis_brightness_min_var = tk.StringVar()
        self.analysis_brightness_max_var = tk.StringVar()
        self.analysis_shadows_min_var = tk.StringVar()
        self.analysis_shadows_max_var = tk.StringVar()
        self.analysis_highlights_min_var = tk.StringVar()
        self.analysis_highlights_max_var = tk.StringVar()
        self.analysis_quality_score_min_var = tk.StringVar()
        self.analysis_hard_fail_sharp_ratio_var = tk.StringVar()
        self.analysis_hard_fail_sharp_center_ratio_var = tk.StringVar()
        self.analysis_hard_fail_teneng_ratio_var = tk.StringVar()
        self.analysis_hard_fail_motion_ratio_var = tk.StringVar()
        self.analysis_hard_fail_brightness_ratio_var = tk.StringVar()
        self.analysis_hard_fail_noise_ratio_var = tk.StringVar()
        self.analysis_hard_fail_shadows_ratio_var = tk.StringVar()
        self.analysis_hard_fail_highlights_ratio_var = tk.StringVar()
        self.analysis_hard_fail_composition_ratio_var = tk.StringVar()
        self.analysis_duplicate_hamming_var = tk.StringVar()
        self.analysis_duplicate_window_seconds_var = tk.StringVar()
        self.analysis_duplicate_bucket_bits_var = tk.StringVar()
        self.analysis_report_path_var = tk.StringVar()
        self.analysis_results_path_var = tk.StringVar()
        self.analysis_face_enabled_var = tk.BooleanVar()
        self.analysis_face_backend_var = tk.StringVar()
        self.analysis_face_det_size_var = tk.StringVar()
        self.analysis_face_ctx_id_var = tk.StringVar()
        self.analysis_face_allowed_modules_var = tk.StringVar()
        self.analysis_face_providers_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.phase_var = tk.StringVar(value="")
        self._refresh_job: Optional[str] = None
        self._cancel_event = threading.Event()
        self._tooltips: list[_Tooltip] = []

        self._build_ui()
        self._apply_startup_config()
        self._bind_state_refresh()
        self._refresh_action_states()
        self.root.after(120, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        scale = self._apply_scaling()
        self.root.title("Automatic Image Culling")
        self._apply_theme()
        base_width, base_height = 860, 620
        min_width, min_height = 760, 540
        width = int(base_width * scale)
        height = int(base_height * scale)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(int(min_width * scale), int(min_height * scale))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        title = ttk.Label(
            container,
            text="Automatic Image Culling",
            font=("TkDefaultFont", 15, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        notebook = ttk.Notebook(container)
        notebook.grid(row=1, column=0, sticky="nsew")

        run_tab = ttk.Frame(notebook, padding=12)
        config_tab = ttk.Frame(notebook, padding=12)
        notebook.add(run_tab, text="Run")
        notebook.add(config_tab, text="Configuration")

        run_tab.grid_rowconfigure(4, weight=1)
        run_tab.grid_columnconfigure(0, weight=1)

        tooltips = {
            "input_dir": (
                "Folder that contains RAW images to process.\n" "Default: ./input."
            ),
            "config_file": (
                "YAML configuration file to load/save.\n" "Default: config.yaml."
            ),
            "concurrency": (
                "Number of worker threads used for preview generation and analysis.\n"
                "Default: 4. Range: >=1."
            ),
            "exclude_dirs": (
                "Comma-separated folders to skip during discovery/analysis. "
                "Analysis/output/previews are always skipped.\n"
                "Default: analysis, output, previews."
            ),
            "preview_long_edge": (
                "Maximum pixel size of the preview's long edge (keeps aspect ratio).\n"
                "Default: 2048. Range: >=1."
            ),
            "preview_format": (
                "File format for generated previews (Pillow format name).\n"
                "Default: webp."
            ),
            "preview_quality": (
                "Compression quality for previews.\n" "Default: 85. Range: 0-100."
            ),
            "sharpness_min": (
                "Minimum focus score (variance of Laplacian) to avoid rejection.\n"
                "Default: 8.0. Range: >=0."
            ),
            "center_sharpness_min": (
                "Minimum focus score in the center crop.\n"
                "Leave blank to use 1.2x sharpness min.\n"
                "Default: 1.2x sharpness min. Range: >=0."
            ),
            "tenengrad_min": (
                "Minimum edge contrast score (Tenengrad).\n"
                "Default: 200.0. Range: >=0."
            ),
            "motion_ratio_min": (
                "Minimum structure-tensor ratio; lower values indicate motion blur.\n"
                "Default: 0.02. Range: 0-1."
            ),
            "noise_std_max": (
                "Maximum noise estimate (pixel intensity std dev).\n"
                "Default: 25.0. Range: >=0."
            ),
            "brightness_min": (
                "Minimum average brightness (0=black, 1=white).\n"
                "Default: 0.08. Range: 0-1."
            ),
            "brightness_max": (
                "Maximum average brightness (0=black, 1=white).\n"
                "Default: 0.92. Range: 0-1."
            ),
            "shadows_min": (
                "Minimum fraction of shadow pixels (<20% brightness).\n"
                "Default: 0.0. Range: 0-1."
            ),
            "shadows_max": (
                "Maximum fraction of shadow pixels (<20% brightness).\n"
                "Default: 0.5. Range: 0-1."
            ),
            "highlights_min": (
                "Minimum fraction of highlight pixels (>70% brightness).\n"
                "Default: 0.0. Range: 0-1."
            ),
            "highlights_max": (
                "Maximum fraction of highlight pixels (>70% brightness).\n"
                "Default: 0.1. Range: 0-1."
            ),
            "quality_score_min": (
                "Minimum overall quality score to keep a frame.\n"
                "Default: 0.75. Range: 0-1."
            ),
            "hard_fail_sharp_ratio": (
                "Minimum normalized sharpness score before hard fail.\n"
                "Default: 0.55. Range: 0-1."
            ),
            "hard_fail_sharp_center_ratio": (
                "Minimum normalized center sharpness score before hard fail.\n"
                "Default: 0.55. Range: 0-1."
            ),
            "hard_fail_teneng_ratio": (
                "Minimum normalized Tenengrad score before hard fail.\n"
                "Default: 0.55. Range: 0-1."
            ),
            "hard_fail_motion_ratio": (
                "Minimum normalized motion score before hard fail.\n"
                "Default: 0.55. Range: 0-1."
            ),
            "hard_fail_brightness_ratio": (
                "Minimum normalized brightness score before hard fail.\n"
                "Default: 0.5. Range: 0-1."
            ),
            "hard_fail_noise_ratio": (
                "Minimum normalized noise score before hard fail.\n"
                "Default: 0.45. Range: 0-1."
            ),
            "hard_fail_shadows_ratio": (
                "Minimum normalized shadows score before hard fail.\n"
                "Default: 0.5. Range: 0-1."
            ),
            "hard_fail_highlights_ratio": (
                "Minimum normalized highlights score before hard fail.\n"
                "Default: 0.5. Range: 0-1."
            ),
            "hard_fail_composition_ratio": (
                "Minimum normalized composition score before hard fail.\n"
                "Default: 0.4. Range: 0-1."
            ),
            "duplicate_hamming": (
                "Maximum pHash Hamming distance to group duplicates.\n"
                "Default: 6. Range: 0-64."
            ),
            "duplicate_window_seconds": (
                "Capture time window (seconds) for duplicate grouping.\n"
                "Default: 8. Range: >=0."
            ),
            "duplicate_bucket_bits": (
                "Number of high pHash bits to bucket comparisons (0 disables).\n"
                "Default: 8. Range: 0-64."
            ),
            "report_path": (
                "HTML report output path. Relative paths are under the analysis folder.\n"
                "Default: ./report.html."
            ),
            "results_path": (
                "Analysis JSON output path. Relative paths are under the analysis folder.\n"
                "Default: ./analysis.json."
            ),
            "face_enabled": (
                "Toggle face detection and scoring during analysis.\n" "Default: true."
            ),
            "face_backend": (
                "Face detection backend.\n"
                "Default: mediapipe. Options: mediapipe, insightface."
            ),
            "face_det_size": (
                "Input size for the InsightFace detector (square).\n"
                "Default: 640. Range: >=1."
            ),
            "face_ctx_id": (
                "InsightFace device index (GPU id).\n" "Default: 0. Range: integer."
            ),
            "face_allowed_modules": (
                "InsightFace modules to load (comma-separated).\n"
                "Default: detection, recognition."
            ),
            "face_providers": (
                "ONNX runtime providers for InsightFace (comma-separated).\n"
                "Leave blank for CPUExecutionProvider.\n"
                "Default: CPUExecutionProvider."
            ),
        }

        form = ttk.Frame(run_tab)
        form.grid(row=0, column=0, sticky="ew")
        form.grid_columnconfigure(1, weight=1)

        self._add_path_row(
            form,
            0,
            "Input folder",
            self.input_var,
            self._browse_input,
            tooltip=tooltips["input_dir"],
        )

        actions = ttk.Frame(run_tab)
        actions.grid(row=1, column=0, pady=(10, 6))
        actions.grid_columnconfigure(0, weight=1)

        button_width = 26
        self.scan_btn = ttk.Button(
            actions, text="Discover", command=self._start_scan, width=button_width
        )
        self.scan_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.analyze_btn = ttk.Button(
            actions,
            text="Analyze + report",
            command=self._start_analyze,
            width=button_width,
        )
        self.analyze_btn.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.open_btn = ttk.Button(
            actions,
            text="Open report",
            command=self._open_report,
            state="disabled",
            width=button_width,
        )
        self.open_btn.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.decisions_dry_btn = ttk.Button(
            actions,
            text="Apply decisions (dry run)",
            command=lambda: self._start_decisions(False),
            width=button_width,
        )
        self.decisions_dry_btn.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.decisions_apply_btn = ttk.Button(
            actions,
            text="Apply decisions (move)",
            command=lambda: self._start_decisions(True),
            width=button_width,
        )
        self.decisions_apply_btn.grid(row=4, column=0, sticky="ew")
        status_frame = ttk.Frame(run_tab)
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_frame, textvariable=self.phase_var).grid(
            row=0, column=1, sticky="e"
        )

        progress_frame = ttk.Frame(run_tab)
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(4, 8))
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(
            progress_frame, mode="determinate", maximum=1, value=0
        )
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_btn = ttk.Button(
            progress_frame, text="Stop", command=self._request_stop, state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, sticky="e")

        log_frame = ttk.Frame(run_tab)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        config_tab.grid_columnconfigure(0, weight=1)
        config_tab.grid_rowconfigure(1, weight=1)

        config_controls: list["tk.Widget"] = []

        config_file_frame = ttk.Frame(config_tab)
        config_file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        config_file_frame.grid_columnconfigure(1, weight=1)

        config_label = ttk.Label(config_file_frame, text="Config file")
        config_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.config_entry = ttk.Entry(config_file_frame, textvariable=self.config_var)
        self.config_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self._add_tooltip(config_label, tooltips["config_file"])
        self._add_tooltip(self.config_entry, tooltips["config_file"])
        self.config_browse_btn = ttk.Button(
            config_file_frame, text="Browse", command=self._browse_config
        )
        self.config_browse_btn.grid(row=0, column=2, sticky="ew", pady=4, padx=(4, 0))
        self.config_load_btn = ttk.Button(
            config_file_frame, text="Reload", command=self._load_config
        )
        self.config_load_btn.grid(row=0, column=3, sticky="ew", pady=4, padx=(4, 0))
        self.config_save_btn = ttk.Button(
            config_file_frame, text="Save", command=self._save_config
        )
        self.config_save_btn.grid(row=0, column=4, sticky="ew", pady=4, padx=(4, 0))
        self.config_reset_btn = ttk.Button(
            config_file_frame, text="Reset to defaults", command=self._reset_config
        )
        self.config_reset_btn.grid(row=0, column=5, sticky="ew", pady=4, padx=(4, 0))
        config_controls.extend(
            [
                self.config_entry,
                self.config_browse_btn,
                self.config_load_btn,
                self.config_save_btn,
                self.config_reset_btn,
            ]
        )

        config_canvas = tk.Canvas(config_tab, highlightthickness=0)
        config_scrollbar = ttk.Scrollbar(
            config_tab, orient="vertical", command=config_canvas.yview
        )
        config_canvas.grid(row=1, column=0, sticky="nsew")
        config_scrollbar.grid(row=1, column=1, sticky="ns")
        config_canvas.configure(yscrollcommand=config_scrollbar.set)

        config_body = ttk.Frame(config_canvas)
        config_body_id = config_canvas.create_window(
            (0, 0), window=config_body, anchor="nw"
        )
        config_body.grid_columnconfigure(0, weight=1)

        def _on_config_frame(event: "tk.Event") -> None:
            config_canvas.configure(scrollregion=config_canvas.bbox("all"))

        def _on_config_canvas(event: "tk.Event") -> None:
            config_canvas.itemconfigure(config_body_id, width=event.width)

        config_body.bind("<Configure>", _on_config_frame)
        config_canvas.bind("<Configure>", _on_config_canvas)

        runtime_frame = ttk.LabelFrame(config_body, text="Runtime")
        runtime_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        runtime_frame.grid_columnconfigure(1, weight=1)
        entry = self._add_config_entry(
            runtime_frame,
            0,
            "Concurrency",
            self.concurrency_var,
            tooltip=tooltips["concurrency"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            runtime_frame,
            1,
            "Exclude dirs (comma-separated)",
            self.exclude_dirs_var,
            tooltip=tooltips["exclude_dirs"],
        )
        config_controls.append(entry)

        preview_frame = ttk.LabelFrame(config_body, text="Preview")
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        preview_frame.grid_columnconfigure(1, weight=1)
        entry = self._add_config_entry(
            preview_frame,
            0,
            "Long edge (px)",
            self.preview_long_edge_var,
            tooltip=tooltips["preview_long_edge"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            preview_frame,
            1,
            "Format",
            self.preview_format_var,
            tooltip=tooltips["preview_format"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            preview_frame,
            2,
            "Quality",
            self.preview_quality_var,
            tooltip=tooltips["preview_quality"],
        )
        config_controls.append(entry)

        analysis_frame = ttk.LabelFrame(config_body, text="Analysis thresholds")
        analysis_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        analysis_frame.grid_columnconfigure(1, weight=1)
        entry = self._add_config_entry(
            analysis_frame,
            0,
            "Sharpness min",
            self.analysis_sharpness_min_var,
            tooltip=tooltips["sharpness_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            1,
            "Center sharpness min (optional)",
            self.analysis_center_sharpness_min_var,
            tooltip=tooltips["center_sharpness_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            2,
            "Tenengrad min",
            self.analysis_tenengrad_min_var,
            tooltip=tooltips["tenengrad_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            3,
            "Motion ratio min",
            self.analysis_motion_ratio_min_var,
            tooltip=tooltips["motion_ratio_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            4,
            "Noise std max",
            self.analysis_noise_std_max_var,
            tooltip=tooltips["noise_std_max"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            5,
            "Brightness min",
            self.analysis_brightness_min_var,
            tooltip=tooltips["brightness_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            6,
            "Brightness max",
            self.analysis_brightness_max_var,
            tooltip=tooltips["brightness_max"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            7,
            "Shadows min",
            self.analysis_shadows_min_var,
            tooltip=tooltips["shadows_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            8,
            "Shadows max",
            self.analysis_shadows_max_var,
            tooltip=tooltips["shadows_max"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            9,
            "Highlights min",
            self.analysis_highlights_min_var,
            tooltip=tooltips["highlights_min"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            10,
            "Highlights max",
            self.analysis_highlights_max_var,
            tooltip=tooltips["highlights_max"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            analysis_frame,
            11,
            "Quality score min",
            self.analysis_quality_score_min_var,
            tooltip=tooltips["quality_score_min"],
        )
        config_controls.append(entry)

        hard_fail_frame = ttk.LabelFrame(config_body, text="Hard-fail ratios")
        hard_fail_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        hard_fail_frame.grid_columnconfigure(1, weight=1)
        entry = self._add_config_entry(
            hard_fail_frame,
            0,
            "Sharpness ratio",
            self.analysis_hard_fail_sharp_ratio_var,
            tooltip=tooltips["hard_fail_sharp_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            1,
            "Center sharpness ratio",
            self.analysis_hard_fail_sharp_center_ratio_var,
            tooltip=tooltips["hard_fail_sharp_center_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            2,
            "Tenengrad ratio",
            self.analysis_hard_fail_teneng_ratio_var,
            tooltip=tooltips["hard_fail_teneng_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            3,
            "Motion ratio",
            self.analysis_hard_fail_motion_ratio_var,
            tooltip=tooltips["hard_fail_motion_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            4,
            "Brightness ratio",
            self.analysis_hard_fail_brightness_ratio_var,
            tooltip=tooltips["hard_fail_brightness_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            5,
            "Noise ratio",
            self.analysis_hard_fail_noise_ratio_var,
            tooltip=tooltips["hard_fail_noise_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            6,
            "Shadows ratio",
            self.analysis_hard_fail_shadows_ratio_var,
            tooltip=tooltips["hard_fail_shadows_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            7,
            "Highlights ratio",
            self.analysis_hard_fail_highlights_ratio_var,
            tooltip=tooltips["hard_fail_highlights_ratio"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            hard_fail_frame,
            8,
            "Composition ratio",
            self.analysis_hard_fail_composition_ratio_var,
            tooltip=tooltips["hard_fail_composition_ratio"],
        )
        config_controls.append(entry)

        duplicates_frame = ttk.LabelFrame(config_body, text="Duplicates & outputs")
        duplicates_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        duplicates_frame.grid_columnconfigure(1, weight=1)
        entry = self._add_config_entry(
            duplicates_frame,
            0,
            "Duplicate hamming",
            self.analysis_duplicate_hamming_var,
            tooltip=tooltips["duplicate_hamming"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            duplicates_frame,
            1,
            "Duplicate window seconds",
            self.analysis_duplicate_window_seconds_var,
            tooltip=tooltips["duplicate_window_seconds"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            duplicates_frame,
            2,
            "Duplicate bucket bits",
            self.analysis_duplicate_bucket_bits_var,
            tooltip=tooltips["duplicate_bucket_bits"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            duplicates_frame,
            3,
            "Report path",
            self.analysis_report_path_var,
            tooltip=tooltips["report_path"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            duplicates_frame,
            4,
            "Results path",
            self.analysis_results_path_var,
            tooltip=tooltips["results_path"],
        )
        config_controls.append(entry)

        face_frame = ttk.LabelFrame(config_body, text="Face detection")
        face_frame.grid(row=5, column=0, sticky="ew")
        face_frame.grid_columnconfigure(1, weight=1)
        face_enabled_label = ttk.Label(face_frame, text="Enabled")
        face_enabled_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        face_enabled = ttk.Checkbutton(
            face_frame, variable=self.analysis_face_enabled_var
        )
        face_enabled.grid(row=0, column=1, sticky="w", pady=4)
        self._add_tooltip(face_enabled_label, tooltips["face_enabled"])
        self._add_tooltip(face_enabled, tooltips["face_enabled"])
        config_controls.append(face_enabled)
        face_backend_label = ttk.Label(face_frame, text="Backend")
        face_backend_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        face_backend = ttk.Combobox(
            face_frame,
            textvariable=self.analysis_face_backend_var,
            values=("mediapipe", "insightface"),
            state="readonly",
        )
        face_backend.grid(row=1, column=1, sticky="ew", pady=4)
        self._add_tooltip(face_backend_label, tooltips["face_backend"])
        self._add_tooltip(face_backend, tooltips["face_backend"])
        config_controls.append(face_backend)
        entry = self._add_config_entry(
            face_frame,
            2,
            "Detection size",
            self.analysis_face_det_size_var,
            tooltip=tooltips["face_det_size"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            face_frame,
            3,
            "Context id",
            self.analysis_face_ctx_id_var,
            tooltip=tooltips["face_ctx_id"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            face_frame,
            4,
            "Allowed modules (comma-separated)",
            self.analysis_face_allowed_modules_var,
            tooltip=tooltips["face_allowed_modules"],
        )
        config_controls.append(entry)
        entry = self._add_config_entry(
            face_frame,
            5,
            "Providers (comma-separated, optional)",
            self.analysis_face_providers_var,
            tooltip=tooltips["face_providers"],
        )
        config_controls.append(entry)

        self._bind_mousewheel_recursive(config_body, config_canvas)

        self.input_entry = self._entry_by_var(self.input_var, form)
        self._controls = [
            self.input_entry,
            self.scan_btn,
            self.analyze_btn,
            self.decisions_dry_btn,
            self.decisions_apply_btn,
            self.open_btn,
        ]
        self._controls.extend(config_controls)
        self._action_buttons = [
            self.scan_btn,
            self.analyze_btn,
            self.decisions_dry_btn,
            self.decisions_apply_btn,
            self.open_btn,
        ]

    @staticmethod
    def _format_number(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    @staticmethod
    def _format_list(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return ", ".join(str(item) for item in value)

    @staticmethod
    def _parse_optional_int(value: str, label: str) -> Optional[int]:
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc

    @staticmethod
    def _parse_optional_float(value: str, label: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc

    @staticmethod
    def _parse_list(value: str) -> list[str]:
        if not value.strip():
            return []
        items: list[str] = []
        for line in value.splitlines():
            for part in line.split(","):
                part = part.strip()
                if part:
                    items.append(part)
        return items

    def _apply_config_to_vars(self, cfg: AppConfig) -> None:
        self.input_var.set(cfg.get("input_dir", DEFAULT_CONFIG["input_dir"]))
        self.exclude_dirs_var.set(
            self._format_list(cfg.get("exclude_dirs", DEFAULT_CONFIG["exclude_dirs"]))
        )
        self.concurrency_var.set(
            self._format_number(cfg.get("concurrency", DEFAULT_CONFIG["concurrency"]))
        )

        preview_cfg = cast(PreviewConfig, cfg.get("preview") or {})
        self.preview_long_edge_var.set(
            self._format_number(
                preview_cfg.get("long_edge", DEFAULT_CONFIG["preview"]["long_edge"])
            )
        )
        self.preview_format_var.set(
            str(preview_cfg.get("format", DEFAULT_CONFIG["preview"]["format"]))
        )
        self.preview_quality_var.set(
            self._format_number(
                preview_cfg.get("quality", DEFAULT_CONFIG["preview"]["quality"])
            )
        )

        analysis_cfg = cast(AnalysisConfig, cfg.get("analysis") or {})
        self.analysis_sharpness_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "sharpness_min", DEFAULT_CONFIG["analysis"]["sharpness_min"]
                )
            )
        )
        center_value = analysis_cfg.get("center_sharpness_min")
        self.analysis_center_sharpness_min_var.set(
            self._format_number(center_value) if center_value is not None else ""
        )
        self.analysis_tenengrad_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "tenengrad_min", DEFAULT_CONFIG["analysis"]["tenengrad_min"]
                )
            )
        )
        self.analysis_motion_ratio_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "motion_ratio_min",
                    DEFAULT_CONFIG["analysis"]["motion_ratio_min"],
                )
            )
        )
        self.analysis_noise_std_max_var.set(
            self._format_number(
                analysis_cfg.get(
                    "noise_std_max", DEFAULT_CONFIG["analysis"]["noise_std_max"]
                )
            )
        )
        self.analysis_brightness_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "brightness_min", DEFAULT_CONFIG["analysis"]["brightness_min"]
                )
            )
        )
        self.analysis_brightness_max_var.set(
            self._format_number(
                analysis_cfg.get(
                    "brightness_max", DEFAULT_CONFIG["analysis"]["brightness_max"]
                )
            )
        )
        self.analysis_shadows_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "shadows_min", DEFAULT_CONFIG["analysis"]["shadows_min"]
                )
            )
        )
        self.analysis_shadows_max_var.set(
            self._format_number(
                analysis_cfg.get(
                    "shadows_max", DEFAULT_CONFIG["analysis"]["shadows_max"]
                )
            )
        )
        self.analysis_highlights_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "highlights_min", DEFAULT_CONFIG["analysis"]["highlights_min"]
                )
            )
        )
        self.analysis_highlights_max_var.set(
            self._format_number(
                analysis_cfg.get(
                    "highlights_max", DEFAULT_CONFIG["analysis"]["highlights_max"]
                )
            )
        )
        self.analysis_quality_score_min_var.set(
            self._format_number(
                analysis_cfg.get(
                    "quality_score_min",
                    DEFAULT_CONFIG["analysis"]["quality_score_min"],
                )
            )
        )
        self.analysis_hard_fail_sharp_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_sharp_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_sharp_ratio"],
                )
            )
        )
        self.analysis_hard_fail_sharp_center_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_sharp_center_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_sharp_center_ratio"],
                )
            )
        )
        self.analysis_hard_fail_teneng_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_teneng_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_teneng_ratio"],
                )
            )
        )
        self.analysis_hard_fail_motion_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_motion_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_motion_ratio"],
                )
            )
        )
        self.analysis_hard_fail_brightness_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_brightness_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_brightness_ratio"],
                )
            )
        )
        self.analysis_hard_fail_noise_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_noise_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_noise_ratio"],
                )
            )
        )
        self.analysis_hard_fail_shadows_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_shadows_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_shadows_ratio"],
                )
            )
        )
        self.analysis_hard_fail_highlights_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_highlights_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_highlights_ratio"],
                )
            )
        )
        self.analysis_hard_fail_composition_ratio_var.set(
            self._format_number(
                analysis_cfg.get(
                    "hard_fail_composition_ratio",
                    DEFAULT_CONFIG["analysis"]["hard_fail_composition_ratio"],
                )
            )
        )
        self.analysis_duplicate_hamming_var.set(
            self._format_number(
                analysis_cfg.get(
                    "duplicate_hamming",
                    DEFAULT_CONFIG["analysis"]["duplicate_hamming"],
                )
            )
        )
        self.analysis_duplicate_window_seconds_var.set(
            self._format_number(
                analysis_cfg.get(
                    "duplicate_window_seconds",
                    DEFAULT_CONFIG["analysis"]["duplicate_window_seconds"],
                )
            )
        )
        self.analysis_duplicate_bucket_bits_var.set(
            self._format_number(
                analysis_cfg.get(
                    "duplicate_bucket_bits",
                    DEFAULT_CONFIG["analysis"]["duplicate_bucket_bits"],
                )
            )
        )
        self.analysis_report_path_var.set(
            str(
                analysis_cfg.get(
                    "report_path", DEFAULT_CONFIG["analysis"]["report_path"]
                )
            )
        )
        self.analysis_results_path_var.set(
            str(
                analysis_cfg.get(
                    "results_path", DEFAULT_CONFIG["analysis"]["results_path"]
                )
            )
        )

        face_cfg = cast(FaceConfig, analysis_cfg.get("face") or {})
        self.analysis_face_enabled_var.set(
            bool(face_cfg.get("enabled", DEFAULT_CONFIG["analysis"]["face"]["enabled"]))
        )
        self.analysis_face_backend_var.set(
            str(face_cfg.get("backend", DEFAULT_CONFIG["analysis"]["face"]["backend"]))
        )
        self.analysis_face_det_size_var.set(
            self._format_number(
                face_cfg.get("det_size", DEFAULT_CONFIG["analysis"]["face"]["det_size"])
            )
        )
        self.analysis_face_ctx_id_var.set(
            self._format_number(
                face_cfg.get("ctx_id", DEFAULT_CONFIG["analysis"]["face"]["ctx_id"])
            )
        )
        allowed_modules = face_cfg.get(
            "allowed_modules", DEFAULT_CONFIG["analysis"]["face"]["allowed_modules"]
        )
        self.analysis_face_allowed_modules_var.set(self._format_list(allowed_modules))
        providers = face_cfg.get("providers")
        self.analysis_face_providers_var.set(
            self._format_list(providers) if providers is not None else ""
        )

    def _collect_config(self) -> AppConfig:
        cfg_path = self.config_var.get().strip()
        cfg = load_config(cfg_path or None)
        drop_path_config(cfg)

        input_dir = self.input_var.get().strip()
        if input_dir:
            cfg["input_dir"] = _clean_path(input_dir)

        exclude_dirs = self._parse_list(self.exclude_dirs_var.get())
        if exclude_dirs:
            cfg["exclude_dirs"] = exclude_dirs
        else:
            cfg.pop("exclude_dirs", None)

        concurrency = self._parse_optional_int(
            self.concurrency_var.get(), "Concurrency"
        )
        if concurrency is not None:
            cfg["concurrency"] = concurrency

        preview_cfg = cast(PreviewConfig, dict(cfg.get("preview") or {}))
        long_edge = self._parse_optional_int(
            self.preview_long_edge_var.get(), "Preview long edge"
        )
        if long_edge is not None:
            preview_cfg["long_edge"] = long_edge
        fmt = self.preview_format_var.get().strip()
        if fmt:
            preview_cfg["format"] = fmt
        quality = self._parse_optional_int(
            self.preview_quality_var.get(), "Preview quality"
        )
        if quality is not None:
            preview_cfg["quality"] = quality
        cfg["preview"] = preview_cfg

        analysis_cfg = cast(AnalysisConfig, dict(cfg.get("analysis") or {}))
        sharpness = self._parse_optional_float(
            self.analysis_sharpness_min_var.get(), "Sharpness min"
        )
        if sharpness is not None:
            analysis_cfg["sharpness_min"] = sharpness
        center = self._parse_optional_float(
            self.analysis_center_sharpness_min_var.get(),
            "Center sharpness min",
        )
        if center is None:
            analysis_cfg.pop("center_sharpness_min", None)
        else:
            analysis_cfg["center_sharpness_min"] = center
        tenengrad = self._parse_optional_float(
            self.analysis_tenengrad_min_var.get(), "Tenengrad min"
        )
        if tenengrad is not None:
            analysis_cfg["tenengrad_min"] = tenengrad
        motion_ratio = self._parse_optional_float(
            self.analysis_motion_ratio_min_var.get(), "Motion ratio min"
        )
        if motion_ratio is not None:
            analysis_cfg["motion_ratio_min"] = motion_ratio
        noise_std = self._parse_optional_float(
            self.analysis_noise_std_max_var.get(), "Noise std max"
        )
        if noise_std is not None:
            analysis_cfg["noise_std_max"] = noise_std
        brightness_min = self._parse_optional_float(
            self.analysis_brightness_min_var.get(), "Brightness min"
        )
        if brightness_min is not None:
            analysis_cfg["brightness_min"] = brightness_min
        brightness_max = self._parse_optional_float(
            self.analysis_brightness_max_var.get(), "Brightness max"
        )
        if brightness_max is not None:
            analysis_cfg["brightness_max"] = brightness_max
        shadows_min = self._parse_optional_float(
            self.analysis_shadows_min_var.get(), "Shadows min"
        )
        if shadows_min is not None:
            analysis_cfg["shadows_min"] = shadows_min
        shadows_max = self._parse_optional_float(
            self.analysis_shadows_max_var.get(), "Shadows max"
        )
        if shadows_max is not None:
            analysis_cfg["shadows_max"] = shadows_max
        highlights_min = self._parse_optional_float(
            self.analysis_highlights_min_var.get(), "Highlights min"
        )
        if highlights_min is not None:
            analysis_cfg["highlights_min"] = highlights_min
        highlights_max = self._parse_optional_float(
            self.analysis_highlights_max_var.get(), "Highlights max"
        )
        if highlights_max is not None:
            analysis_cfg["highlights_max"] = highlights_max
        quality_min = self._parse_optional_float(
            self.analysis_quality_score_min_var.get(), "Quality score min"
        )
        if quality_min is not None:
            analysis_cfg["quality_score_min"] = quality_min
        hard_fail_sharp = self._parse_optional_float(
            self.analysis_hard_fail_sharp_ratio_var.get(), "Sharpness ratio"
        )
        if hard_fail_sharp is not None:
            analysis_cfg["hard_fail_sharp_ratio"] = hard_fail_sharp
        hard_fail_center = self._parse_optional_float(
            self.analysis_hard_fail_sharp_center_ratio_var.get(),
            "Center sharpness ratio",
        )
        if hard_fail_center is not None:
            analysis_cfg["hard_fail_sharp_center_ratio"] = hard_fail_center
        hard_fail_teneng = self._parse_optional_float(
            self.analysis_hard_fail_teneng_ratio_var.get(), "Tenengrad ratio"
        )
        if hard_fail_teneng is not None:
            analysis_cfg["hard_fail_teneng_ratio"] = hard_fail_teneng
        hard_fail_motion = self._parse_optional_float(
            self.analysis_hard_fail_motion_ratio_var.get(), "Motion ratio"
        )
        if hard_fail_motion is not None:
            analysis_cfg["hard_fail_motion_ratio"] = hard_fail_motion
        hard_fail_brightness = self._parse_optional_float(
            self.analysis_hard_fail_brightness_ratio_var.get(), "Brightness ratio"
        )
        if hard_fail_brightness is not None:
            analysis_cfg["hard_fail_brightness_ratio"] = hard_fail_brightness
        hard_fail_noise = self._parse_optional_float(
            self.analysis_hard_fail_noise_ratio_var.get(), "Noise ratio"
        )
        if hard_fail_noise is not None:
            analysis_cfg["hard_fail_noise_ratio"] = hard_fail_noise
        hard_fail_shadows = self._parse_optional_float(
            self.analysis_hard_fail_shadows_ratio_var.get(), "Shadows ratio"
        )
        if hard_fail_shadows is not None:
            analysis_cfg["hard_fail_shadows_ratio"] = hard_fail_shadows
        hard_fail_highlights = self._parse_optional_float(
            self.analysis_hard_fail_highlights_ratio_var.get(), "Highlights ratio"
        )
        if hard_fail_highlights is not None:
            analysis_cfg["hard_fail_highlights_ratio"] = hard_fail_highlights
        hard_fail_composition = self._parse_optional_float(
            self.analysis_hard_fail_composition_ratio_var.get(), "Composition ratio"
        )
        if hard_fail_composition is not None:
            analysis_cfg["hard_fail_composition_ratio"] = hard_fail_composition
        duplicate_hamming = self._parse_optional_int(
            self.analysis_duplicate_hamming_var.get(), "Duplicate hamming"
        )
        if duplicate_hamming is not None:
            analysis_cfg["duplicate_hamming"] = duplicate_hamming
        duplicate_window = self._parse_optional_float(
            self.analysis_duplicate_window_seconds_var.get(),
            "Duplicate window seconds",
        )
        if duplicate_window is not None:
            analysis_cfg["duplicate_window_seconds"] = duplicate_window
        duplicate_bucket = self._parse_optional_int(
            self.analysis_duplicate_bucket_bits_var.get(), "Duplicate bucket bits"
        )
        if duplicate_bucket is not None:
            analysis_cfg["duplicate_bucket_bits"] = duplicate_bucket
        report_path = self.analysis_report_path_var.get().strip()
        if report_path:
            analysis_cfg["report_path"] = report_path
        results_path = self.analysis_results_path_var.get().strip()
        if results_path:
            analysis_cfg["results_path"] = results_path

        face_cfg = cast(FaceConfig, dict(analysis_cfg.get("face") or {}))
        face_cfg["enabled"] = bool(self.analysis_face_enabled_var.get())
        backend = self.analysis_face_backend_var.get().strip()
        if backend:
            face_cfg["backend"] = backend
        det_size = self._parse_optional_int(
            self.analysis_face_det_size_var.get(), "Detection size"
        )
        if det_size is not None:
            face_cfg["det_size"] = det_size
        ctx_id = self._parse_optional_int(
            self.analysis_face_ctx_id_var.get(), "Context id"
        )
        if ctx_id is not None:
            face_cfg["ctx_id"] = ctx_id
        allowed_modules = self._parse_list(self.analysis_face_allowed_modules_var.get())
        if allowed_modules:
            face_cfg["allowed_modules"] = allowed_modules
        else:
            face_cfg.pop("allowed_modules", None)
        providers = self._parse_list(self.analysis_face_providers_var.get())
        if providers:
            face_cfg["providers"] = providers
        else:
            face_cfg.pop("providers", None)
        analysis_cfg["face"] = face_cfg
        cfg["analysis"] = analysis_cfg

        return cfg

    def _apply_startup_config(self) -> None:
        """Populate fields from config.yaml when available; otherwise defaults."""
        if not self.config_var.get():
            self.config_var.set("config.yaml")
        cfg_path = self.config_var.get().strip()
        cfg = load_config(cfg_path or None)
        drop_path_config(cfg)
        self._apply_config_to_vars(cfg)

    def _bind_state_refresh(self) -> None:
        for var in (
            self.input_var,
            self.preview_format_var,
            self.config_var,
        ):
            var.trace_add("write", lambda *__: self._schedule_state_refresh())

    def _schedule_state_refresh(self) -> None:
        if self._refresh_job is not None:
            return
        self._refresh_job = self.root.after(250, self._refresh_action_states)

    @staticmethod
    def _entry_by_var(var: "tk.StringVar", parent: "tk.Widget") -> "ttk.Entry":
        for child in parent.winfo_children():
            if isinstance(child, ttk.Entry) and child.cget("textvariable") == str(var):
                return child
        raise RuntimeError("Entry widget not found")

    def _apply_scaling(self) -> float:
        """Scale the UI for high-DPI displays."""
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        scale = min(width / 1920.0, height / 1080.0)
        scale = max(1.0, min(2.0, scale))
        try:
            current = float(self.root.tk.call("tk", "scaling"))
            scale = max(current, scale)
            self.root.tk.call("tk", "scaling", scale)
        except Exception:
            pass
        return scale

    def _apply_theme(self) -> None:
        style = ttk.Style(self.root)
        preferred = ("vista", "xpnative", "aqua", "default")
        available = {name.lower(): name for name in style.theme_names()}
        for name in preferred:
            key = name.lower()
            if key in available:
                style.theme_use(available[key])
                return

    def _add_path_row(
        self,
        parent: "tk.Widget",
        row: int,
        label: str,
        variable: "tk.StringVar",
        browse_cmd: Callable[[], None],
        *,
        tooltip: Optional[str] = None,
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(parent, text="Browse", command=browse_cmd)
        button.grid(row=row, column=2, sticky="ew", pady=4, padx=(4, 0))
        if tooltip:
            self._add_tooltip(label_widget, tooltip)
            self._add_tooltip(entry, tooltip)

    def _add_config_entry(
        self,
        parent: "tk.Widget",
        row: int,
        label: str,
        variable: "tk.StringVar",
        *,
        tooltip: Optional[str] = None,
    ) -> "ttk.Entry":
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if tooltip:
            self._add_tooltip(label_widget, tooltip)
            self._add_tooltip(entry, tooltip)
        return entry

    def _add_tooltip(self, widget: "tk.Widget", text: str) -> None:
        if not text:
            return
        self._tooltips.append(_Tooltip(widget, text))

    def _on_mousewheel(self, canvas: "tk.Canvas", event: "tk.Event") -> str:
        if getattr(event, "num", None) == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")
        return "break"

    def _bind_mousewheel_recursive(
        self, widget: "tk.Widget", canvas: "tk.Canvas"
    ) -> None:
        widget.bind(
            "<MouseWheel>",
            lambda event, target=canvas: self._on_mousewheel(target, event),
            add="+",
        )
        widget.bind(
            "<Button-4>",
            lambda event, target=canvas: self._on_mousewheel(target, event),
            add="+",
        )
        widget.bind(
            "<Button-5>",
            lambda event, target=canvas: self._on_mousewheel(target, event),
            add="+",
        )
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child, canvas)

    def _browse_input(self) -> None:
        self._browse_directory(self.input_var, "Select input folder")

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
        drop_path_config(cfg)
        self._apply_config_to_vars(cfg)
        self._append_log(f"Loaded config from {cfg_path}")

    def _reset_config(self) -> None:
        cfg = default_config()
        self._apply_config_to_vars(cfg)
        self._append_log("Reset config to defaults (not saved).")

    def _save_config(self) -> None:
        path = self.config_var.get().strip()
        if not path:
            path = "config.yaml"
            self.config_var.set(path)
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            self._show_message(str(exc))
            return
        try:
            self._persist_config(cfg, log=True)
        except RuntimeError as exc:
            self._show_message(str(exc))
            return

    def _persist_config(self, cfg: AppConfig, *, log: bool = False) -> None:
        path = self.config_var.get().strip()
        if not path:
            path = "config.yaml"
            self.config_var.set(path)
        cfg_path = pathlib.Path(_clean_path(path))
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        drop_path_config(cfg)
        save_config(str(cfg_path), cfg)
        if log:
            self._append_log(f"Saved config to {cfg_path}")

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
        self._cancel_event.clear()
        try:
            cfg = self._build_config()
            self._persist_config(cfg)
        except ValueError as exc:
            self._show_message(str(exc))
            return
        except RuntimeError as exc:
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

    def _start_analyze(self) -> None:
        self._start_task("analysis", self._run_analysis, clear_report=True)

    def _start_decisions(self, apply_changes: bool) -> None:
        if apply_changes and messagebox is not None:
            confirm = messagebox.askyesno(
                "Automatic Image Culling",
                "This will move files based on analysis/decisions.json. Continue?",
            )
            if not confirm:
                return
        label = "decisions (apply)" if apply_changes else "decisions (dry run)"
        self._start_task(label, self._run_decisions, target_args=(apply_changes,))

    def _request_stop(self) -> None:
        if not self.running or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.status_var.set("Stopping...")
        self.stop_btn.configure(state="disabled")
        self._append_log("Cancellation requested.")

    def _build_config(self) -> AppConfig:
        cfg = self._collect_config()
        input_value = cfg.get("input_dir")
        if not input_value:
            raise ValueError("Input folder is required.")
        if not pathlib.Path(input_value).exists():
            raise ValueError(f"Input folder not found: {input_value}")
        return cfg

    def _read_config_for_state(self) -> AppConfig:
        cfg_path = self.config_var.get().strip()
        cfg = load_config(cfg_path or None)
        drop_path_config(cfg)

        input_dir = self.input_var.get().strip()

        if input_dir:
            cfg["input_dir"] = _clean_path(input_dir)

        preview_cfg = cast(PreviewConfig, dict(cfg.get("preview") or {}))
        preview_format = self.preview_format_var.get().strip()
        if preview_format:
            preview_cfg["format"] = preview_format
        cfg["preview"] = preview_cfg

        return cfg

    @staticmethod
    def _resolve_analysis_paths(cfg: AppConfig) -> Tuple[pathlib.Path, pathlib.Path]:
        analysis_cfg = cast(AnalysisConfig, dict(cfg.get("analysis") or {}))
        input_dir = input_dir_from_cfg(cfg)
        analysis_dir = analysis_dir_for_input(input_dir)

        report_path = pathlib.Path(
            analysis_cfg.get("report_path", DEFAULT_CONFIG["analysis"]["report_path"])
        )
        if not report_path.is_absolute():
            report_path = analysis_dir / report_path

        results_path = pathlib.Path(
            analysis_cfg.get("results_path", DEFAULT_CONFIG["analysis"]["results_path"])
        )
        if not results_path.is_absolute():
            results_path = analysis_dir / results_path

        return report_path, results_path

    def _resolve_decisions_path(self, cfg: AppConfig) -> pathlib.Path:
        input_dir = input_dir_from_cfg(cfg)
        return decisions_path_for_input(input_dir)

    @staticmethod
    def _has_arw_files(directory: pathlib.Path) -> bool:
        if not directory.exists():
            return False
        for __, ___, filenames in os.walk(directory):
            for name in filenames:
                if name.lower().endswith(".arw"):
                    return True
        return False

    def _refresh_action_states(self) -> None:
        self._refresh_job = None
        if self.running:
            for button in self._action_buttons:
                button.configure(state="disabled")
            return

        cfg = self._read_config_for_state()
        input_dir = input_dir_from_cfg(cfg)

        input_exists = input_dir.exists()
        arw_exists = input_exists and self._has_arw_files(input_dir)

        report_path, results_path = self._resolve_analysis_paths(cfg)
        analysis_done = report_path.exists() or results_path.exists()

        decisions_path = self._resolve_decisions_path(cfg)
        decisions_exist = decisions_path.exists()

        self.scan_btn.configure(state="normal" if input_exists else "disabled")
        self.analyze_btn.configure(state="normal" if arw_exists else "disabled")

        decisions_state = "normal" if analysis_done and decisions_exist else "disabled"
        self.decisions_dry_btn.configure(state=decisions_state)
        self.decisions_apply_btn.configure(state=decisions_state)

        self.open_btn.configure(state="normal" if analysis_done else "disabled")

    def _run_scan(self, cfg: AppConfig) -> None:
        try:
            input_dir = input_dir_from_cfg(cfg)
            files = find_arw_files(str(input_dir), exclude_dirs=_exclude_list(cfg))
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
                if self._cancel_event.is_set():
                    self._send("log", "Discover cancelled.")
                    self._send("done", None)
                    return
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
        pool = ThreadPoolExecutor(max_workers=concurrency)
        future_map = {
            pool.submit(generate_preview, path, preview_dir, preview_cfg): path
            for path in files
        }
        cancelled = False
        try:
            done = 0
            for future in as_completed(future_map):
                if self._cancel_event.is_set():
                    cancelled = True
                    for job in future_map:
                        job.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
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
        finally:
            if not cancelled:
                pool.shutdown(wait=True)
        return missing_dep

    def _run_decisions(self, cfg: AppConfig, apply_changes: bool) -> None:
        try:
            input_dir = input_dir_from_cfg(cfg)
            if self._cancel_event.is_set():
                self._send("log", "Decisions cancelled.")
                self._send("done", None)
                return
            decisions_file = decisions_path_for_input(input_dir)
            if not decisions_file.exists():
                self._send("error", f"Decisions file not found: {decisions_file}")
                return
            self._send(
                "log",
                f"Using decisions file: {decisions_file}",
            )

            def progress_cb(current: int, total: int) -> None:
                if self._cancel_event.is_set():
                    raise _CancelError()
                self._send("progress", ("Decisions", current, total))

            summary = apply_decisions(
                decisions_file,
                cfg,
                apply=apply_changes,
                copy_files=False,
                progress_cb=progress_cb,
                log_cb=lambda msg: self._send("log", msg),
            )
            if self._cancel_event.is_set():
                self._send("log", "Decisions cancelled.")
                self._send("done", None)
                return
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
        except _CancelError:
            self._send("log", "Decisions cancelled.")
            self._send("done", None)
        except Exception as exc:
            self._send("error", str(exc))

    def _run_analysis(self, cfg: AppConfig) -> None:
        try:
            input_dir = input_dir_from_cfg(cfg)
            files = find_arw_files(str(input_dir), exclude_dirs=_exclude_list(cfg))
            if not files:
                self._send("log", "No .ARW files found.")
                self._send("done", None)
                self._send("log", "Done with analysis")
                return
            if self._cancel_event.is_set():
                self._send("log", "Analysis cancelled.")
                self._send("done", None)
                return
            total = len(files)
            self._send("log", f"Found {total} .ARW files.")

            preview_cfg = _preview_config(cfg)
            preview_dir = preview_dir_for_input(input_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            workers = max(
                1,
                int(
                    cfg.get("concurrency", DEFAULT_CONFIG["concurrency"])
                    or DEFAULT_CONFIG["concurrency"]
                ),
            )
            missing_dep = self._generate_previews(
                files, preview_dir, preview_cfg, workers, "Previews"
            )
            if self._cancel_event.is_set():
                self._send("log", "Analysis cancelled.")
                self._send("done", None)
                return
            if missing_dep:
                self._send(
                    "log",
                    "Preview generation unavailable; check rawpy/Pillow installation.",
                )

            analysis_dir = analysis_dir_for_input(input_dir)
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
                cfg,
                files,
                preview_dir,
                preview_cfg,
                progress_cb=progress_cb,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self._send("log", "Analysis cancelled.")
                self._send("done", None)
                return
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
        self.stop_btn.configure(state="normal" if running else "disabled")
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
    __ = app
    root.mainloop()


if __name__ == "__main__":
    main()
