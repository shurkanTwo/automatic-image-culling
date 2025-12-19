"""Report rendering utilities."""

import json
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence

TEMPLATE_FILE = pathlib.Path(__file__).with_name("report_template.html")
CSS_FILE = pathlib.Path(__file__).with_name("report.css")
JS_FILE = pathlib.Path(__file__).with_name("report.js")


def _write_sidecar(source: pathlib.Path, target: pathlib.Path) -> None:
    """Copy a sidecar asset when the source exists."""
    if not source.exists():
        return
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def write_html_report(
    results: Sequence[Mapping[str, Any]],
    path: pathlib.Path,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the analysis results into an HTML report."""
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    payload = {"results": results, "config": config or {}}
    rendered = template.replace(
        "__DATA_JSON__",
        json.dumps(payload, separators=(",", ":")),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")

    _write_sidecar(CSS_FILE, path.with_name(CSS_FILE.name))
    _write_sidecar(JS_FILE, path.with_name(JS_FILE.name))
