import json
import pathlib
from typing import Dict, List


TEMPLATE_FILE = pathlib.Path(__file__).with_name("report_template.html")
CSS_FILE = pathlib.Path(__file__).with_name("report.css")


def write_html_report(results: List[Dict], path: pathlib.Path, config: Dict = None) -> None:
    """
    Render the analysis results into an HTML report using an external template and CSS.
    """
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    payload = {"results": results, "config": config or {}}
    rendered = template.replace("__DATA_JSON__", json.dumps(payload))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")

    if CSS_FILE.exists():
        target_css = path.with_name(CSS_FILE.name)
        target_css.write_text(CSS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
