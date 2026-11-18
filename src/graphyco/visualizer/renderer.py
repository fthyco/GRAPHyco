"""Renderer module: assembles the separated visualiser assets into deployable HTML.

Handles two modes:
- assemble(data) → complete HTML string with inlined CSS/JS/data
  (for QWebEngineView desktop app and HTTP serving)
- export(data, path) → writes self-contained HTML file to disk
  (for portable sharing — opens in any browser with no dependencies)
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Union


def _json_safe(value: Any) -> Any:
    """Map nonfinite telemetry floats to JSON null rather than invalid NaN tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class Renderer:
    """Assembles the visualiser HTML from separated static assets.

    The static assets (visualizer.html, visualizer.css, visualizer.js) live
    in the ``static/`` directory adjacent to this module. The Renderer reads
    them, inlines CSS and JS into the HTML, and injects the provided data
    dictionary into the ``<!-- DATA_PLACEHOLDER -->`` template slot.

    This replaces the previous approach of string-surgery on a monolithic HTML
    file, providing a clean, explicit template contract.
    """

    def __init__(self) -> None:
        self._static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    @property
    def static_dir(self) -> str:
        """Absolute path to the static assets directory."""
        return self._static_dir

    def _read(self, filename: str) -> str:
        """Reads a file from the static assets directory."""
        path = os.path.join(self._static_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def assemble(self, data: Optional[Union[Dict[str, Any], str]] = None) -> str:
        """Assembles a complete, self-contained HTML string.

        CSS and JS are inlined directly into the HTML. The provided data
        (dictionary or JSON string) is injected into the embedded-data
        script tag.

        Args:
            data: Benchmark data dictionary, JSON string, or None.
                  When None, the resulting HTML will load data via file
                  upload or fetch fallback.

        Returns:
            Complete HTML string ready for QWebEngineView.setHtml() or
            writing to disk.
        """
        html = self._read("visualizer.html")
        css = self._read("visualizer.css")
        js = self._read("visualizer.js")

        # Inline CSS: replace placeholder with <style> block and remove external link
        html = html.replace(
            "<!-- INLINE_CSS_PLACEHOLDER -->",
            f"<style>\n{css}\n</style>",
        )
        html = html.replace('<link rel="stylesheet" href="visualizer.css">\n', "")

        # Inline JS: replace placeholder with <script> block and remove external script
        html = html.replace(
            "<!-- INLINE_JS_PLACEHOLDER -->",
            f"<script>\n{js}\n</script>",
        )
        html = html.replace('<script src="visualizer.js"></script>\n', "")

        # The embedding is inside a <script type="application/json"> tag.
        # A JSON string containing </script> still terminates that HTML
        # element even though it would be valid JSON. Escaping raw HTML
        # delimiters prevents data from becoming executable markup.
        if data is None:
            data_json = ""
        else:
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                raise TypeError("Embedded benchmark data must be a JSON object")
            data_json = json.dumps(
                _json_safe(data), ensure_ascii=True, allow_nan=False
            )
            data_json = (
                data_json.replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
            )

        html = html.replace("<!-- DATA_PLACEHOLDER -->", data_json)

        return html

    def export(self, data: Union[Dict[str, Any], str], path: str) -> str:
        """Writes a self-contained HTML file for portable sharing.

        The exported file contains all CSS, JS, and data inlined — it can
        be opened in any browser with zero dependencies.

        Args:
            data: Benchmark data dictionary or JSON string.
            path: Output file path.

        Returns:
            The absolute path of the written file.
        """
        content = self.assemble(data)
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return abs_path
