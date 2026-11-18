"""Adversarial JSON embedding and distribution/security regression tests."""
from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from graphyco.visualizer.renderer import Renderer


class _ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self._script = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script":
            self._script = {"attrs": dict(attrs), "content": ""}
            self.scripts.append(self._script)

    def handle_data(self, data):
        if self._script is not None:
            self._script["content"] += data

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self._script = None


@pytest.mark.parametrize("attack", [
    "</script><script>window.GRAPHYCO_HACKED=true</script>",
    "</ScRiPt><img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    "<!--\u2028--!>\u2029<script>alert(1)</script>",
    "</script>\u2028\u2029&<>",
])
def test_renderer_script_escape_roundtrip(attack):
    data = {
        "architectures": {
            attack: {
                "summary": {
                    "bottleneck_node": attack,
                    "metadata": ["safe", attack, {"code": attack}],
                }
            }
        }
    }
    html = Renderer().assemble(data)
    embedded = re.search(
        r'<script id="embedded-data" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    assert embedded is not None
    assert json.loads(embedded.group(1)) == data
    assert "<script>window.GRAPHYCO_HACKED" not in html
    assert "<img src=x onerror" not in html
    assert "<svg/onload" not in html
    assert "\\u003c" in embedded.group(1)

    parser = _ScriptCollector()
    parser.feed(html)
    assert len(parser.scripts) == 2
    assert parser.scripts[0]["attrs"]["type"] == "application/json"
    assert json.loads(parser.scripts[0]["content"]) == data


def test_renderer_treats_json_string_as_data_not_html():
    data = {"architectures": {"</script><script>alert(1)</script>": {}}}
    output = Renderer().assemble(json.dumps(data))
    assert json.loads(
        re.search(
            r'<script id="embedded-data" type="application/json">(.*?)</script>',
            output, re.DOTALL
        ).group(1)
    ) == data

    with pytest.raises(json.JSONDecodeError):
        Renderer().assemble("</script><img src=x onerror=alert(1)>")
    with pytest.raises(TypeError):
        Renderer().assemble(json.dumps([1, 2]))
    with pytest.raises(TypeError):
        Renderer().assemble(17)


def test_portable_export_preserves_safe_embedding(tmp_path):
    path = tmp_path / "file.html"
    data = {"architectures": {"<script>test</script>": {}}}
    filename = Renderer().export(data, str(path))
    assert filename == str(path)
    assert json.loads(
        re.search(
            r'<script id="embedded-data" type="application/json">(.*?)</script>',
            path.read_text(), re.DOTALL,
        ).group(1)
    ) == data


def test_gui_does_not_disable_webengine_sandbox():
    # The core/headless install must remain free of PySide6.
    import graphyco.visualizer as visualizer

    path = Path(visualizer.__file__).parent / "gui.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert "QTWEBENGINE_DISABLE_SANDBOX" not in text.split("import json", 1)[0]
    assert 'os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX"' not in text
    assert '--no-sandbox' not in text.split("import json", 1)[0]
    assert "LocalContentCanAccessRemoteUrls" in text
    assert "LocalContentCanAccessFileUrls" in text
    assert "JavascriptCanOpenWindows" in text
    assert "acceptNavigationRequest" in text


def test_embedded_json_displays_literal_unsafe_node_label():
    data = {"architectures": {"<img src=x onerror=alert(1)>": {}}}
    html = Renderer().assemble(data)
    assert "<img src=x onerror=alert(1)>" not in html
    assert "createElement('option')" in html
    assert "opt.textContent = name" in html
