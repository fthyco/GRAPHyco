"""Runtime QtWebEngine sandbox policies. Separate optional [gui] CI job."""
from __future__ import annotations

import os

import pytest

qt_web = pytest.importorskip("PySide6.QtWebEngineWidgets")
pytest.importorskip("PySide6.QtWebEngineCore")
from PySide6.QtCore import QUrl, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings


def test_gui_does_not_change_sandbox_environment_and_blocks_navigation():
    before = {
        key: os.environ.get(key)
        for key in ("QTWEBENGINE_DISABLE_SANDBOX", "QTWEBENGINE_CHROMIUM_FLAGS")
    }
    from graphyco.visualizer import gui

    after = {key: os.environ.get(key) for key in before}
    assert before == after
    app = QApplication.instance() or QApplication([])
    page = gui._RestrictedGraphPage()
    click = QWebEnginePage.NavigationType.NavigationTypeLinkClicked
    for url in (
        "https://example.com",
        "http://example.com",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com",
    ):
        assert not page.acceptNavigationRequest(QUrl(url), click, True)
    page.deleteLater()


def test_desktop_web_view_uses_local_resource_restrictions():
    from graphyco.visualizer.gui import VisualizerDesktopApp
    app = QApplication.instance() or QApplication([])
    window = VisualizerDesktopApp()
    if window.web_view is None:
        pytest.skip("PySide6.QtWebEngineWidgets unavailable")
    settings = window.web_view.settings()
    assert not settings.testAttribute(
        QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls
    )
    assert not settings.testAttribute(
        QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls
    )
    assert not settings.testAttribute(
        QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows
    )
    window.close()


def test_qt_html_data_embedding_stays_parsed_not_executed():
    import json
    from graphyco.visualizer.gui import VisualizerDesktopApp

    app = QApplication.instance() or QApplication([])
    window = VisualizerDesktopApp()
    if window.web_view is None:
        pytest.skip("No installed QtWebEngine")
    page = window.web_view.page()
    assert page is not None

    finished = []
    loop = QEventLoop()

    def loaded(success):
        finished.append(bool(success))
        loop.quit()

    page.loadFinished.connect(loaded)
    try:
        data = {
            "architectures": {
                "</script><script>window.injection=1</script>": {
                    "summary": {}, "nodes": {},
                    "static_graph": {"nodes": {}, "edges": []},
                }
            },
            "step_trajectories": {},
            "ablation_validation": {},
        }
        window.load_data(data)
        QTimer.singleShot(12000, loop.quit)
        loop.exec()
        assert finished == [True], "Embedded visualizer did not finish loading"
        extracted = []
        js_loop = QEventLoop()

        def received(value):
            extracted.append(value)
            js_loop.quit()

        page.runJavaScript(
            'document.getElementById("embedded-data").textContent',
            received,
        )
        QTimer.singleShot(12000, js_loop.quit)
        js_loop.exec()
        assert extracted
        assert json.loads(extracted[0]) == data

        attack = []
        check_loop = QEventLoop()

        def received_attack(value):
            attack.append(value)
            check_loop.quit()

        page.runJavaScript('window.injection === 1', received_attack)
        QTimer.singleShot(12000, check_loop.quit)
        check_loop.exec()
        assert attack == [False]
    finally:
        page.loadFinished.disconnect(loaded)
        window.close()
