from __future__ import annotations

import os
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu --disable-gpu-compositing")


import json
import os
import sys
from typing import Any, Dict, Optional, Union


import torch
from typing import Callable, Any, Tuple
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QTextEdit,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:
    HAS_WEBENGINE = False

from .diagnostics import GradientHealthStatus, LiveTrainingDiagnostics
from .query import QueryEngine
from .renderer import Renderer


class VisualizerDesktopApp(QMainWindow):
    """Standalone native desktop application for computational graph visualization
    and real-time PyTorch training loop diagnostics.
    """

    def __init__(
        self,
        data: Optional[Union[str, Dict[str, Any], QueryEngine]] = None,
        diagnostics: Optional[LiveTrainingDiagnostics] = None,
    ):
        super().__init__()
        self.setWindowTitle("Graphyco — Computational Graph Flow & Live Diagnostics")
        self.resize(1280, 850)

        self.diagnostics = diagnostics
        self.query_engine: Optional[QueryEngine] = None
        self._renderer = Renderer()

        # Auto-build graph flow for live training if dummy_input is provided
        if data is None and diagnostics is not None and getattr(diagnostics, "dummy_input", None) is not None:
            from .query import QueryEngine
            data = QueryEngine(diagnostics.model, inputs=diagnostics.dummy_input, steps=1)

        if data is not None:
            self.load_data(data)
        elif self.diagnostics is None:
            cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynamic_flow_benchmark.json")
            if os.path.exists(cand):
                self.load_data(cand)

        self._setup_ui()
        self._apply_theme()
        self._setup_menu()

        # Timer for live training loop polling
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(200)
        self.live_timer.timeout.connect(self._poll_live_diagnostics)
        if self.diagnostics is not None:
            self.live_timer.start()

    def load_data(self, data: Union[str, Dict[str, Any], QueryEngine]) -> None:
        """Loads or updates benchmark data in the application."""
        if isinstance(data, QueryEngine):
            self.query_engine = data
        else:
            self.query_engine = QueryEngine(data)

        if hasattr(self, "web_view") and self.web_view is not None:
            self._render_graph_view()
        if hasattr(self, "invariants_table"):
            self._populate_invariants_table()

    def _render_graph_view(self) -> None:
        """Renders graph via Renderer assembly — no string surgery or external file dependency."""
        if not HAS_WEBENGINE or self.web_view is None:
            return

        data_dict = self.query_engine.data if self.query_engine else {}
        html_content = self._renderer.assemble(data_dict)
        base_url = QUrl.fromLocalFile(self._renderer.static_dir + "/")
        self.web_view.setHtml(html_content, base_url)



    def _apply_theme(self) -> None:
        style = """
            QMainWindow, QDialog, QTabWidget::pane {
                background-color: #ffffff;
                color: #111827;
                border: none;
            }
            QWidget {
                font-family: "Ubuntu", sans-serif;
                font-size: 13px;
                color: #111827;
                background-color: transparent;
            }
            
            /* Segmented Button Style for Tabs */
            QTabBar {
                alignment: center;
            }
            QTabBar::tab {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                padding: 6px 14px;
                color: #4b5563;
                font-size: 13px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            QTabBar::tab:first {
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
            }
            QTabBar::tab:last {
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QTabBar::tab:!first {
                margin-left: -1px;
            }
            QTabBar::tab:hover:!selected {
                background: #f3f4f6;
            }
            QTabBar::tab:selected {
                background: #111827;
                color: #ffffff;
                font-weight: bold;
            }

            /* Booktabs Table Style */
            QTableWidget, QTableView {
                background-color: #ffffff;
                border-top: 2px solid #111827;
                border-bottom: 2px solid #111827;
                border-left: none;
                border-right: none;
                gridline-color: transparent;
                selection-background-color: #f3f4f6;
                selection-color: #111827;
            }
            QHeaderView::section {
                background-color: #ffffff;
                color: #111827;
                font-family: "Ubuntu", sans-serif;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #111827;
                padding: 6px 8px;
            }
            QTableWidget::item {
                border-bottom: 1px solid #f3f4f6;
                padding: 5px 8px;
            }

            /* Text Editors / Alerts */
            QTextEdit, QPlainTextEdit {
                background-color: #fafafa;
                border: 1px solid #e5e7eb;
                color: #111827;
                font-family: "Ubuntu Mono", "Ubuntu", monospace;
                padding: 8px;
                border-radius: 4px;
            }
            
            QSplitter::handle {
                background-color: #e5e7eb;
            }
            QStatusBar {
                background-color: #ffffff;
                color: #4b5563;
                border-top: 1px solid #e5e7eb;
            }
        """
        self.setStyleSheet(style)

    def _setup_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget(self)
        self.tabs.setFont(QFont("Ubuntu", 11))
        main_layout.addWidget(self.tabs)

        # ── Tab 1: Interactive Graph Flow View ──
        tab_graph = QWidget()
        layout_graph = QVBoxLayout(tab_graph)
        layout_graph.setContentsMargins(0, 0, 0, 0)

        if HAS_WEBENGINE:
            self.web_view = QWebEngineView(tab_graph)
            layout_graph.addWidget(self.web_view)
            self._render_graph_view()
        else:
            lbl = QLabel("QtWebEngine is not available. Native tables are loaded in other tabs.")
            lbl.setAlignment(Qt.AlignCenter)
            layout_graph.addWidget(lbl)
            self.web_view = None

        if self.query_engine is not None or self.diagnostics is None:
            self.tabs.addTab(tab_graph, "Graph Flow && Telemetry")

        # ── Tab 2: Live Training Diagnostics ──
        tab_diag = QWidget()
        layout_diag = QVBoxLayout(tab_diag)

        # Header Cards
        card_box = QHBoxLayout()

        self.card_step = QLabel("Step: 0")
        self.card_step.setStyleSheet("background: #fafafa; border: 1px solid #e5e7eb; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        card_box.addWidget(self.card_step)

        self.card_loss = QLabel("Loss: N/A")
        self.card_loss.setStyleSheet("background: #fafafa; border: 1px solid #e5e7eb; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        card_box.addWidget(self.card_loss)

        self.card_health = QLabel("Health: WAITING")
        self.card_health.setStyleSheet("background: #fafafa; border: 1px solid #e5e7eb; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        card_box.addWidget(self.card_health)

        self.card_bneck = QLabel("Dynamic Bottleneck: None")
        self.card_bneck.setStyleSheet("background: #fafafa; border: 1px solid #e5e7eb; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        card_box.addWidget(self.card_bneck)

        layout_diag.addLayout(card_box)

        # Splitter: Layer table and Alert feed
        splitter = QSplitter(Qt.Vertical)

        # Layer Telemetry Table
        self.live_layer_table = QTableWidget()
        self.live_layer_table.setShowGrid(False)
        self.live_layer_table.setColumnCount(4)
        self.live_layer_table.setHorizontalHeaderLabels(["Layer Name", "Latest Grad RMS", "Latest Act RMS", "Dead Neuron Ratio"])
        self.live_layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self.live_layer_table)

        # Alerts and Warning Log
        alert_box = QWidget()
        alert_layout = QVBoxLayout(alert_box)
        alert_layout.setContentsMargins(0, 0, 0, 0)
        lbl_alerts = QLabel("Real-time Anomaly & Invariant Alerts:")
        lbl_alerts.setStyleSheet("font-weight: bold; color: #4b5563; margin-top: 4px;")
        alert_layout.addWidget(lbl_alerts)

        self.alert_feed = QTextEdit()
        self.alert_feed.setReadOnly(True)
        self.alert_feed.setStyleSheet("background: #fafafa; color: #111827; font-family: ui-monospace, \"Cascadia Code\", monospace; font-size: 13px; border: 1px solid #e5e7eb; border-radius: 4px; padding: 6px;")
        alert_layout.addWidget(self.alert_feed)

        splitter.addWidget(alert_box)
        splitter.setSizes([450, 200])
        layout_diag.addWidget(splitter)

        if self.diagnostics is not None:
            self.tabs.addTab(tab_diag, "Live Training Diagnostics")
            self.tabs.setCurrentWidget(tab_diag)

        # ── Tab 3: Invariant & Bottleneck Inspector ──
        tab_inv = QWidget()
        layout_inv = QVBoxLayout(tab_inv)

        lbl_inv_title = QLabel("Topological and Dynamic Graph Invariants:")
        lbl_inv_title.setStyleSheet("font-weight: bold; font-size: 16px; font-family: \"Charter\", \"STIX Two Text\", serif; color: #111827; margin-bottom: 6px;")
        layout_inv.addWidget(lbl_inv_title)

        self.invariants_table = QTableWidget()
        self.invariants_table.setShowGrid(False)
        self.invariants_table.setColumnCount(3)
        self.invariants_table.setHorizontalHeaderLabels(["Invariant / Metric", "Computed Value", "Theoretical Interpretation"])
        self.invariants_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.invariants_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.invariants_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout_inv.addWidget(self.invariants_table)

        self._populate_invariants_table()
        if self.query_engine is not None or self.diagnostics is None:
            self.tabs.addTab(tab_inv, "Invariant && Bottleneck Inspector")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Desktop Visualizer Ready.")

    def _setup_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Benchmark JSON...", self)
        open_action.triggered.connect(self._open_json_dialog)
        file_menu.addAction(open_action)

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        refresh_action = QAction("&Reload Graph View", self)
        refresh_action.triggered.connect(self._render_graph_view)
        view_menu.addAction(refresh_action)

    def _open_json_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Computational Graph Benchmark JSON",
            os.getcwd(),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            try:
                self.load_data(path)
                self.status_bar.showMessage(f"Loaded benchmark: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load JSON:\n{e}")

    def _populate_invariants_table(self) -> None:
        if self.query_engine is None:
            return

        archs = self.query_engine.list_architectures()
        if not archs:
            return

        arch = archs[0]
        s = self.query_engine.get_summary(arch)

        invariants_data = [
            ("Target Architecture", str(s.get("name", arch)), "Primary computational graph model"),
            ("Graph Density (D)", f"{s.get('density', 0.0):.4f}", "|E| / (|V|(|V|-1)) connectivity ratio"),
            ("Connectivity Coherence (C)", f"{s.get('coherence', 0.0):.4f}", "Conservation of incoming vs outgoing gradient flow"),
            ("Topological Bottleneck", f"{s.get('topological_bottleneck', 0.0):.4f}", "Peak betweenness centrality across DAG nodes"),
            ("Structural Resilience (R)", f"{s.get('resilience', 0.0):.4f}", "Fraction of downstream paths preserved under ablation"),
            ("Dynamic Bottleneck (G_max)", f"{s.get('grad_bottleneck_gmax', 0.0):.4f}", "Peak gradient accumulation concentration factor"),
            ("Primary Bottleneck Node", str(s.get("bottleneck_node", "None")), "Layer exhibiting maximal gradient accumulation"),
            ("Spearman Rank (rho_AG)", f"{s.get('spearman_act_grad', 0.0):.4f}", "Rank correlation between activations and gradients"),
            ("Stability CV", f"{s.get('mean_stability_cv', 0.0):.4f}", "Temporal coefficient of variation across recorded steps"),
        ]

        self.invariants_table.setRowCount(len(invariants_data))
        for row, (k, v, desc) in enumerate(invariants_data):
            self.invariants_table.setItem(row, 0, QTableWidgetItem(k))
            self.invariants_table.setItem(row, 1, QTableWidgetItem(v))
            self.invariants_table.setItem(row, 2, QTableWidgetItem(desc))

    def _poll_live_diagnostics(self) -> None:
        if self.diagnostics is None:
            return

        snapshot = self.diagnostics.get_snapshot()
        if not snapshot:
            return

        step = snapshot["step"]
        loss_hist_len = len(snapshot.get("loss_history", []))
        current_state = (step, loss_hist_len)
        if getattr(self, "_last_rendered_state", None) == current_state:
            return
        self._last_rendered_state = current_state
        loss = snapshot["latest_loss"]
        health = snapshot["health_status"]
        bneck = snapshot["active_bottleneck_layer"]
        gmax = snapshot["gmax"]

        self.card_step.setText(f"Step: {step}")
        self.card_loss.setText(f"Loss: {loss:.4f}" if loss is not None else "Loss: N/A")

        if health == GradientHealthStatus.HEALTHY:
            self.card_health.setText("Health: HEALTHY")
            self.card_health.setStyleSheet("background: #fafafa; border: 1px solid #e5e7eb; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        elif "WARNING" in health:
            self.card_health.setText(f"Health: {health}")
            self.card_health.setStyleSheet("background: #ffffff; border: 1px solid #111827; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")
        else:
            self.card_health.setText(f"Health: {health}")
            self.card_health.setStyleSheet("background: #ffffff; border: 1px solid #111827; color: #111827; font-size: 15px; font-weight: bold; padding: 10px; border-radius: 6px;")

        if bneck:
            self.card_bneck.setText(f"Bottleneck: {bneck} ({gmax:.2f}x)")
        else:
            self.card_bneck.setText("Bottleneck: None")

        # Update layer table
        layers = snapshot.get("layers", {})
        self.live_layer_table.setRowCount(len(layers))
        for row, (name, stats) in enumerate(sorted(layers.items())):
            self.live_layer_table.setItem(row, 0, QTableWidgetItem(name))
            self.live_layer_table.setItem(row, 1, QTableWidgetItem(f"{stats.get('latest_grad_rms', 0.0):.6f}"))
            self.live_layer_table.setItem(row, 2, QTableWidgetItem(f"{stats.get('latest_act_rms', 0.0):.6f}"))
            dead_ratio = stats.get("dead_neuron_ratio", 0.0)
            item_dead = QTableWidgetItem(f"{dead_ratio * 100:.1f}%")
            if dead_ratio > 0.5:
                item_dead.setForeground(QColor("#111827")) # Removed color
            self.live_layer_table.setItem(row, 3, item_dead)

        # Update alerts feed
        alerts = snapshot.get("alerts", [])
        if alerts:
            lines = [
                f"[{a['timestamp']}] Step {a['step']} [{a['level']}]: {a['message']}"
                for a in alerts[-15:]
            ]
            self.alert_feed.setPlainText("\n".join(lines))

        # ── Push live telemetry to Graph Flow renderer ──
        if self.web_view is not None and self.web_view.page():
            node_records = {}
            for name, stats in layers.items():
                fx_name = name.replace(".", "_")
                grad = stats.get("latest_grad_rms", 0.0)
                act = stats.get("latest_act_rms", 0.0)
                rec = {
                    "grad_rms": grad,
                    "act_rms": act,
                    "ratio_grad_act": grad / (act + 1e-8),
                    "param_grad_rms": None,
                }
                node_records[fx_name] = rec
                if fx_name and fx_name[0].isdigit():
                    node_records[f"_{fx_name}"] = rec

            live_summary = {
                "grad_bottleneck_gmax": gmax,
                "bottleneck_node": bneck or "-",
            }

            loss_js = f"{loss:.6f}" if loss is not None else "null"
            js_cmd = (
                f"if (typeof window.injectLiveTelemetry === 'function') {{"
                f"window.injectLiveTelemetry("
                f"{step}, {loss_js}, "
                f"{json.dumps(node_records)}, "
                f"{json.dumps(live_summary)});"
                f"}}"
            )
            self.web_view.page().runJavaScript(js_cmd)


def launch_desktop_app(
    data: Optional[Union[str, Dict[str, Any], QueryEngine]] = None,
    diagnostics: Optional[LiveTrainingDiagnostics] = None,
) -> VisualizerDesktopApp:
    """Launches the standalone native desktop GUI application.

    Args:
        data: JSON filepath, benchmark dictionary, or QueryEngine.
        diagnostics: Optional active LiveTrainingDiagnostics instance.

    Returns:
        The initialized VisualizerDesktopApp window.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = VisualizerDesktopApp(data=data, diagnostics=diagnostics)
    window.show()
    return window
def launch_app(
    data: Union[str, Dict[str, Any], QueryEngine, None] = None,
    open_browser: bool = True,
    **kwargs
) -> None:
    """Launches the interactive visualization dashboard.

    Args:
        data: JSON filepath, data dictionary, QueryEngine, or None.
        open_browser: If True, opens dashboard.
    """
    if data is None:
        cand1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynamic_flow_benchmark.json")
        cand2 = os.path.join(os.getcwd(), "dynamic_flow_benchmark.json")
        cand3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "validation", "dynamic_flow_benchmark.json")
        cand4 = os.path.join(os.getcwd(), "validation", "dynamic_flow_benchmark.json")
        for cand in (cand3, cand4, cand1, cand2):
            if os.path.exists(cand):
                data = os.path.abspath(cand)
                break
        if data is None:
            # Build canonical benchmark on the fly
            print("[Visualizer] dynamic_flow_benchmark.json not found. Extracting canonical benchmark data...")
            data = extract_benchmark_json("all")

    if open_browser:
        try:
            
            from PySide6.QtWidgets import QApplication
            import sys
            app = QApplication.instance() or QApplication(sys.argv)
            window = launch_desktop_app(data=data)
            app.exec()
        except ImportError:
            import tempfile, webbrowser
            from .renderer import Renderer
            if isinstance(data, str) and data.endswith(".json"):
                with open(data, "r", encoding="utf-8") as f:
                    data_dict = json.load(f)
            elif isinstance(data, QueryEngine):
                data_dict = data.data
            else:
                data_dict = data
            html = Renderer().assemble(data_dict)
            fd, path = tempfile.mkstemp(suffix=".html")
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(html)
            webbrowser.open(f"file://{path}")


def visualize(
    target: Any = None,
    inputs: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable] = None,
    steps: int = 10,
    open_browser: bool = True,
    export_json: Optional[str] = None,
    export_html: Optional[str] = None,
    **kwargs
) -> None:
    """Unified high-level entrypoint to profile, extract, and visualize any PyTorch model or benchmark.

    Usage Examples:
        # 1. Visualize precomputed benchmark
        visualize("dynamic_flow_benchmark.json")

        # 2. Visualize any custom PyTorch model with 1 line of code
        visualize(my_model, inputs=torch.randn(8, 64), steps=5)

        # 3. Visualize canonical architectures
        visualize("ResNet Block")

        # 4. Export self-contained HTML for portable sharing
        visualize(my_model, inputs=x, export_html="report.html")
    """
    # Case 1: Target is a JSON file or existing benchmark dictionary
    if isinstance(target, str) and target.endswith(".json"):
        if export_html:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            from .renderer import Renderer
            Renderer().export(data, export_html)
        return launch_app(data=target, open_browser=open_browser)

    if isinstance(target, dict) and "architectures" in target:
        if export_html:
            from .renderer import Renderer
            Renderer().export(target, export_html)
        return launch_app(data=target, open_browser=open_browser)

    # Case 2: Target is a PyTorch model, canonical name, or monitor
    if target is not None:
        data = extract_benchmark_json(
            target=target,
            inputs=inputs,
            loss_fn=loss_fn,
            steps=steps,
            export_path=export_json,
        )
        if export_html:
            from .renderer import Renderer
            Renderer().export(data, export_html)
        return launch_app(data=data, open_browser=open_browser)

    # Case 3: target is None -> default launch
    return launch_app(data=None, open_browser=open_browser)

def visualize_inline(target: Any = None, inputs: Optional[torch.Tensor] = None, steps: int = 10, **kwargs):
    """Displays the benchmark inline in a Jupyter notebook."""
    data = extract_benchmark_json(target, inputs=inputs, steps=steps)
    from .renderer import Renderer
    html = Renderer().assemble(data)
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except ImportError:
        print("IPython is not available. Use visualize() for desktop app.")

