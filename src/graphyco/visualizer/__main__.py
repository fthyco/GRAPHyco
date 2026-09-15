from __future__ import annotations

import argparse
import json
import sys

from .query import extract_benchmark_json\nfrom .gui import launch_app, visualize
from .query import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graphyco.visualizer",
        description="Graphyco: Computational Graph Flow Visualizer & Query Engine",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to dynamic_flow_benchmark.json (default: search in workspace).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlp", "resnet", "dense", "unet", "attention", "all"],
        help="Profile and visualize one or all canonical PyTorch architectures.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps to profile for model trajectory (default: 10).",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export benchmark data to JSON file before launching.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="CLI query without starting web server (e.g., 'bottlenecks', 'summary', 'nodes').",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Architecture filter for CLI query.",
    )

    args = parser.parse_args()

    model_map = {
        "mlp": "Sequential MLP",
        "resnet": "ResNet Block",
        "dense": "DenseBlock",
        "unet": "U-Net Toy",
        "attention": "Self-Attention Toy",
        "all": "all",
    }

    target = None
    if args.model:
        target = model_map[args.model]
    elif args.json:
        target = args.json

    if args.query:
        # CLI Query Mode
        if target:
            if isinstance(target, str) and target.endswith(".json"):
                qe = QueryEngine(target)
            else:
                data = extract_benchmark_json(target, steps=args.steps, export_path=args.export)
                qe = QueryEngine(data)
        else:
            # Default JSON
            qe = QueryEngine("dynamic_flow_benchmark.json")

        result = qe.query(arch=args.arch, metric=args.query)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Launch Desktop GUI
    from .gui import launch_desktop_app
    from PySide6.QtWidgets import QApplication

    if target and not (isinstance(target, str) and target.endswith(".json")):
        data = extract_benchmark_json(target, steps=args.steps, export_path=args.export)
    else:
        data = target

    app = QApplication.instance() or QApplication(sys.argv)
    window = launch_desktop_app(data=data)
    print("[Visualizer] Running native desktop application window...")
    sys.exit(app.exec())

    


if __name__ == "__main__":
    main()

