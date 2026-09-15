from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union


import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from typing import Callable, Tuple
from bridge.grad_bridge import DynamicExecutionBridge, DynamicGraphState, LiveTrainingMonitor

class QueryEngine:
    """Unified query engine for computational graph topology and dynamic flow metrics.

    Provides intuitive, structured querying across architectures, nodes,
    trajectory steps, bottlenecks, invariants, and structural-functional classifications.
    """

    def __init__(
        self,
        data: Union[str, Dict[str, Any], Any],
        inputs: Optional[Any] = None,
        steps: int = 10,
        arch_name: Optional[str] = None,
    ):
        if isinstance(data, str) and (data.endswith(".json") or os.path.exists(data)):
            if not os.path.exists(data):
                pkg_dir = os.path.dirname(os.path.abspath(__file__))
                cands = [
                    os.path.join(os.getcwd(), "validation", os.path.basename(data)),
                    os.path.join(os.getcwd(), os.path.basename(data)),
                    os.path.join(pkg_dir, "..", "..", "..", "validation", os.path.basename(data)),
                    os.path.join(pkg_dir, "..", "..", "validation", os.path.basename(data)),
                    os.path.join(pkg_dir, "..", "validation", os.path.basename(data)),
                ]
                found = None
                for c in cands:
                    if os.path.exists(c):
                        found = os.path.abspath(c)
                        break
                if found:
                    data = found
                else:
                    raise FileNotFoundError(f"Benchmark JSON file not found: {data}")
            with open(data, "r", encoding="utf-8") as f:
                self._data: Dict[str, Any] = json.load(f)
        elif isinstance(data, dict):
            self._data = data
        elif hasattr(data, "to_dict"):
            # Single DynamicGraphState or similar
            state_dict = data.to_dict()
            name = arch_name or "Model"
            if "static_graph" in state_dict and "name" in state_dict.get("static_evaluation", {}):
                name = state_dict["static_evaluation"].get("name", name)
            self._data = {
                "architectures": {name: state_dict},
                "step_trajectories": {name: [state_dict]},
                "ablation_validation": {name: {}},
            }
        else:
            # Pure in-memory extraction from nn.Module, LiveTrainingMonitor, or canonical name (no JSON file)
            
            self._data = extract_benchmark_json(
                target=data,
                inputs=inputs,
                steps=steps,
                arch_name=arch_name,
                export_path=None,
            )

        self._architectures: Dict[str, Any] = self._data.get("architectures", {})
        self._trajectories: Dict[str, Any] = self._data.get("step_trajectories", {})
        self._ablation: Dict[str, Any] = self._data.get("ablation_validation", {})

        arch_list = list(self._architectures.keys())
        self._default_arch: str = arch_name if (arch_name and arch_name in self._architectures) else (arch_list[0] if arch_list else "")

    @classmethod
    def from_model(
        cls,
        model: Any,
        inputs: Optional[Any] = None,
        steps: int = 10,
        arch_name: Optional[str] = None,
    ) -> QueryEngine:
        """Constructs a QueryEngine directly from a PyTorch nn.Module without file I/O."""
        return cls(data=model, inputs=inputs, steps=steps, arch_name=arch_name)

    @property
    def data(self) -> Dict[str, Any]:
        """Returns the underlying raw benchmark dictionary."""
        return self._data

    @property
    def default_arch(self) -> str:
        """Returns the active default architecture name."""
        if self._default_arch and self._default_arch in self._architectures:
            return self._default_arch
        archs = self.list_architectures()
        return archs[0] if archs else ""

    @default_arch.setter
    def default_arch(self, name: str) -> None:
        self._default_arch = self._resolve_arch(name)

    def set_default(self, arch: str) -> str:
        """Sets and recalls the default architecture name for all future queries."""
        self.default_arch = arch
        return self.default_arch

    def list_architectures(self) -> List[str]:
        """Returns a list of all architecture names loaded in the benchmark."""
        return list(self._architectures.keys())

    def _resolve_arch(self, arch: Optional[str] = None) -> str:
        archs = self.list_architectures()
        if not archs:
            raise ValueError("No architectures loaded in QueryEngine.")
        if arch is None or arch == "":
            return self.default_arch
        if arch in self._architectures:
            return arch
        # Case-insensitive substring fallback
        for a in archs:
            if arch.lower() in a.lower():
                return a
        raise KeyError(f"Architecture '{arch}' not found. Available: {archs}")

    def describe(
        self,
        arch: Optional[str] = None,
        siblings: bool = False,
        include_description: bool = True,
    ) -> Any:
        """Returns a concise summary DataFrame similar to pandas df.describe().

        Args:
            arch: Architecture name (defaults to active default_arch).
            siblings: If True, returns a comparative table of all peer/sibling architectures side-by-side.
            include_description: If True and siblings=False, includes formal metric definitions.

        Usage:
            engine.describe()                # Single query for active model
            engine.describe(siblings=True)   # Cross-architecture comparison
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for describe(). "
                "Install it with: pip install graphyco[query]"
            ) from None

        metric_definitions = [
            ("density", "Graph Density (D)", "Edge connectivity ratio |E| / (|V|(|V|-1))"),
            ("coherence", "Flow Coherence (C)", "Conservation of incoming vs outgoing gradient flow"),
            ("topological_bottleneck", "Topological Bottleneck", "Peak node betweenness centrality in DAG"),
            ("resilience", "Structural Resilience (R)", "Path connectivity retention under single-node ablation"),
            ("grad_bottleneck_gmax", "Dynamic Bottleneck (G_max)", "Peak gradient concentration factor relative to mean"),
            ("bottleneck_node", "Bottleneck Node", "Layer exhibiting maximal gradient concentration"),
            ("spearman_act_grad", "Spearman Rank (rho_AG)", "Rank correlation between forward activations and gradients"),
            ("mean_stability_cv", "Mean Stability (CV)", "Coefficient of variation across recorded steps"),
        ]

        if siblings or (arch is not None and str(arch).lower() in ("all", "siblings")):
            archs = self.list_architectures()
            data = {}
            for a in archs:
                s = self._architectures[a].get("summary", {})
                col_vals = {}
                for key, label, _ in metric_definitions:
                    val = s.get(key)
                    col_vals[label] = round(val, 4) if isinstance(val, float) else val
                data[a] = col_vals
            df = pd.DataFrame(data)
            df.index.name = "Metric"
            return df

        target = self._resolve_arch(arch)
        s = self._architectures[target].get("summary", {})

        rows = []
        for key, label, desc in metric_definitions:
            val = s.get(key)
            formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
            item = {"Metric": label, "Value": formatted_val}
            if include_description:
                item["Description"] = desc
            rows.append(item)

        df = pd.DataFrame(rows)
        df.set_index("Metric", inplace=True)
        return df

    def get_siblings(self, node_id: str, arch: Optional[str] = None) -> List[str]:
        """Returns sibling nodes in the computational graph sharing common predecessor input(s)."""
        target = self._resolve_arch(arch)
        edges = self._architectures[target].get("edges", {})

        parents = set()
        for e in edges:
            if "->" in e:
                src, dst = e.split("->", 1)
                if dst == node_id:
                    parents.add(src)

        if not parents:
            return []

        siblings = set()
        for e in edges:
            if "->" in e:
                src, dst = e.split("->", 1)
                if src in parents and dst != node_id:
                    siblings.add(dst)

        return sorted(list(siblings))

    def get_neighbors(self, node_id: str, arch: Optional[str] = None) -> Dict[str, List[str]]:
        """Returns predecessors (inputs), successors (outputs), and siblings (co-inputs) for a node."""
        target = self._resolve_arch(arch)
        edges = self._architectures[target].get("edges", {})

        preds = set()
        succs = set()
        for e in edges:
            if "->" in e:
                src, dst = e.split("->", 1)
                if dst == node_id:
                    preds.add(src)
                if src == node_id:
                    succs.add(dst)

        siblings = set()
        for e in edges:
            if "->" in e:
                src, dst = e.split("->", 1)
                if src in preds and dst != node_id:
                    siblings.add(dst)

        return {
            "node_id": node_id,
            "predecessors": sorted(list(preds)),
            "successors": sorted(list(succs)),
            "siblings": sorted(list(siblings)),
        }

    def get_summary(self, arch: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Returns summary invariants (topological bottleneck, G_max, coherence, resilience, etc.).

        If arch is None, returns the summary for the default architecture.
        If arch is 'all', returns a dict mapping all architecture names -> summaries.
        """
        if arch == "all":
            return {
                a: self._architectures[a].get("summary", {})
                for a in self._architectures
            }
        target = self._resolve_arch(arch)
        return self._architectures[target].get("summary", {})

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Returns comparative bottleneck analysis across all loaded architectures."""
        results = []
        for name, arch_data in self._architectures.items():
            summary = arch_data.get("summary", {})
            dyn_eval = arch_data.get("dynamic_evaluation", {})
            dyn_bneck = dyn_eval.get("bottleneck", {})

            results.append({
                "architecture": name,
                "topological_bottleneck": summary.get("topological_bottleneck"),
                "gradient_bottleneck_gmax": summary.get("grad_bottleneck_gmax") or dyn_bneck.get("max_gradient_concentration"),
                "bottleneck_node": summary.get("bottleneck_node") or dyn_bneck.get("bottleneck_node"),
                "structural_resilience": summary.get("resilience"),
                "coherence": summary.get("coherence"),
            })
        return results

    def get_nodes(
        self,
        arch: Optional[str] = None,
        category: Optional[str] = None,
        has_params: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Returns node telemetry records for an architecture, with optional filtering.

        Args:
            arch: Architecture name (default: first architecture).
            category: Filter by classification: 'Type I', 'Type II', 'Type III', 'Type IV'.
            has_params: Filter by whether the node contains learnable parameters.
        """
        target = self._resolve_arch(arch)
        arch_data = self._architectures[target]
        nodes_dict = arch_data.get("nodes", {})
        dyn_eval = arch_data.get("dynamic_evaluation", {})
        categories = dyn_eval.get("correspondence", {}).get("node_categories", {})
        static_graph = arch_data.get("static_graph", {})
        static_nodes = static_graph.get("nodes", {})

        result = []
        for nid, rec in nodes_dict.items():
            cat = categories.get(nid, "Type IV")
            if category and cat.lower() != category.lower():
                continue
            p_flag = rec.get("has_parameters", False)
            if has_params is not None and p_flag != has_params:
                continue

            static_meta = static_nodes.get(nid, {})
            result.append({
                "node_id": nid,
                "name": static_meta.get("name", nid),
                "type": rec.get("node_type", "unknown"),
                "has_parameters": p_flag,
                "category": cat,
                "activation_rms": rec.get("activation", {}).get("rms") if rec.get("activation") else 0.0,
                "gradient_rms": rec.get("activation_gradient", {}).get("rms") if rec.get("activation_gradient") else 0.0,
                "ratio_grad_act": rec.get("ratio_grad_act", 0.0),
                "attributes": static_meta.get("attributes", []),
            })

        return result

    def get_node_telemetry(
        self,
        node_id: str,
        arch: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns full metrics and telemetry for a specific node at an optional step."""
        target = self._resolve_arch(arch)
        arch_data = self._architectures[target]
        nodes_dict = arch_data.get("nodes", {})
        if node_id not in nodes_dict:
            raise KeyError(f"Node '{node_id}' not found in architecture '{target}'.")

        rec = nodes_dict[node_id]
        dyn_eval = arch_data.get("dynamic_evaluation", {})
        static_eval = arch_data.get("static_evaluation", {})
        categories = dyn_eval.get("correspondence", {}).get("node_categories", {})
        temporal = dyn_eval.get("temporal", {})
        ablation = self._ablation.get(target, {})

        # Step data override if requested
        step_rec = None
        if step is not None:
            traj = self._trajectories.get(target, [])
            step_idx = max(0, min(len(traj) - 1, step - 1))
            if traj and step_idx < len(traj):
                step_nodes = traj[step_idx].get("nodes", {})
                step_rec = step_nodes.get(node_id)

        act_rms = step_rec.get("act_rms") if step_rec else (rec.get("activation", {}).get("rms") if rec.get("activation") else 0.0)
        grad_rms = step_rec.get("grad_rms") if step_rec else (rec.get("activation_gradient", {}).get("rms") if rec.get("activation_gradient") else 0.0)
        ratio_ga = step_rec.get("ratio_grad_act") if step_rec else rec.get("ratio_grad_act", 0.0)
        param_grad = step_rec.get("param_grad_rms") if step_rec else (rec.get("parameter_gradient", {}).get("rms") if rec.get("parameter_gradient") else None)

        t_stat = temporal.get(node_id, {})

        return {
            "node_id": node_id,
            "architecture": target,
            "step": step if step is not None else 1,
            "category": categories.get(node_id, "Type IV"),
            "node_type": rec.get("node_type"),
            "has_parameters": rec.get("has_parameters", False),
            "activation_rms": act_rms,
            "gradient_rms": grad_rms,
            "ratio_grad_act": ratio_ga,
            "parameter_gradient_rms": param_grad,
            "perturbation_cost": static_eval.get("perturbation_profile", {}).get(node_id, 1),
            "ablation_reachability_loss": ablation.get(node_id, 0.0),
            "temporal_cv": t_stat.get("cv", 0.0),
            "temporal_classification": t_stat.get("classification", "unknown"),
        }

    def get_step_trajectory(
        self,
        arch: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Returns step trajectory data for temporal playback."""
        target = self._resolve_arch(arch)
        traj = self._trajectories.get(target, [])
        if step is not None:
            idx = max(0, min(len(traj) - 1, step - 1))
            return traj[idx] if traj else {}
        return traj

    def get_attenuation(self, arch: Optional[str] = None) -> Dict[str, Any]:
        """Returns edge gradient attenuation and flow ratios."""
        target = self._resolve_arch(arch)
        return self._architectures[target].get("dynamic_evaluation", {}).get("attenuation", {})

    def get_anomalies(self, arch: Optional[str] = None) -> Dict[str, Any]:
        """Returns vanishing or exploding gradient anomalies detected for an architecture."""
        target = self._resolve_arch(arch)
        return self._architectures[target].get("dynamic_evaluation", {}).get("anomalies", {})

    def get_ablation(self, arch: Optional[str] = None) -> Dict[str, float]:
        """Returns node deletion reachability collapse metrics."""
        target = self._resolve_arch(arch)
        return self._ablation.get(target, {})

    def query(
        self,
        arch: Optional[str] = None,
        node: Optional[str] = None,
        metric: Optional[str] = None,
        step: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Any:
        """Universal query dispatcher.

        Examples:
            engine.query(metric='architectures')
            engine.query(metric='bottlenecks')
            engine.query(arch='ResNet Block', metric='summary')
            engine.query(arch='ResNet Block', category='Type I')
            engine.query(arch='ResNet Block', node='add', step=3)
            engine.query(arch='Sequential MLP', node='net_6', metric='gradient')
        """
        if metric in ("architectures", "archs"):
            return self.list_architectures()
        if metric == "bottlenecks":
            return self.get_bottlenecks()

        target_arch = self._resolve_arch(arch) if (arch or not node) else None

        if node:
            telemetry = self.get_node_telemetry(node, arch=target_arch, step=step)
            if metric:
                if metric in telemetry:
                    return telemetry[metric]
                # Aliases
                aliases = {
                    "grad": "gradient_rms",
                    "gradient": "gradient_rms",
                    "act": "activation_rms",
                    "activation": "activation_rms",
                    "ratio": "ratio_grad_act",
                    "param_grad": "parameter_gradient_rms",
                    "topo": "perturbation_cost",
                    "cv": "temporal_cv",
                    "type": "category",
                    "loss": "ablation_reachability_loss",
                }
                if metric in aliases:
                    return telemetry[aliases[metric]]
            return telemetry

        if category:
            return self.get_nodes(target_arch, category=category)

        if metric == "summary":
            return self.get_summary(target_arch)
        if metric == "nodes":
            return self.get_nodes(target_arch)
        if metric == "attenuation":
            return self.get_attenuation(target_arch)
        if metric == "anomalies":
            return self.get_anomalies(target_arch)
        if metric in ("trajectory", "steps"):
            return self.get_step_trajectory(target_arch, step=step)

        # Default fallback: full architecture view
        return self._architectures.get(target_arch, {})


def query_metrics(
    target: Any,
    inputs: Optional[Any] = None,
    steps: int = 10,
    arch_name: Optional[str] = None,
    metric: str = "summary",
) -> Any:
    """Queries computational graph metrics directly from an in-memory model or object without JSON.

    Usage:
        summary = query_metrics(model, inputs=x, metric="summary")
        bottlenecks = query_metrics(model, inputs=x, metric="bottlenecks")
        nodes = query_metrics(model, inputs=x, metric="nodes")
    """
    engine = QueryEngine(data=target, inputs=inputs, steps=steps, arch_name=arch_name)
    if metric in ("describe", "description"):
        return engine.describe(arch=arch_name)
    if metric in ("siblings", "comparison"):
        return engine.describe(siblings=True)
    if metric == "summary":
        return engine.get_summary(arch_name)
    if metric == "bottlenecks":
        return engine.get_bottlenecks()
    if metric == "nodes":
        return engine.get_nodes(arch_name)
    if metric == "attenuation":
        return engine.get_attenuation(arch_name)
    if metric == "ablation":
        return engine.get_ablation(arch_name)
    return engine.query(arch=arch_name, metric=metric)






def compute_ablation_map(graph_state) -> Dict[str, float]:
    """Computes node ablation reachability collapse metrics for any graph state."""
    adj: Dict[str, List[str]] = {}
    for e in graph_state.edges:
        adj.setdefault(e.source, []).append(e.target)

    input_nodes = [nid for nid in graph_state.nodes if not any(e.target == nid for e in graph_state.edges)]
    start_node = input_nodes[0] if input_nodes else (list(graph_state.nodes.keys())[0] if graph_state.nodes else "")

    if not start_node:
        return {}

    def reach(deleted: str = "") -> int:
        if start_node == deleted:
            return 0
        visited = set()
        queue = [start_node]
        while queue:
            u = queue.pop(0)
            if u in visited:
                continue
            visited.add(u)
            for v in adj.get(u, []):
                if v != deleted and v not in visited:
                    queue.append(v)
        return len(visited)

    base = reach("")
    loss_map: Dict[str, float] = {}
    for nid in graph_state.nodes:
        if nid == start_node:
            loss_map[nid] = 0.0
        else:
            rem = reach(nid)
            loss_map[nid] = float((base - rem) / base) if base > 0 else 0.0
    return loss_map


def get_canonical_models() -> Dict[str, Tuple[nn.Module, torch.Tensor]]:
    """Returns the 5 canonical benchmark architectures with default inputs."""
    from validate_framework import DenseBlock, MLP, ResNetBlock, SelfAttentionToy, UNetToy
    return {
        "Sequential MLP": (MLP(depth=4, width=64), torch.randn(8, 64)),
        "ResNet Block": (ResNetBlock(dim=64), torch.randn(8, 64)),
        "DenseBlock": (DenseBlock(dim=32), torch.randn(8, 32)),
        "U-Net Toy": (UNetToy(dim=32), torch.randn(8, 32)),
        "Self-Attention Toy": (SelfAttentionToy(dim=64), torch.randn(8, 64)),
    }


def extract_benchmark_json(
    target: Union[nn.Module, str, Dict[str, Any], DynamicExecutionBridge, LiveTrainingMonitor],
    inputs: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable] = None,
    steps: int = 10,
    arch_name: Optional[str] = None,
    export_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts and formats complete structural and dynamic graph data into visualizer JSON schema.

    Compatible with:
    - Any PyTorch nn.Module (runs multi-step trajectory evaluation and computes invariants)
    - Canonical architecture names ("Sequential MLP", "ResNet Block", "DenseBlock", "U-Net Toy", "Self-Attention Toy", or "all")
    - Active LiveTrainingMonitor or DynamicExecutionBridge instances
    - Pre-existing benchmark dictionaries

    Args:
        target: Model, bridge, monitor, canonical name, or dictionary to extract.
        inputs: Input tensor for model execution (required if target is custom nn.Module).
        loss_fn: Optional loss function (default: output.sum()).
        steps: Number of temporal trajectory steps to record (default: 10).
        arch_name: Optional display name for the architecture.
        export_path: Optional file path to write the extracted JSON.

    Returns:
        A dictionary with 'architectures', 'step_trajectories', and 'ablation_validation'.
    """
    if isinstance(target, dict) and "architectures" in target:
        benchmark_data = target
        if export_path:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
        return benchmark_data

    benchmark_data: Dict[str, Any] = {
        "architectures": {},
        "step_trajectories": {},
        "ablation_validation": {},
    }

    # Case A: Canonical name "all" -> run the 5 canonical architectures
    if isinstance(target, str) and target.lower() in ("all", "canonical"):
        canonical = get_canonical_models()
        for name, (mod, inp) in canonical.items():
            sub = extract_benchmark_json(mod, inputs=inp, steps=steps, arch_name=name)
            benchmark_data["architectures"].update(sub["architectures"])
            benchmark_data["step_trajectories"].update(sub["step_trajectories"])
            benchmark_data["ablation_validation"].update(sub["ablation_validation"])

        if export_path:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
        return benchmark_data

    # Case B: Canonical name for a single architecture
    if isinstance(target, str) and target in get_canonical_models():
        mod, default_inp = get_canonical_models()[target]
        inp = inputs if inputs is not None else default_inp
        return extract_benchmark_json(mod, inputs=inp, steps=steps, arch_name=target, export_path=export_path)

    # Case C: LiveTrainingMonitor
    if isinstance(target, LiveTrainingMonitor):
        name = arch_name or target.model.__class__.__name__
        traj = target.trajectory
        if not traj:
            raise ValueError("LiveTrainingMonitor has no recorded steps yet.")
        latest = target.get_latest_state() or traj[-1]
        static_graph = latest.graph_state.to_dict()
        static_eval = latest.static_evaluation
        dyn_eval = latest.dynamic_evaluation

        summary = _build_summary(name, static_eval, dyn_eval, latest.node_records)
        step_list = _build_step_list(traj)
        ablation = compute_ablation_map(latest.graph_state)

        benchmark_data["architectures"][name] = {
            "summary": summary,
            "static_graph": static_graph,
            "static_evaluation": static_eval,
            "dynamic_evaluation": dyn_eval,
            "nodes": {nid: rec.to_dict() for nid, rec in sorted(latest.node_records.items())},
            "edges": {f"{k[0]}->{k[1]}": stats.to_dict() for k, stats in sorted(latest.edge_records.items())},
            "attenuation": dyn_eval.get("attenuation", {}),
        }
        benchmark_data["step_trajectories"][name] = step_list
        benchmark_data["ablation_validation"][name] = ablation

        if export_path:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
        return benchmark_data

    # Case D: PyTorch nn.Module or DynamicExecutionBridge
    if isinstance(target, nn.Module) or isinstance(target, DynamicExecutionBridge):
        if isinstance(target, nn.Module):
            model = target
            bridge = DynamicExecutionBridge(model, mode="trace")
        else:
            bridge = target
            model = bridge.model

        name = arch_name or model.__class__.__name__

        if inputs is None:
            # Try to infer input shape from model or default
            inputs = torch.randn(8, 64)

        # Execute multi-step trajectory
        for _ in range(steps):
            # Vary inputs slightly per step to model real dynamic progression
            step_in = inputs + 0.05 * torch.randn_like(inputs)
            bridge.step(inputs=step_in, loss_fn=loss_fn)

        latest = bridge.trajectory[-1]
        static_eval = bridge.static_evaluation
        dyn_eval = latest.dynamic_evaluation

        summary = _build_summary(name, static_eval, dyn_eval, latest.node_records)
        step_list = _build_step_list(bridge.trajectory)
        ablation = compute_ablation_map(bridge.static_state)

        benchmark_data["architectures"][name] = {
            "summary": summary,
            "static_graph": bridge.static_state.to_dict(),
            "static_evaluation": static_eval,
            "dynamic_evaluation": dyn_eval,
            "nodes": {nid: rec.to_dict() for nid, rec in sorted(latest.node_records.items())},
            "edges": {f"{k[0]}->{k[1]}": stats.to_dict() for k, stats in sorted(latest.edge_records.items())},
            "attenuation": dyn_eval.get("attenuation", {}),
        }
        benchmark_data["step_trajectories"][name] = step_list
        benchmark_data["ablation_validation"][name] = ablation

        if export_path:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
        return benchmark_data

    raise TypeError(f"Cannot extract visualizer JSON from object of type {type(target)}")


def _build_summary(
    name: str,
    static_eval: Dict[str, Any],
    dyn_eval: Dict[str, Any],
    node_records: Dict[str, Any],
) -> Dict[str, Any]:
    """Constructs the summary invariants object required by the visualizer sidebar."""
    SCALE = 10000
    density = static_eval.get("graph_density", 0) / SCALE if isinstance(static_eval.get("graph_density"), int) else float(static_eval.get("graph_density", 0))
    coherence = static_eval.get("connectivity_coherence", 0) / SCALE if isinstance(static_eval.get("connectivity_coherence"), int) else float(static_eval.get("connectivity_coherence", 0))
    bneck = static_eval.get("topological_bottleneck_ratio", 0) / SCALE if isinstance(static_eval.get("topological_bottleneck_ratio"), int) else float(static_eval.get("topological_bottleneck_ratio", 0))
    resil = static_eval.get("structural_perturbation_resilience", 0) / SCALE if isinstance(static_eval.get("structural_perturbation_resilience"), int) else float(static_eval.get("structural_perturbation_resilience", 0))

    dyn_bneck = dyn_eval.get("bottleneck", {})
    gmax = dyn_bneck.get("max_gradient_concentration", 1.0)
    bnode = dyn_bneck.get("bottleneck_node", "-")

    align = dyn_eval.get("alignment", {})
    rho_ag = align.get("spearman_rho", 0.0)
    r_ag = align.get("pearson_r", 0.0)

    temporal = dyn_eval.get("temporal") or {}
    cv_vals = [s.get("cv", 0.0) for s in temporal.values() if isinstance(s, dict)]
    mean_cv = float(sum(cv_vals) / len(cv_vals)) if cv_vals else 0.0

    cats = (dyn_eval.get("correspondence") or {}).get("node_categories") or {}
    type_counts = {"Type I": 0, "Type II": 0, "Type III": 0, "Type IV": 0}
    for c in cats.values():
        if c in type_counts:
            type_counts[c] += 1

    act_rms_list = [r.activation.rms for r in node_records.values() if r.activation]
    grad_rms_list = [r.activation_gradient.rms for r in node_records.values() if r.activation_gradient]

    return {
        "name": name,
        "density": round(density, 4),
        "coherence": round(coherence, 4),
        "topological_bottleneck": round(bneck, 4),
        "resilience": round(resil, 4),
        "mean_act_flow": float(sum(act_rms_list) / max(1, len(act_rms_list))),
        "mean_grad_flow": float(sum(grad_rms_list) / max(1, len(grad_rms_list))),
        "grad_bottleneck_gmax": round(gmax, 4),
        "bottleneck_node": bnode,
        "spearman_act_grad": round(rho_ag, 4),
        "pearson_act_grad": round(r_ag, 4),
        "mean_stability_cv": round(mean_cv, 4),
        "type_counts": type_counts,
    }


def _build_step_list(trajectory: List[DynamicGraphState]) -> List[Dict[str, Any]]:
    """Builds step_trajectories list for multi-step temporal playback."""
    steps = []
    for s in trajectory:
        step_nodes = {}
        for nid, rec in s.node_records.items():
            step_nodes[nid] = {
                "act_rms": rec.activation.rms if rec.activation else 0.0,
                "grad_rms": rec.activation_gradient.rms if rec.activation_gradient else 0.0,
                "ratio_grad_act": rec.ratio_grad_act if rec.ratio_grad_act is not None else 0.0,
                "param_grad_rms": rec.parameter_gradient.rms if rec.parameter_gradient else None,
            }
        step_edges = {}
        for edge_tuple, stats in s.edge_records.items():
            step_edges[f"{edge_tuple[0]}->{edge_tuple[1]}"] = stats.rms

        steps.append({
            "step": s.step,
            "nodes": step_nodes,
            "edges": step_edges,
        })
    return steps


