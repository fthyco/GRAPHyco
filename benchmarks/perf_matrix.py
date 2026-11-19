"""Repeatable CPU performance matrix, measured in isolated processes.

This is a relative regression guard, not a latency SLA or GPU measurement.
Each repetition executes the original calibrated monitor_overhead.py cases.
Report timings, peak process RSS, versions, and all raw independent samples.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


CASES = (
    "trace",
    "module",
    "trace_device",
    "module_device",
    "trace_sampled",
    "module_sampled",
)


def measure(width: int, steps: int, warmup: int) -> dict:
    script = Path(__file__).with_name("monitor_overhead.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            str(steps),
            "--warmup",
            str(warmup),
            "--width",
            str(width),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=480,
    )
    payload = json.loads(result.stdout)
    assert set(CASES) == set(payload["overhead_factor"])
    assert set(CASES) | {"plain"} == set(payload["results"])
    for case, row in payload["results"].items():
        assert row["mode"] == case
        assert row["mean_step_ms"] > 0.0, (case, row)
        assert row["peak_process_rss_mib"] > 0.0, (case, row)
        if case == "plain":
            assert row["retained_steps"] == 0
        else:
            assert 0 < row["retained_steps"] <= min(steps + warmup, 32)
    return payload


def build_matrix(
    widths: List[int],
    repetitions: int,
    steps: int,
    warmup: int,
    catastrophic_ratio: float,
) -> dict:
    raw: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}
    violations: List[str] = []
    for width in widths:
        raw[str(width)] = [measure(width, steps, warmup) for _ in range(repetitions)]
        samples = raw[str(width)]
        timings: Dict[str, dict] = {}
        for case in ("plain",) + CASES:
            values = [sample["results"][case]["mean_step_ms"] for sample in samples]
            rss = [
                sample["results"][case]["peak_process_rss_mib"]
                for sample in samples
            ]
            ratios = (
                [1.0] * repetitions
                if case == "plain"
                else [sample["overhead_factor"][case] for sample in samples]
            )
            median_ratio = float(statistics.median(ratios))
            timings[case] = {
                "median_step_ms": float(statistics.median(values)),
                "min_step_ms": min(values),
                "max_step_ms": max(values),
                "median_relative_overhead": median_ratio,
                "median_peak_process_rss_mib": float(statistics.median(rss)),
                "all_step_ms": values,
                "all_overhead_factors": ratios,
            }
            if case != "plain" and median_ratio > catastrophic_ratio:
                violations.append(
                    "width={} case={} median overhead {:.2f}x exceeds {:.2f}x".format(
                        width, case, median_ratio, catastrophic_ratio
                    )
                )
        summary[str(width)] = timings

    return {
        "method": "isolated-process CPU median wall-time benchmark",
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "parameters": {
            "widths": widths,
            "repetitions": repetitions,
            "steps": steps,
            "warmup": warmup,
            "catastrophic_ratio": catastrophic_ratio,
        },
        "summary": summary,
        "raw": raw,
        "violations": violations,
        "interpretation": (
            "Relative timings from this one runner only; not an SLA. "
            "Peak RSS includes interpreter and PyTorch allocations. "
            "Initial FX tracing excluded, CPU only. "
            "The catastrophic guard detects only gross regressions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--catastrophic-ratio", type=float, default=80.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.widths
        or min(args.widths) < 2
        or min(args.steps, args.warmup, args.repetitions) < 1
        or args.catastrophic_ratio <= 1.0
    ):
        parser.error("invalid widths, steps, warmup, repetitions or ratio")
    report = build_matrix(
        args.widths,
        args.repetitions,
        args.steps,
        args.warmup,
        args.catastrophic_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for width in args.widths:
        print("width={}".format(width))
        for case, item in report["summary"][str(width)].items():
            print(
                "  {:<16} {:>10.4f} ms  {:>7.2f}x   peak RSS {:>8.1f} MiB".format(
                    case,
                    item["median_step_ms"],
                    item["median_relative_overhead"],
                    item["median_peak_process_rss_mib"],
                )
            )
    if report["violations"]:
        raise SystemExit("PERF GATE FAILED: " + "; ".join(report["violations"]))
    print("PERF GATE PASSED: catastrophic regression guard only")


if __name__ == "__main__":
    main()
