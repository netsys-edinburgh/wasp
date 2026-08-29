#!/usr/bin/env python3
"""Run and analyze the D2 single-coordinator scaling sweep."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PORT = 39100
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wasp.scaling.coordinator_scaling_analysis import (
    RunMeasurement,
    aggregate_repetitions,
    parse_metrics_jsonl,
)
from wasp.scaling.coordinator_scaling_conclusions import (
    ScalingPoint,
    assess_saturation,
)
from wasp.scaling.coordinator_scaling_execution import (
    IMAGE,
    SUBSTRATE,
    RunSpec,
    build_inner_command,
    run_command,
)


class PortAllocationError(ValueError):
    pass


def port_for_run(run_index: int) -> int:
    port = BASE_PORT + run_index
    if run_index < 0 or port > 65535:
        raise PortAllocationError(f"run index {run_index} exceeds port range")
    return port


def _failed_measurement(error: str) -> RunMeasurement:
    return RunMeasurement(
        None, None, None, None, None, None, None, None, {}, {}, error
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_run(
    run_directory: Path,
) -> tuple[RunMeasurement, int, dict[str, object]]:
    result_path = run_directory / "result.json"
    if not result_path.exists():
        error = "run result is unavailable"
        return _failed_measurement(error), 125, {"error": error}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics_path = run_directory / "metrics.jsonl"
    measurement = (
        parse_metrics_jsonl(metrics_path)
        if metrics_path.exists()
        else _failed_measurement("metrics JSONL is unavailable")
    )
    return measurement, int(result["exit_status"]), result


def _write_summary(
    output_root: Path,
    device_counts: Sequence[int],
    repetitions: int,
    image: str,
    invocation: Sequence[str],
) -> Path:
    logical_cpu_count = os.cpu_count() or 1
    rows: list[dict[str, object]] = []
    scaling_points: list[ScalingPoint] = []
    for device_count in device_counts:
        loaded = [
            _load_run(
                output_root / f"devices-{device_count}" / f"rep-{repetition}"
            )
            for repetition in range(1, repetitions + 1)
        ]
        measurements = tuple(item[0] for item in loaded)
        statuses = tuple(item[1] for item in loaded)
        aggregate = aggregate_repetitions(device_count, measurements, statuses)
        if (
            aggregate.iteration_runtime_seconds is not None
            and aggregate.median_cpu_percent is not None
            and aggregate.peak_cpu_percent is not None
        ):
            scaling_points.append(
                ScalingPoint(
                    device_count,
                    aggregate.iteration_runtime_seconds,
                    aggregate.median_cpu_percent,
                    aggregate.peak_cpu_percent,
                )
            )
        rows.append(
            {
                **dataclasses.asdict(aggregate),
                "runs": [
                    {
                        "exit_status": status,
                        "measurement": dataclasses.asdict(measurement),
                        "evidence": evidence,
                    }
                    for measurement, status, evidence in loaded
                ],
                "substrate": SUBSTRATE,
            }
        )
    saturation = assess_saturation(scaling_points, logical_cpu_count)
    summary = {
        "schema_version": 1,
        "experiment": "D2 measured single-coordinator operating envelope",
        "timestamp": _utc_now(),
        "invocation": list(invocation),
        "configuration": {
            "image": image,
            "device_counts": list(device_counts),
            "repetitions": repetitions,
            "model": "facebook/opt-125m",
            "batch_size": 1,
            "sequence_length": 32,
            "block_size": 256,
            "warmup_optimizer_steps": 1,
            "measured_optimizer_steps": 3,
            "worker_thread_count": 56,
            "metrics_nic": "lo",
            "aggregate_phase_semantics": (
                "per-repetition mean across successful repetitions; "
                "raw per-run totals retained in rows[].runs"
            ),
        },
        "environment": {
            "real_vs_emulated": "real coordinator process; emulated devices",
            "host_nic_substrate": SUBSTRATE,
            "logical_cpu_count": logical_cpu_count,
            "worker_thread_count": 56,
            "coordinator_cpu_scope": (
                "coordinator process CPU includes local non-GEMM model work "
                "and optimizer execution"
            ),
        },
        "saturation": dataclasses.asdict(saturation),
        "observed_bottleneck": saturation.bottleneck,
        "limitations": [
            "queue occupancy is unavailable in the current backend",
            "worker thread count is fixed at 56",
            "coordinator process CPU includes local non-GEMM model work and optimizer execution",
            "single host and loopback networking do not represent a physical edge network",
            "results cover the measured range through 8 local emulated devices and do not establish a host real-connection limit",
            "one or few repetitions bound repeatability and do not establish tail distributions",
        ],
        "rows": rows,
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-counts", nargs="+", type=int, default=[1, 2, 4, 8]
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/coord_scaling/single")
    )
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inside_docker = Path("/.dockerenv").exists()
    if not args.analyze_only:
        run_index = 0
        for device_count in args.device_counts:
            for repetition in range(1, args.repetitions + 1):
                relative = Path(f"devices-{device_count}") / f"rep-{repetition}"
                run = RunSpec(
                    device_count,
                    repetition,
                    port_for_run(run_index),
                    args.output_dir / relative,
                    (args.output_dir / relative)
                    if inside_docker
                    else Path("/scaling-output") / relative,
                )
                if args.dry_run:
                    print(
                        " ".join(
                            build_inner_command(run, args.metrics_interval)
                        )
                    )
                else:
                    run_command(
                        run,
                        args.output_dir,
                        args.image,
                        args.metrics_interval,
                        args.timeout_seconds,
                    )
                run_index += 1
    if args.dry_run:
        return 0
    summary_path = _write_summary(
        args.output_dir,
        args.device_counts,
        args.repetitions,
        args.image,
        sys.argv,
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
