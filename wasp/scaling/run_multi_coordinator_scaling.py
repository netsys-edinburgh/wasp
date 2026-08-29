#!/usr/bin/env python3
"""Execute D5-D7 strong/weak scaling and emit provenance-rich summaries."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from wasp.coordinator.multi_coordinator_rank_process import RankProcess
from wasp.coordinator.multi_coordinator_rank_process import (
    wait_for_rank_processes as _wait_for_rank_processes,
)
from wasp.scaling.multi_coordinator_scaling_cli import parse_scaling_cli
from wasp.scaling.multi_coordinator_scaling_conclusions import (
    build_scaling_conclusion,
)
from wasp.scaling.multi_coordinator_scaling_config import (
    AffinityMode,
    build_rank_launch,
    select_affinity_mode,
)
from wasp.scaling.multi_coordinator_scaling_execution import (
    ExperimentRun,
    apply_affinity_mode,
    build_experiment_plan,
)
from wasp.scaling.multi_coordinator_scaling_results import (
    GlobalResult,
    LossCorrectness,
    RankResult,
    build_global_result,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_binding_available() -> bool:
    if shutil.which("numactl") is None:
        return False
    probe = subprocess.run(
        ("numactl", "--cpunodebind=0", "--membind=0", "true"),
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _load_rank_result(path: Path) -> RankResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    correctness = payload["loss_correctness"]
    phases = payload["phases_seconds"]
    return RankResult(
        rank=int(payload["rank"]),
        warmup_iteration_seconds=float(payload["warmup_iteration_seconds"]),
        iteration_total_seconds=float(phases["iteration_total"]),
        device_dispatch_aggregation_seconds=float(
            phases["device_dispatch_aggregation"]
        ),
        gradient_sync_seconds=float(phases["gradient_sync"]),
        optimizer_seconds=float(phases["optimizer"]),
        idle_other_seconds=float(phases["idle_other"]),
        throughput_samples_per_second=float(
            payload["throughput_samples_per_second"]
        ),
        measured_losses=tuple(
            float(value) for value in payload["measured_losses"]
        ),
        golden_losses=tuple(float(value) for value in payload["golden_losses"]),
        loss_correctness=LossCorrectness(
            passed=bool(correctness["passed"]),
            decreasing=bool(correctness["decreasing"]),
            tracks_golden=bool(correctness["tracks_golden"]),
            max_relative_error=float(correctness["max_relative_error"]),
        ),
    )


def _rank_commands(
    run: ExperimentRun, affinity_mode: AffinityMode
) -> tuple[tuple[str, ...], ...]:
    config = dataclasses.replace(run.config, affinity_mode=affinity_mode)
    commands: list[tuple[str, ...]] = []
    for rank in range(config.coordinators):
        launch = build_rank_launch(config, rank, run.run_directory)
        commands.append(
            launch.command
            + (
                "--scaling_mode",
                config.mode,
                "--master_port",
                str(run.master_port),
                "--base_proxy_port",
                str(run.base_proxy_port),
            )
        )
    return tuple(commands)


def _execute_run(
    run: ExperimentRun, affinity_mode: AffinityMode, timeout_seconds: int
) -> GlobalResult:
    run.run_directory.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(run.config, affinity_mode=affinity_mode)
    commands = _rank_commands(run, affinity_mode)
    processes: list[RankProcess] = []
    with ExitStack() as stack:
        for rank, command in enumerate(commands):
            launch = build_rank_launch(config, rank, run.run_directory)
            environment = os.environ.copy()
            environment.update(launch.environment)
            environment["GLOO_SOCKET_IFNAME"] = "lo"
            stdout = stack.enter_context(
                (run.run_directory / f"rank-{rank}.stdout.log").open("wb")
            )
            stderr = stack.enter_context(
                (run.run_directory / f"rank-{rank}.stderr.log").open("wb")
            )
            processes.append(
                RankProcess(
                    subprocess.Popen(
                        command,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                )
            )
        _wait_for_rank_processes(processes, timeout_seconds)
    ranks = tuple(
        _load_rank_result(run.run_directory / f"rank-{rank}.json")
        for rank in range(config.coordinators)
    )
    result = build_global_result(config, ranks)
    payload = {
        "schema_version": 1,
        "experiment": f"D5-D7 {config.mode} multi-coordinator scaling",
        "timestamp": _utc_now(),
        "configuration": dataclasses.asdict(config),
        "commands": commands,
        "transport": "Gloo over loopback",
        "substrate": result.substrate,
        "physical_nic_routing": False,
        "result": dataclasses.asdict(result),
    }
    (run.run_directory / "global.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _write_summaries(
    output_root: Path,
    runs: Sequence[ExperimentRun],
    results: Sequence[GlobalResult],
    invocation: Sequence[str],
) -> None:
    grouped: dict[str, list[tuple[ExperimentRun, GlobalResult]]] = {}
    for run, result in zip(runs, results):
        grouped.setdefault(run.config.mode, []).append((run, result))
    breakdown_rows: list[dict[str, object]] = []
    for mode, points in grouped.items():
        baseline, scaled = points
        conclusion = build_scaling_conclusion(
            baseline[0].config,
            scaled[0].config,
            baseline[1],
            scaled[1],
        )
        summary = {
            "schema_version": 1,
            "experiment": f"{mode} multi-coordinator scaling",
            "timestamp": _utc_now(),
            "invocation": list(invocation),
            "transport": "Gloo over loopback",
            "substrate": "same-host loopback lower bound",
            "physical_nic_routing": False,
            "points": [
                {
                    "configuration": dataclasses.asdict(run.config),
                    "result": dataclasses.asdict(result),
                }
                for run, result in points
            ],
            "conclusion": dataclasses.asdict(conclusion),
        }
        (output_root / f"{mode}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for run, result in points:
            slowest = max(
                result.ranks, key=lambda rank: rank.iteration_total_seconds
            )
            breakdown_rows.append(
                {
                    "mode": mode,
                    "coordinators": run.config.coordinators,
                    "iteration_total": slowest.iteration_total_seconds,
                    "device_dispatch_aggregation": slowest.device_dispatch_aggregation_seconds,
                    "gradient_sync": slowest.gradient_sync_seconds,
                    "optimizer": slowest.optimizer_seconds,
                    "idle_other": slowest.idle_other_seconds,
                }
            )
    breakdown = {
        "schema_version": 1,
        "timestamp": _utc_now(),
        "component_semantics": {
            "device_dispatch_aggregation": "combined forward and backward device-facing dispatch and aggregation",
            "gradient_sync": "inter-coordinator Gloo AllReduce over loopback",
            "optimizer": "rank-local optimizer step",
            "idle_other": "residual uninstrumented work and waiting",
        },
        "rows": breakdown_rows,
    }
    (output_root / "breakdown.json").write_text(
        json.dumps(breakdown, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    cli = parse_scaling_cli()
    runs = build_experiment_plan(
        modes=cli.modes,
        output_root=cli.output_directory,
        workload=cli.workload,
    )
    memory_binding = _memory_binding_available()
    affinity_mode = select_affinity_mode(
        numactl_available=shutil.which("numactl") is not None,
        memory_binding_available=memory_binding,
    )
    runs = apply_affinity_mode(runs, affinity_mode)
    if cli.dry_run:
        for run in runs:
            for command in _rank_commands(run, affinity_mode):
                print(" ".join(command))
        return 0
    results = tuple(
        _execute_run(run, affinity_mode, cli.timeout_seconds) for run in runs
    )
    _write_summaries(cli.output_directory, runs, results, sys.argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
