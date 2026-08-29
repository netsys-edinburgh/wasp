"""Execution plan values for D5-D7 multi-coordinator runs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from wasp.scaling.multi_coordinator_scaling_config import (
    TINY_WORKLOAD,
    AffinityMode,
    ScalingConfig,
    ScalingMode,
    WorkloadConfig,
    build_scaling_configs,
)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    config: ScalingConfig
    run_directory: Path
    master_port: int
    base_proxy_port: int


def build_experiment_plan(
    *,
    modes: Sequence[ScalingMode],
    output_root: Path,
    workload: WorkloadConfig = TINY_WORKLOAD,
) -> tuple[ExperimentRun, ...]:
    runs: list[ExperimentRun] = []
    index = 0
    for mode in modes:
        for config in build_scaling_configs(mode, workload):
            runs.append(
                ExperimentRun(
                    config=config,
                    run_directory=(
                        output_root
                        / mode
                        / f"coordinators-{config.coordinators}"
                    ),
                    master_port=29600 + index,
                    base_proxy_port=39200 + index * 10,
                )
            )
            index += 1
    return tuple(runs)


def apply_affinity_mode(
    runs: Sequence[ExperimentRun], affinity_mode: AffinityMode
) -> tuple[ExperimentRun, ...]:
    return tuple(
        dataclasses.replace(
            run,
            config=dataclasses.replace(run.config, affinity_mode=affinity_mode),
        )
        for run in runs
    )
