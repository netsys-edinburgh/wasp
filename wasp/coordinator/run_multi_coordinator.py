#!/usr/bin/env python3
"""Run correct data-parallel training across one or two coordinators."""

from __future__ import annotations

import argparse
import os

import torch.multiprocessing as mp

from morphling.api import add_metrics_arguments
from wasp.coordinator._multi_coordinator_training import worker_boundary
from wasp.scaling.multi_coordinator_scaling_conclusions import (
    ScalingConclusion,
    build_scaling_conclusion,
)
from wasp.scaling.multi_coordinator_scaling_config import (
    OPT_125M_WORKLOAD,
    TINY_WORKLOAD,
    RankLaunch,
    ScalingConfig,
    WorkloadConfig,
    build_iteration_plan,
    build_rank_launch,
    build_scaling_configs,
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
    RankMeasurement,
    RankResult,
    build_global_result,
    build_rank_result,
    rank_result_payload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--scaling_mode", choices=("strong", "weak"), default="strong"
    )
    parser.add_argument("--coords", type=int, choices=(1, 2), default=2)
    parser.add_argument("--devices_per_coord", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--local_batch", type=int, default=8)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--base_proxy_port", type=int, default=39000)
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=120)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--result_output", type=str, default=None)
    add_metrics_arguments(parser)
    return parser


def _spawned_worker(rank: int, args: argparse.Namespace) -> None:
    worker_boundary(rank, args)


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        rank = int(os.environ["RANK"])
        worker_boundary(rank, args)
        return 0

    mp.spawn(_spawned_worker, args=(args,), nprocs=args.coords, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
