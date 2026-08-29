"""CLI boundary for the multi-coordinator scaling experiment."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from wasp.scaling.multi_coordinator_scaling_config import (
    OPT_125M_WORKLOAD,
    TINY_WORKLOAD,
    ScalingMode,
    WorkloadConfig,
)


@dataclass(frozen=True, slots=True)
class ScalingCliConfig:
    modes: tuple[ScalingMode, ...]
    output_directory: Path
    timeout_seconds: int
    dry_run: bool
    workload: WorkloadConfig


def parse_scaling_cli() -> ScalingCliConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("strong", "weak"),
        default=("strong", "weak"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/coord_scaling")
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--measured-iterations", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    workload = (
        TINY_WORKLOAD
        if args.tiny
        else dataclasses.replace(
            OPT_125M_WORKLOAD,
            model_name=args.model_name,
            sequence_length=args.sequence_length,
            measured_iterations=args.measured_iterations,
            block_size=args.block_size,
        )
    )
    return ScalingCliConfig(
        modes=tuple(args.modes),
        output_directory=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        workload=workload,
    )
