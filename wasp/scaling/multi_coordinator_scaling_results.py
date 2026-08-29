"""Typed rank and global results for multi-coordinator measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, TypedDict

try:  # Python 3.11+
    from typing import assert_never
except ImportError:  # Python 3.10
    from typing_extensions import assert_never

from wasp.scaling.multi_coordinator_scaling_analysis import reconcile_breakdown
from wasp.scaling.multi_coordinator_scaling_config import ScalingConfig


@dataclass(frozen=True, slots=True)
class RankMeasurement:
    rank: int
    phase_durations: Mapping[str, float]
    measured_losses: tuple[float, ...]
    golden_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LossCorrectness:
    passed: bool
    decreasing: bool
    tracks_golden: bool
    max_relative_error: float


@dataclass(frozen=True, slots=True)
class RankResult:
    rank: int
    warmup_iteration_seconds: float
    iteration_total_seconds: float
    device_dispatch_aggregation_seconds: float
    gradient_sync_seconds: float
    optimizer_seconds: float
    idle_other_seconds: float
    throughput_samples_per_second: float
    measured_losses: tuple[float, ...]
    golden_losses: tuple[float, ...]
    loss_correctness: LossCorrectness


@dataclass(frozen=True, slots=True)
class GlobalResult:
    coordinators: int
    iteration_total_seconds: float
    throughput_samples_per_second: float
    loss_correctness: LossCorrectness
    substrate: str
    ranks: tuple[RankResult, ...]


class LossCorrectnessPayload(TypedDict):
    passed: bool
    decreasing: bool
    tracks_golden: bool
    max_relative_error: float


class PhasePayload(TypedDict):
    iteration_total: float
    device_dispatch_aggregation: float
    gradient_sync: float
    optimizer: float
    idle_other: float


class RankResultPayload(TypedDict):
    schema_version: int
    rank: int
    coordinators: int
    warmup_iterations: int
    measured_iterations: int
    local_batch: int
    devices_per_coordinator: int
    relative_tolerance: float
    model_name: str
    sequence_length: int
    warmup_iteration_seconds: float
    throughput_samples_per_second: float
    measured_losses: tuple[float, ...]
    golden_losses: tuple[float, ...]
    loss_correctness: LossCorrectnessPayload
    phases_seconds: PhasePayload


def _loss_correctness(
    measured: tuple[float, ...],
    golden: tuple[float, ...],
    relative_tolerance: float,
) -> LossCorrectness:
    finite = all(math.isfinite(loss) for loss in measured)
    decreasing = finite and len(measured) > 1 and measured[-1] < measured[0]
    relative_errors = tuple(
        abs(actual - expected) / max(abs(expected), 1e-6)
        for actual, expected in zip(measured, golden)
    )
    max_relative_error = max(relative_errors, default=math.inf)
    tracks = (
        finite
        and len(measured) == len(golden)
        and max_relative_error <= relative_tolerance
    )
    return LossCorrectness(
        passed=tracks,
        decreasing=decreasing,
        tracks_golden=tracks,
        max_relative_error=max_relative_error,
    )


def build_rank_result(
    config: ScalingConfig, measurement: RankMeasurement
) -> RankResult:
    phases = measurement.phase_durations
    iteration_total = phases.get("iteration_total", 0.0)
    breakdown = reconcile_breakdown(
        iteration_total_seconds=iteration_total,
        forward_device_seconds=phases.get(
            "forward_device_dispatch_aggregation", 0.0
        ),
        backward_device_seconds=phases.get(
            "backward_device_dispatch_aggregation", 0.0
        ),
        gradient_sync_seconds=phases.get("gradient_sync", 0.0),
        optimizer_seconds=phases.get("optimizer", 0.0),
        tolerance_seconds=max(1e-9, iteration_total * 1e-6),
    )
    throughput = (
        config.local_batch * config.measured_iterations / iteration_total
        if iteration_total > 0
        else 0.0
    )
    return RankResult(
        rank=measurement.rank,
        warmup_iteration_seconds=phases.get("warmup_iteration", 0.0),
        iteration_total_seconds=iteration_total,
        device_dispatch_aggregation_seconds=(
            breakdown.device_dispatch_aggregation_seconds
        ),
        gradient_sync_seconds=breakdown.gradient_sync_seconds,
        optimizer_seconds=breakdown.optimizer_seconds,
        idle_other_seconds=breakdown.idle_other_seconds,
        throughput_samples_per_second=throughput,
        measured_losses=measurement.measured_losses,
        golden_losses=measurement.golden_losses,
        loss_correctness=_loss_correctness(
            measurement.measured_losses,
            measurement.golden_losses,
            config.relative_tolerance,
        ),
    )


def build_global_result(
    config: ScalingConfig, ranks: tuple[RankResult, ...]
) -> GlobalResult:
    makespan = max(rank.iteration_total_seconds for rank in ranks)
    throughput = (
        config.global_batch * config.measured_iterations / makespan
        if makespan > 0
        else 0.0
    )
    correctness = LossCorrectness(
        passed=all(rank.loss_correctness.passed for rank in ranks),
        decreasing=all(rank.loss_correctness.decreasing for rank in ranks),
        tracks_golden=all(
            rank.loss_correctness.tracks_golden for rank in ranks
        ),
        max_relative_error=max(
            rank.loss_correctness.max_relative_error for rank in ranks
        ),
    )
    match config.affinity_mode:
        case "numactl":
            placement = "NUMA CPU and memory binding"
        case "taskset":
            placement = "CPU affinity with first-touch memory"
        case unreachable:
            assert_never(unreachable)
    return GlobalResult(
        coordinators=config.coordinators,
        iteration_total_seconds=makespan,
        throughput_samples_per_second=throughput,
        loss_correctness=correctness,
        substrate=f"same-host Gloo loopback lower bound; {placement}; "
        "no physical NIC-to-NIC routing",
        ranks=ranks,
    )


def rank_result_payload(
    config: ScalingConfig, result: RankResult
) -> RankResultPayload:
    return {
        "schema_version": 1,
        "rank": result.rank,
        "coordinators": config.coordinators,
        "warmup_iterations": config.warmup_iterations,
        "measured_iterations": config.measured_iterations,
        "local_batch": config.local_batch,
        "devices_per_coordinator": config.devices_per_coordinator,
        "relative_tolerance": config.relative_tolerance,
        "model_name": config.model_name,
        "sequence_length": config.sequence_length,
        "warmup_iteration_seconds": result.warmup_iteration_seconds,
        "throughput_samples_per_second": result.throughput_samples_per_second,
        "measured_losses": result.measured_losses,
        "golden_losses": result.golden_losses,
        "loss_correctness": {
            "passed": result.loss_correctness.passed,
            "decreasing": result.loss_correctness.decreasing,
            "tracks_golden": result.loss_correctness.tracks_golden,
            "max_relative_error": result.loss_correctness.max_relative_error,
        },
        "phases_seconds": {
            "iteration_total": result.iteration_total_seconds,
            "device_dispatch_aggregation": (
                result.device_dispatch_aggregation_seconds
            ),
            "gradient_sync": result.gradient_sync_seconds,
            "optimizer": result.optimizer_seconds,
            "idle_other": result.idle_other_seconds,
        },
    }
