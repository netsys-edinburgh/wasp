"""Conclusion rules for measured coordinator scaling points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    device_count: int
    iteration_runtime_seconds: float
    median_cpu_percent: float
    peak_cpu_percent: float


@dataclass(frozen=True, slots=True)
class SaturationAssessment:
    runtime_plateau_observed: bool
    resource_saturation_observed: bool
    median_cpu_normalized_percent: float
    peak_cpu_normalized_percent: float
    bottleneck: str
    runtime_plateau_criterion: str
    resource_saturation_criterion: str


def assess_saturation(
    points: Sequence[ScalingPoint], logical_cpu_count: int
) -> SaturationAssessment:
    runtime_criterion = "largest-device iteration runtime is at least 95% of the preceding point"
    resource_criterion = "largest-device median process CPU is at least 90% of logical host CPU capacity"
    if len(points) < 2 or logical_cpu_count <= 0:
        return SaturationAssessment(
            False,
            False,
            0.0,
            0.0,
            "none within measured range through local emulated devices",
            runtime_criterion,
            resource_criterion,
        )
    previous, current = points[-2:]
    median_normalized = current.median_cpu_percent / logical_cpu_count
    peak_normalized = current.peak_cpu_percent / logical_cpu_count
    runtime_plateau = (
        current.iteration_runtime_seconds / previous.iteration_runtime_seconds
        >= 0.95
    )
    resource_saturation = median_normalized >= 90.0
    if runtime_plateau and resource_saturation:
        bottleneck = "coordinator CPU"
    elif runtime_plateau:
        bottleneck = "coordinator-side host work / specific resource unisolated"
    else:
        bottleneck = (
            "none within measured range through "
            f"{current.device_count} local emulated devices"
        )
    return SaturationAssessment(
        runtime_plateau,
        resource_saturation,
        median_normalized,
        peak_normalized,
        bottleneck,
        runtime_criterion,
        resource_criterion,
    )
