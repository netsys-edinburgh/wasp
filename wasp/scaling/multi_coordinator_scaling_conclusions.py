"""Evidence-derived conclusions for D5 and D6 scaling points."""

from __future__ import annotations

from dataclasses import dataclass

from wasp.scaling.multi_coordinator_scaling_analysis import scaling_efficiency
from wasp.scaling.multi_coordinator_scaling_config import ScalingConfig
from wasp.scaling.multi_coordinator_scaling_results import GlobalResult


@dataclass(frozen=True, slots=True)
class ScalingConclusion:
    mode: str
    speedup: float
    efficiency: float
    throughput_ratio: float
    correctness_passed: bool


def build_scaling_conclusion(
    baseline_config: ScalingConfig,
    scaled_config: ScalingConfig,
    baseline: GlobalResult,
    scaled: GlobalResult,
) -> ScalingConclusion:
    """Compare two measured points without inferring beyond the observations."""
    return ScalingConclusion(
        mode=baseline_config.mode,
        speedup=(
            baseline.iteration_total_seconds / scaled.iteration_total_seconds
        ),
        efficiency=scaling_efficiency(
            mode=baseline_config.mode,
            baseline_seconds=baseline.iteration_total_seconds,
            measured_seconds=scaled.iteration_total_seconds,
            coordinators=scaled_config.coordinators,
        ),
        throughput_ratio=(
            scaled.throughput_samples_per_second
            / baseline.throughput_samples_per_second
        ),
        correctness_passed=(
            baseline.loss_correctness.passed and scaled.loss_correctness.passed
        ),
    )
