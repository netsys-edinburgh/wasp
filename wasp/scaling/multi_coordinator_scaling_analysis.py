"""Pure D5-D7 scaling and phase-breakdown calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

try:  # Python 3.11+
    from typing import assert_never
except ImportError:  # Python 3.10
    from typing_extensions import assert_never

ScalingMode = Literal["strong", "weak"]
ResponseCounterRecord = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class BreakdownReconciliationError(ValueError):
    residual_seconds: float

    def __str__(self) -> str:
        return f"breakdown components exceed iteration total by {-self.residual_seconds:.9f}s"


@dataclass(frozen=True, slots=True)
class PhaseBreakdown:
    iteration_total_seconds: float
    device_dispatch_aggregation_seconds: float
    gradient_sync_seconds: float
    optimizer_seconds: float
    idle_other_seconds: float
    reconciles: bool


def scaling_efficiency(
    *,
    mode: ScalingMode,
    baseline_seconds: float,
    measured_seconds: float,
    coordinators: int,
) -> float:
    """Compute conventional strong- or weak-scaling efficiency."""
    match mode:
        case "strong":
            return baseline_seconds / (coordinators * measured_seconds)
        case "weak":
            return baseline_seconds / measured_seconds
        case unreachable:
            assert_never(unreachable)


def response_counters_are_balanced(
    records: Sequence[ResponseCounterRecord],
) -> bool:
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for operation_id, count, previous, current in records:
        if count != 1 or previous != current + 1 or current < 0:
            return False
        grouped.setdefault(operation_id, []).append((count, previous, current))
    if not grouped:
        return False
    return all(
        sorted(previous for _, previous, _ in operation_records)
        == list(range(1, len(operation_records) + 1))
        for operation_records in grouped.values()
    )


def reconcile_breakdown(
    *,
    iteration_total_seconds: float,
    forward_device_seconds: float,
    backward_device_seconds: float,
    gradient_sync_seconds: float,
    optimizer_seconds: float,
    tolerance_seconds: float = 1e-9,
) -> PhaseBreakdown:
    """Combine device phases and assign uninstrumented time to residual idle."""
    device_seconds = forward_device_seconds + backward_device_seconds
    accounted = device_seconds + gradient_sync_seconds + optimizer_seconds
    residual = iteration_total_seconds - accounted
    if residual < -tolerance_seconds:
        raise BreakdownReconciliationError(residual)
    idle_other = max(0.0, residual)
    reconciled_total = accounted + idle_other
    return PhaseBreakdown(
        iteration_total_seconds=iteration_total_seconds,
        device_dispatch_aggregation_seconds=device_seconds,
        gradient_sync_seconds=gradient_sync_seconds,
        optimizer_seconds=optimizer_seconds,
        idle_other_seconds=idle_other,
        reconciles=abs(reconciled_total - iteration_total_seconds)
        <= tolerance_seconds,
    )
