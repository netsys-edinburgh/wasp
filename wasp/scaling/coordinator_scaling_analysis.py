"""Pure analysis functions for coordinator scaling measurements."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wasp.scaling.coordinator_scaling_conclusions import (
    ScalingPoint,
    assess_saturation,
)
from wasp.scaling.multi_coordinator_scaling_analysis import (
    BreakdownReconciliationError,
    PhaseBreakdown,
    reconcile_breakdown,
    response_counters_are_balanced,
    scaling_efficiency,
)

__all__ = (
    "BreakdownReconciliationError",
    "PhaseBreakdown",
    "ScalingPoint",
    "assess_saturation",
    "reconcile_breakdown",
    "scaling_efficiency",
)


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    iteration_runtime_seconds: float | None
    peak_cpu_percent: float | None
    median_cpu_percent: float | None
    peak_rss_bytes: int | None
    peak_nic_tx_bytes_per_sec: float | None
    median_nic_tx_bytes_per_sec: float | None
    peak_nic_rx_bytes_per_sec: float | None
    median_nic_rx_bytes_per_sec: float | None
    phase_counts: Mapping[str, int]
    phase_durations_seconds: Mapping[str, float]
    error: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScalingAggregate:
    device_count: int
    repetitions: int
    successful_repetitions: int
    iteration_runtime_seconds: float | None
    peak_cpu_percent: float | None
    median_cpu_percent: float | None
    peak_rss_bytes: int | None
    peak_nic_tx_bytes_per_sec: float | None
    median_nic_tx_bytes_per_sec: float | None
    peak_nic_rx_bytes_per_sec: float | None
    median_nic_rx_bytes_per_sec: float | None
    phase_counts: Mapping[str, float]
    phase_durations_seconds: Mapping[str, float]


def _empty_measurement(
    error: str, warnings: tuple[str, ...] = ()
) -> RunMeasurement:
    return RunMeasurement(
        iteration_runtime_seconds=None,
        peak_cpu_percent=None,
        median_cpu_percent=None,
        peak_rss_bytes=None,
        peak_nic_tx_bytes_per_sec=None,
        median_nic_tx_bytes_per_sec=None,
        peak_nic_rx_bytes_per_sec=None,
        median_nic_rx_bytes_per_sec=None,
        phase_counts={},
        phase_durations_seconds={},
        error=error,
        warnings=warnings,
    )


def parse_metrics_jsonl(path: Path) -> RunMeasurement:
    numbered_lines = [
        (line_number, line)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    records = []
    warnings: list[str] = []
    for position, (line_number, line) in enumerate(numbered_lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            message = (
                f"malformed metrics JSONL at line {line_number}: {error.msg}"
            )
            if position != len(numbered_lines) - 1:
                return _empty_measurement(message)
            warnings.append(
                f"trailing partial metrics JSONL record at line {line_number}: "
                f"{error.msg}"
            )
    terminal_errors = [
        record
        for record in records
        if record.get("record_type") == "terminal_error"
    ]
    samples = [
        record for record in records if record.get("record_type") == "sample"
    ]
    measured = [
        sample
        for sample in samples
        if sample.get("phase_counts", {}).get("iteration_total", 0) > 0
    ]
    if not measured:
        if terminal_errors:
            terminal = terminal_errors[-1]
            return _empty_measurement(
                f"{terminal['error_type']}: {terminal['error_message']}",
                tuple(warnings),
            )
        return _empty_measurement(
            "no measured iteration samples", tuple(warnings)
        )

    final = measured[-1]
    phase_counts = {
        str(name): int(count) for name, count in final["phase_counts"].items()
    }
    phase_durations = {
        str(name): float(duration)
        for name, duration in final["phase_durations_seconds"].items()
    }
    iteration_count = phase_counts.get("iteration_total", 0)
    iteration_total = phase_durations.get("iteration_total", 0.0)
    error = None
    if terminal_errors:
        terminal = terminal_errors[-1]
        error = f"{terminal['error_type']}: {terminal['error_message']}"
    return RunMeasurement(
        iteration_runtime_seconds=(
            iteration_total / iteration_count if iteration_count > 0 else None
        ),
        peak_cpu_percent=max(
            float(sample["process_cpu_percent"]) for sample in measured
        ),
        median_cpu_percent=statistics.median(
            float(sample["process_cpu_percent"]) for sample in measured
        ),
        peak_rss_bytes=max(int(sample["rss_bytes"]) for sample in measured),
        peak_nic_tx_bytes_per_sec=max(
            float(sample["nic_tx_bytes_per_sec"]) for sample in measured
        ),
        median_nic_tx_bytes_per_sec=statistics.median(
            float(sample["nic_tx_bytes_per_sec"]) for sample in measured
        ),
        peak_nic_rx_bytes_per_sec=max(
            float(sample["nic_rx_bytes_per_sec"]) for sample in measured
        ),
        median_nic_rx_bytes_per_sec=statistics.median(
            float(sample["nic_rx_bytes_per_sec"]) for sample in measured
        ),
        phase_counts=phase_counts,
        phase_durations_seconds=phase_durations,
        error=error,
        warnings=tuple(warnings),
    )


def _mean_present(values: Sequence[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def aggregate_repetitions(
    device_count: int,
    measurements: Sequence[RunMeasurement],
    exit_statuses: Sequence[int],
) -> ScalingAggregate:
    successful = [
        measurement
        for measurement, exit_status in zip(measurements, exit_statuses)
        if exit_status == 0
        and measurement.error is None
        and measurement.iteration_runtime_seconds is not None
    ]
    phase_names = {
        name for measurement in successful for name in measurement.phase_counts
    }
    duration_names = {
        name
        for measurement in successful
        for name in measurement.phase_durations_seconds
    }
    peak_cpu_values = [
        measurement.peak_cpu_percent for measurement in successful
    ]
    peak_rss_values = [measurement.peak_rss_bytes for measurement in successful]
    return ScalingAggregate(
        device_count=device_count,
        repetitions=len(measurements),
        successful_repetitions=len(successful),
        iteration_runtime_seconds=_mean_present(
            [
                measurement.iteration_runtime_seconds
                for measurement in successful
            ]
        ),
        peak_cpu_percent=max(peak_cpu_values) if peak_cpu_values else None,
        median_cpu_percent=_mean_present(
            [measurement.median_cpu_percent for measurement in successful]
        ),
        peak_rss_bytes=(
            max(value for value in peak_rss_values if value is not None)
            if any(value is not None for value in peak_rss_values)
            else None
        ),
        peak_nic_tx_bytes_per_sec=max(
            (
                measurement.peak_nic_tx_bytes_per_sec or 0.0
                for measurement in successful
            ),
            default=0.0,
        )
        if successful
        else None,
        median_nic_tx_bytes_per_sec=_mean_present(
            [
                measurement.median_nic_tx_bytes_per_sec
                for measurement in successful
            ]
        ),
        peak_nic_rx_bytes_per_sec=max(
            (
                measurement.peak_nic_rx_bytes_per_sec or 0.0
                for measurement in successful
            ),
            default=0.0,
        )
        if successful
        else None,
        median_nic_rx_bytes_per_sec=_mean_present(
            [
                measurement.median_nic_rx_bytes_per_sec
                for measurement in successful
            ]
        ),
        phase_counts={
            name: statistics.mean(
                measurement.phase_counts.get(name, 0)
                for measurement in successful
            )
            for name in sorted(phase_names)
        },
        phase_durations_seconds={
            name: statistics.mean(
                measurement.phase_durations_seconds.get(name, 0.0)
                for measurement in successful
            )
            for name in sorted(duration_names)
        },
    )
