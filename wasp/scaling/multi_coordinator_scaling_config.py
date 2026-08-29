"""Configuration and rank launch commands for multi-coordinator scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping

try:  # Python 3.11+
    from typing import assert_never
except ImportError:  # Python 3.10
    from typing_extensions import assert_never

ScalingMode = Literal["strong", "weak"]
AffinityMode = Literal["numactl", "taskset"]
IterationKind = Literal["warmup", "measured"]

NUMA_CPU_RANGES: Final = ("0-27", "28-55")
WARMUP_ITERATIONS: Final = 1
MEASURED_ITERATIONS: Final = 3


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    tiny: bool
    model_name: str
    sequence_length: int
    measured_iterations: int
    block_size: int
    learning_rate: float


TINY_WORKLOAD: Final = WorkloadConfig(
    tiny=True,
    model_name="tiny-mlp",
    sequence_length=1,
    measured_iterations=3,
    block_size=128,
    learning_rate=0.01,
)
OPT_125M_WORKLOAD: Final = WorkloadConfig(
    tiny=False,
    model_name="facebook/opt-125m",
    sequence_length=32,
    measured_iterations=2,
    block_size=256,
    learning_rate=5e-5,
)


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    mode: ScalingMode
    coordinators: int
    total_devices: int
    global_batch: int
    devices_per_coordinator: int
    local_batch: int
    warmup_iterations: int = WARMUP_ITERATIONS
    measured_iterations: int = MEASURED_ITERATIONS
    affinity_mode: AffinityMode = "numactl"
    relative_tolerance: float = 0.05
    tiny: bool = True
    model_name: str = "tiny-mlp"
    sequence_length: int = 1
    block_size: int = 128
    learning_rate: float = 0.01
    distributed_timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class RankLaunch:
    rank: int
    numa_node: int
    cpu_range: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    substrate: str


def build_scaling_configs(
    mode: ScalingMode,
    workload: WorkloadConfig = TINY_WORKLOAD,
) -> tuple[ScalingConfig, ...]:
    """Return the fixed D5 strong or D6 weak scaling matrix."""
    if workload.tiny:
        strong_points = ((1, 8, 16, 8, 16), (2, 8, 16, 4, 8))
        weak_points = ((1, 4, 8, 4, 8), (2, 8, 16, 4, 8))
    else:
        strong_points = ((1, 8, 2, 8, 2), (2, 8, 2, 4, 1))
        weak_points = ((1, 4, 1, 4, 1), (2, 8, 2, 4, 1))
    match mode:
        case "strong":
            points = strong_points
        case "weak":
            points = weak_points
        case unreachable:
            assert_never(unreachable)
    return tuple(
        ScalingConfig(
            mode=mode,
            coordinators=coordinators,
            total_devices=total_devices,
            global_batch=global_batch,
            devices_per_coordinator=devices_per_coordinator,
            local_batch=local_batch,
            measured_iterations=workload.measured_iterations,
            tiny=workload.tiny,
            model_name=workload.model_name,
            sequence_length=workload.sequence_length,
            block_size=workload.block_size,
            learning_rate=workload.learning_rate,
        )
        for (
            coordinators,
            total_devices,
            global_batch,
            devices_per_coordinator,
            local_batch,
        ) in points
    )


def select_affinity_mode(
    *, numactl_available: bool, memory_binding_available: bool
) -> AffinityMode:
    """Prefer NUMA memory binding and otherwise use CPU first-touch placement."""
    if numactl_available and memory_binding_available:
        return "numactl"
    return "taskset"


def build_iteration_plan(config: ScalingConfig) -> tuple[IterationKind, ...]:
    return ("warmup",) * config.warmup_iterations + (
        "measured",
    ) * config.measured_iterations


def build_rank_launch(
    config: ScalingConfig,
    rank: int,
    output_directory: Path,
) -> RankLaunch:
    node = rank
    cpu_range = NUMA_CPU_RANGES[node]
    common = (
        "python3",
        "scripts/run_multi_coordinator.py",
        "--worker",
        "--coords",
        str(config.coordinators),
        "--devices_per_coord",
        str(config.devices_per_coordinator),
        "--local_batch",
        str(config.local_batch),
        "--warmup_steps",
        str(config.warmup_iterations),
        "--steps",
        str(config.measured_iterations),
        "--rtol",
        str(config.relative_tolerance),
        "--model-name",
        config.model_name,
        "--sequence-length",
        str(config.sequence_length),
        "--block_size",
        str(config.block_size),
        "--lr",
        str(config.learning_rate),
        "--distributed-timeout-seconds",
        str(config.distributed_timeout_seconds),
        "--metrics_output",
        str(output_directory / "metrics.jsonl"),
        "--result_output",
        str(output_directory / f"rank-{rank}.json"),
        "--metrics_nics",
        "lo",
    )
    match config.affinity_mode:
        case "numactl":
            prefix = (
                "numactl",
                f"--cpunodebind={node}",
                f"--membind={node}",
            )
            substrate = "same-host Gloo loopback lower bound; NUMA CPU and memory binding"
        case "taskset":
            prefix = ("taskset", "--cpu-list", cpu_range)
            substrate = (
                "same-host Gloo loopback lower bound; CPU affinity with "
                "first-touch memory"
            )
        case unreachable:
            assert_never(unreachable)
    tiny_flag = ("--tiny",) if config.tiny else ()
    return RankLaunch(
        rank=rank,
        numa_node=node,
        cpu_range=cpu_range,
        command=prefix + common + tiny_flag,
        environment={
            "RANK": str(rank),
            "WORLD_SIZE": str(config.coordinators),
            "MASTER_ADDR": "127.0.0.1",
        },
        substrate=substrate,
    )
