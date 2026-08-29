"""Rank-local training workload for the multi-coordinator launcher."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Final

import torch.distributed as dist

from morphling.api import (
    CoordinatorMetricsCollector,
    PhaseRecorder,
    metrics_config_from_args,
    start_metrics_collector,
    track_phase,
)
from wasp.coordinator._multi_coordinator_fleet import spawn_fleet as _spawn_fleet
from wasp.coordinator._multi_coordinator_fleet import terminate_fleet as _terminate_fleet
from wasp.coordinator._multi_coordinator_workload import run_workload
from wasp.scaling.multi_coordinator_scaling_config import ScalingConfig
from wasp.scaling.multi_coordinator_scaling_results import (
    RankMeasurement,
    build_rank_result,
    rank_result_payload,
)


def _write_rank_config(
    template_path: Path, rank: int, listen_port: int, device_count: int
) -> Path:
    replacements = {
        "listen_port": f"listen_port = {listen_port}\n",
        "listen_ip": "listen_ip = 127.0.0.1\n",
        "num_device": f"num_device = {device_count}\n",
    }
    lines = template_path.read_text(encoding="utf-8").splitlines(keepends=True)
    configured = [
        replacements.get(line.lstrip().split("=", maxsplit=1)[0].strip(), line)
        for line in lines
    ]
    path = Path("/tmp") / f"svr_c{rank}.ini"
    path.write_text("".join(configured), encoding="utf-8")
    return path


def _scaling_config(args: argparse.Namespace) -> ScalingConfig:
    return ScalingConfig(
        mode=args.scaling_mode,
        coordinators=args.coords,
        total_devices=args.coords * args.devices_per_coord,
        global_batch=args.coords * args.local_batch,
        devices_per_coordinator=args.devices_per_coord,
        local_batch=args.local_batch,
        warmup_iterations=args.warmup_steps,
        measured_iterations=args.steps,
        relative_tolerance=args.rtol,
        tiny=args.tiny,
        model_name=args.model_name,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        learning_rate=args.lr,
        distributed_timeout_seconds=args.distributed_timeout_seconds,
    )


def _write_result(
    args: argparse.Namespace,
    rank: int,
    recorder: PhaseRecorder,
    losses: tuple[float, ...],
    golden: tuple[float, ...],
) -> None:
    if args.result_output is None:
        return
    config = _scaling_config(args)
    result = build_rank_result(
        config,
        RankMeasurement(
            rank, recorder.snapshot().durations_seconds, losses, golden
        ),
    )
    path = Path(args.result_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            rank_result_payload(config, result), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def _write_failure(
    args: argparse.Namespace, rank: int, error: Exception
) -> None:
    if args.result_output is None:
        return
    path = Path(args.result_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rank": rank,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_coordinator(rank: int, args: argparse.Namespace) -> int:
    proxy_port = args.base_proxy_port + rank
    os.environ.update(
        MORPHLING_PROXY_HOST="127.0.0.1",
        MORPHLING_PROXY_PORT=str(proxy_port),
        NUM_DEVICES=str(args.devices_per_coord),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(args.master_port),
    )
    recorder = PhaseRecorder()
    collector: CoordinatorMetricsCollector | None = None
    backend = None
    fleet = ()
    group_initialized = False
    metrics_config = metrics_config_from_args(args, rank=rank)
    try:
        if metrics_config is not None:
            collector = start_metrics_collector(metrics_config, recorder)
        with track_phase(recorder, "idle_wait"):
            dist.init_process_group(
                "gloo",
                rank=rank,
                world_size=args.coords,
                timeout=timedelta(seconds=args.distributed_timeout_seconds),
            )
        group_initialized = True
        from wasp.coordinator._runtime_common import start_backend, wait_for_connections

        root = Path(__file__).resolve().parents[1]
        rank_config = _write_rank_config(
            root / "config/proxy/svr.ini",
            rank,
            proxy_port,
            args.devices_per_coord,
        )
        backend = start_backend("proxy", args.block_size, str(rank_config))
        import morphling.hooks.autograd as hooks_autograd

        hooks_autograd._backend = backend
        with track_phase(recorder, "idle_wait"):
            time.sleep(3)
        fleet = _spawn_fleet(
            rank, args.devices_per_coord, root / "config/proxy/cli.ini"
        )
        if hasattr(backend, "get_connection_count"):
            with track_phase(recorder, "idle_wait"):
                wait_for_connections(backend, args.devices_per_coord, 120)
        with track_phase(recorder, "idle_wait"):
            time.sleep(3)
        config = _scaling_config(args)
        losses, golden = run_workload(rank, config, recorder)
        _write_result(args, rank, recorder, losses, golden)
        passed = build_rank_result(
            config,
            RankMeasurement(
                rank, recorder.snapshot().durations_seconds, losses, golden
            ),
        ).loss_correctness.passed
        with track_phase(recorder, "idle_wait"):
            dist.barrier()
        return 0 if passed else 1
    finally:
        if collector is not None:
            collector.stop()
        if group_initialized and dist.is_initialized():
            dist.destroy_process_group()
        _terminate_fleet(fleet)
        if backend is not None and hasattr(backend, "stop"):
            backend.stop()


def worker_boundary(rank: int, args: argparse.Namespace) -> None:
    try:
        status = run_coordinator(rank, args)
    except Exception as error:  # noqa: BLE001 - top-level worker boundary
        _write_failure(args, rank, error)
        traceback.print_exc()
        status = 1
    os._exit(status)
