"""Rank-owned fake-device process lifecycle."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol, Sequence

import torch


class FleetProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def spawn_fleet(
    rank: int, count: int, device_config: Path
) -> tuple[subprocess.Popen[bytes], ...]:
    gpu_count = max(1, torch.cuda.device_count())
    log_directory = os.environ.get("MORPHLING_DEV_LOG_DIR")
    processes: list[subprocess.Popen[bytes]] = []
    for index in range(count):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(
            (rank * count + index) % gpu_count
        )
        command = (
            "morphling_device",
            "--id",
            str(rank * count + index),
            "--flops",
            "100T",
            "--memory",
            "8G",
            "--ul_bw",
            "10G",
            "--dl_bw",
            "10G",
            "--ul_lat",
            "0",
            "--dl_lat",
            "0",
            "--backend",
            "proxy",
            "--cfg",
            str(device_config),
        )
        output_path = (
            Path(log_directory) / f"dev_c{rank}_{index}.log"
            if log_directory
            else Path(os.devnull)
        )
        with output_path.open("wb") as output:
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=output,
                    stderr=output,
                    start_new_session=False,
                )
            )
    return tuple(processes)


def terminate_fleet(processes: Sequence[FleetProcess]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
