"""Rank process-group lifecycle for the scaling launcher."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Sequence


class RankProcess:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def terminate_group(self) -> None:
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def kill_group(self) -> None:
        try:
            os.killpg(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def cleanup_rank_processes(processes: Sequence[RankProcess]) -> None:
    for process in processes:
        process.terminate_group()
    for process in processes:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        process.kill_group()
        process.wait()


def wait_for_rank_processes(
    processes: Sequence[RankProcess], timeout_seconds: int
) -> tuple[int, ...]:
    try:
        statuses = tuple(
            process.wait(timeout=timeout_seconds) for process in processes
        )
    except subprocess.TimeoutExpired:
        cleanup_rank_processes(processes)
        raise
    if any(status != 0 for status in statuses):
        cleanup_rank_processes(processes)
        raise subprocess.CalledProcessError(max(statuses), "rank workers")
    return statuses
