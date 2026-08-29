"""Subprocess execution for coordinator scaling measurements."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

IMAGE = "device-emulator:latest"
SUBSTRATE = "single-host loopback/emulated devices"


@dataclass(frozen=True, slots=True)
class RunSpec:
    device_count: int
    repetition: int
    port: int
    run_directory: Path
    container_directory: Path


def render_proxy_config(template: str, device_count: int, port: int) -> str:
    loopback = re.sub(
        r"(?m)^listen_ip\s*=\s*\S+\s*$",
        "listen_ip = 127.0.0.1",
        template,
        count=1,
    )
    configured = re.sub(
        r"(?m)^listen_port\s*=\s*\d+\s*$",
        f"listen_port = {port}",
        loopback,
        count=1,
    )
    return re.sub(
        r"(?m)^num_device\s*=\s*\d+\s*$",
        f"num_device = {device_count}",
        configured,
        count=1,
    )


def build_inner_command(run: RunSpec, metrics_interval: float) -> list[str]:
    return [
        "python3",
        "scripts/run_single_coordinator_training.py",
        "--num_devices",
        str(run.device_count),
        "--model_name",
        "facebook/opt-125m",
        "--steps",
        "3",
        "--warmup_steps",
        "1",
        "--batch_size",
        "1",
        "--seq_length",
        "32",
        "--block_size",
        "256",
        "--cfg",
        str(run.container_directory / "proxy.ini"),
        "--metrics_output",
        str(run.container_directory / "metrics.jsonl"),
        "--metrics_interval",
        str(metrics_interval),
        "--metrics_nics",
        "lo",
    ]


def docker_container_name(run: RunSpec) -> str:
    return f"morphling-d2-d{run.device_count}-r{run.repetition}-p{run.port}"


def build_docker_command(
    run: RunSpec, output_root: Path, image: str, metrics_interval: float
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        docker_container_name(run),
        "--gpus",
        "all",
        "--ulimit",
        "memlock=-1",
        "--ipc",
        "host",
        "-v",
        f"{output_root.resolve()}:/scaling-output",
        "-e",
        f"CFG_PATH={run.container_directory / 'proxy.ini'}",
        image,
        *build_inner_command(run, metrics_interval),
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_command(
    run: RunSpec,
    output_root: Path,
    image: str,
    metrics_interval: float,
    timeout_seconds: int,
) -> None:
    run.run_directory.mkdir(parents=True, exist_ok=True)
    template = Path("config/proxy/svr.ini").read_text(encoding="utf-8")
    config_text = render_proxy_config(template, run.device_count, run.port)
    (run.run_directory / "proxy.ini").write_text(config_text, encoding="utf-8")
    inner = build_inner_command(run, metrics_interval)
    inside_docker = Path("/.dockerenv").exists()
    if inside_docker:
        command = inner
        environment = os.environ.copy()
        environment["CFG_PATH"] = str(run.container_directory / "proxy.ini")
    else:
        command = build_docker_command(
            run, output_root, image, metrics_interval
        )
        environment = None
    started = _utc_now()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
        exit_status = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_status = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if not inside_docker:
            subprocess.run(
                ["docker", "rm", "-f", docker_container_name(run)],
                capture_output=True,
                text=True,
                check=False,
            )
    (run.run_directory / "stdout.log").write_text(stdout, encoding="utf-8")
    (run.run_directory / "stderr.log").write_text(stderr, encoding="utf-8")
    _write_json(
        run.run_directory / "result.json",
        {
            "command": command,
            "config": {
                "device_count": run.device_count,
                "repetition": run.repetition,
                "port": run.port,
                "model": "facebook/opt-125m",
                "batch_size": 1,
                "sequence_length": 32,
                "block_size": 256,
                "warmup_optimizer_steps": 1,
                "measured_optimizer_steps": 3,
                "worker_thread_count": 56,
                "metrics_interval_seconds": metrics_interval,
                "metrics_nics": ["lo"],
                "proxy_config": config_text,
            },
            "started_at": started,
            "finished_at": _utc_now(),
            "exit_status": exit_status,
            "timed_out": timed_out,
            "substrate": SUBSTRATE,
            "real_vs_emulated": "real coordinator process; emulated devices",
        },
    )
