#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SCS_PRESETS = {
    15: {"slot_ns": 1_000_000, "budget_min_us": 500, "budget_max_us": 714},
    30: {"slot_ns": 500_000, "budget_min_us": 200, "budget_max_us": 429},
    60: {"slot_ns": 250_000, "budget_min_us": 150, "budget_max_us": 411},
    120: {"slot_ns": 125_000, "budget_min_us": 100, "budget_max_us": 321},
}


def _fgn_autocov(H: float, k: int) -> float:
    k_abs = abs(int(k))
    return 0.5 * (
        abs(k_abs - 1) ** (2.0 * H)
        - 2.0 * (k_abs ** (2.0 * H))
        + (k_abs + 1) ** (2.0 * H)
    )


def generate_fgn(
    n: int, H: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    if n <= 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([rng.normal()], dtype=np.float64)

    r = np.array([_fgn_autocov(H, k) for k in range(n)], dtype=np.float64)
    g = np.concatenate([r, [0.0], r[1:][::-1]])
    m = g.size

    eig = np.fft.fft(g).real
    eig = np.maximum(eig, 0.0)

    V = np.zeros(m, dtype=np.complex128)
    V[0] = np.sqrt(eig[0] / m) * rng.normal()
    V[n] = np.sqrt(eig[n] / m) * rng.normal()
    for k in range(1, n):
        a = rng.normal()
        b = rng.normal()
        scale = np.sqrt(eig[k] / (2.0 * m))
        V[k] = scale * (a + 1j * b)
        V[m - k] = np.conj(V[k])

    x = np.fft.fft(V).real[:n]
    return x.astype(np.float64)


def load_to_sm(
    load: NDArray[np.float64] | float,
    sm_min: int,
    sm_max: int,
    sm_step: int,
) -> NDArray[np.int64]:
    arr = np.asarray(load, dtype=np.float64)
    steps = (sm_max - sm_min) / sm_step
    if steps > 0:
        max_load = 1.0 - 0.5 / steps - 1e-9
    else:
        max_load = 1.0
    arr = np.clip(arr, 0.0, max_load)
    target = sm_min + arr * float(sm_max - sm_min)
    q = np.rint((target - sm_min) / sm_step)
    sm = sm_min + q * sm_step
    sm = np.clip(sm, sm_min, sm_max)
    return sm.astype(np.int64)


def generate_random_load(
    n: int,
    H: float = 0.85,
    mean_rho: float = 0.40,
    seed: int | None = None,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    noise = generate_fgn(n, H, rng)
    std = float(np.std(noise))
    if std < 1e-12:
        norm = np.zeros_like(noise)
    else:
        norm = noise / std
    load = mean_rho + 0.18 * norm
    return np.clip(load, 0.0, 1.0)


def generate_shaped_load(
    n: int,
    H: float = 0.85,
    mean_rho: float = 0.40,
    alpha: float = 0.15,
    slew_cap: float = 0.08,
    seed: int | None = None,
) -> NDArray[np.float64]:
    raw = generate_random_load(n=n, H=H, mean_rho=mean_rho, seed=seed)
    if n <= 0:
        return raw
    shaped = np.empty_like(raw)
    state = float(raw[0])
    shaped[0] = state
    deadband = max(0.04, 0.5 * slew_cap)
    for i in range(1, n):
        ewma = alpha * float(raw[i]) + (1.0 - alpha) * state
        delta = np.clip(ewma - state, -slew_cap, slew_cap)
        next_state = float(np.clip(state + delta, 0.0, 1.0))
        if abs(next_state - state) < deadband:
            next_state = state
        state = next_state
        shaped[i] = state
    return shaped


def generate_timing(
    n: int,
    scs: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    preset = SCS_PRESETS[scs]
    slot_ns_nom = int(preset["slot_ns"])
    bmin_us = int(preset["budget_min_us"])
    bmax_us = int(preset["budget_max_us"])

    sched = np.empty(n, dtype=np.int64)
    decode = np.empty(n, dtype=np.int64)

    t = 0
    for i in range(n):
        sched[i] = t
        budget_us = int(rng.integers(bmin_us, bmax_us + 1))
        decode[i] = t + budget_us * 1_000

        jitter = float(rng.uniform(-0.02, 0.02))
        slot_ns = max(1, int(round(slot_ns_nom * (1.0 + jitter))))
        t += slot_ns

    return sched, decode


def _sm_step_from_pynvml() -> int | None:
    try:
        pynvml = importlib.import_module("pynvml")

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count <= 0:
            pynvml.nvmlShutdown()
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, _minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()
        return 8 if int(major) >= 9 else 2
    except Exception:
        return None


def _sm_step_from_nvidia_smi() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = proc.stdout.strip().splitlines()[0].strip()
        major = int(line.split(".")[0])
        return 8 if major >= 9 else 2
    except Exception:
        return None


def detect_sm_step() -> int:
    step = _sm_step_from_pynvml()
    if step is not None:
        return step
    step = _sm_step_from_nvidia_smi()
    if step is not None:
        return step
    return 2


def _stability_runs(sm: NDArray[np.int64]) -> list[int]:
    if sm.size == 0:
        return []
    runs: list[int] = []
    run = 1
    for i in range(1, sm.size):
        if sm[i] == sm[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic RAN scheduling traces for bench_trace_switch"
    )
    parser.add_argument(
        "-n",
        "--num-slots",
        type=int,
        default=10_000,
        help="Number of slots to generate",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["random", "shaped"],
        default="random",
        help="Traffic mode",
    )
    parser.add_argument(
        "--scs",
        type=int,
        choices=[15, 30, 60, 120],
        default=30,
        help="Subcarrier spacing preset (kHz)",
    )
    parser.add_argument(
        "--sm-min", type=int, default=8, help="Minimum SM count"
    )
    parser.add_argument(
        "--sm-max", type=int, default=64, help="Maximum SM count"
    )
    parser.add_argument(
        "--sm-step",
        default="auto",
        help="SM quantization step (auto or integer)",
    )
    parser.add_argument(
        "--hurst", type=float, default=0.85, help="Hurst parameter"
    )
    parser.add_argument(
        "--mean-load",
        type=float,
        default=0.40,
        help="Mean normalized load rho",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="EWMA alpha for shaped mode",
    )
    parser.add_argument(
        "--slew",
        type=float,
        default=0.08,
        help="Per-slot slew cap for shaped mode",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "-o",
        "--output",
        default="data/trace_synthetic.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main_ran() -> None:
    args = parse_args()

    if args.num_slots <= 0:
        raise ValueError("--num-slots must be > 0")
    if not (0.0 < args.hurst < 1.0):
        raise ValueError("--hurst must be in (0, 1)")
    if not (0.0 <= args.mean_load <= 1.0):
        raise ValueError("--mean-load must be in [0, 1]")
    if args.sm_min <= 0 or args.sm_max < args.sm_min:
        raise ValueError("Require 0 < --sm-min <= --sm-max")

    if str(args.sm_step).lower() == "auto":
        sm_step = detect_sm_step()
    else:
        sm_step = int(args.sm_step)
    if sm_step <= 0:
        raise ValueError("--sm-step must be positive")

    rng = np.random.default_rng(args.seed)

    if args.mode == "random":
        load = generate_random_load(
            n=args.num_slots,
            H=args.hurst,
            mean_rho=args.mean_load,
            seed=args.seed,
        )
    else:
        load = generate_shaped_load(
            n=args.num_slots,
            H=args.hurst,
            mean_rho=args.mean_load,
            alpha=args.alpha,
            slew_cap=args.slew,
            seed=args.seed,
        )

    sched_ns, decode_ns = generate_timing(args.num_slots, args.scs, rng)
    sm_count = load_to_sm(load, args.sm_min, args.sm_max, sm_step)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["time_slot_sched_ns", "time_decode_start_est_ns", "sm_count"]
        )
        writer.writerows(
            zip(sched_ns.tolist(), decode_ns.tolist(), sm_count.tolist())
        )

    transitions = int(np.count_nonzero(sm_count[1:] != sm_count[:-1]))
    total_edges = max(1, args.num_slots - 1)
    transition_rate = transitions / total_edges
    budgets_us = (decode_ns - sched_ns) / 1_000.0
    unique_sms = np.unique(sm_count)

    print(f"Mode: {args.mode}")
    print(f"SCS: {args.scs} kHz")
    print(
        f"Unique SMs: {unique_sms.size} "
        f"(min={int(unique_sms.min())}, max={int(unique_sms.max())})"
    )
    print(
        f"SM transitions: {transitions}/{total_edges} "
        f"({100.0 * transition_rate:.2f}%)"
    )
    print(f"Budget range (us): {budgets_us.min():.0f}..{budgets_us.max():.0f}")

    if args.mode == "shaped":
        runs = _stability_runs(sm_count)
        mean_run = float(np.mean(runs)) if runs else 0.0
        print(
            "Stability runs: "
            f"count={len(runs)}, mean={mean_run:.2f}, max={max(runs) if runs else 0}"
        )

    print(f"Wrote: {out_path}")


def _drift_series(
    n: int, hurst: float, std: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    noise = generate_fgn(n, hurst, rng)
    s = float(np.std(noise))
    norm = noise / s if s > 1e-12 else np.zeros_like(noise)
    return np.clip(1.0 + std * norm, 0.2, 1.2)


def generate_within_batch_traces(
    num_devices: int,
    num_bins: int,
    magnitude: float,
    event_rate: float,
    duration_frac: float,
    network_drift_std: float,
    hurst: float,
    seed: int,
) -> dict:
    """Per-device within-batch time-varying compute and network multipliers.

    The batch is divided into ``num_bins`` equal virtual-time bins. For each
    device, ``flops_mult`` models thermal-throttle / foreground-contention
    events (Poisson-arrival count ``event_rate``, each of depth ``magnitude``
    and length ``duration_frac`` of the batch); ``ul_bw_mult`` / ``dl_bw_mult``
    model network-rate drift as fractional Gaussian noise (std
    ``network_drift_std``) with an added dip while a device is throttled
    (contention couples compute and link). Multipliers lie in (0, ~1.2]; the
    consumer divides a phase's nominal duration by the multiplier.
    """
    rng = np.random.default_rng(seed)
    dur_bins = max(1, int(round(duration_frac * num_bins)))
    devices: dict[str, dict[str, list[float]]] = {}

    for dev in range(num_devices):
        flops = np.ones(num_bins, dtype=np.float64)
        n_events = int(rng.poisson(event_rate))
        for _ in range(n_events):
            start = int(rng.integers(0, num_bins))
            depth = magnitude * float(rng.uniform(0.7, 1.0))
            mult = max(0.1, 1.0 - depth)
            end = min(num_bins, start + dur_bins)
            flops[start:end] = np.minimum(flops[start:end], mult)

        ul = _drift_series(num_bins, hurst, network_drift_std, rng)
        dl = _drift_series(num_bins, hurst, network_drift_std, rng)
        contention = flops < 1.0
        ul[contention] *= 0.7
        dl[contention] *= 0.7

        devices[str(dev)] = {
            "flops_mult": [round(float(x), 6) for x in flops],
            "ul_bw_mult": [round(float(x), 6) for x in ul],
            "dl_bw_mult": [round(float(x), 6) for x in dl],
        }

    return {
        "schema": "within_batch_trace_v1",
        "num_bins": num_bins,
        "meta": {
            "num_devices": num_devices,
            "magnitude": magnitude,
            "event_rate": event_rate,
            "duration_frac": duration_frac,
            "network_drift_std": network_drift_std,
            "hurst": hurst,
            "seed": seed,
        },
        "devices": devices,
    }


def validate_within_batch(trace: dict) -> dict:
    meta = trace["meta"]
    devices = trace["devices"]
    flops = np.array([devices[k]["flops_mult"] for k in devices])
    ul = np.array([devices[k]["ul_bw_mult"] for k in devices])

    throttled_fraction = float(np.mean(flops < 1.0))
    expected_fraction = float(
        1.0 - np.exp(-float(meta["event_rate"]) * float(meta["duration_frac"]))
    )
    min_flops_mult = float(flops.min())
    network_std = float(np.std(ul))

    stats = {
        "throttled_bin_fraction": throttled_fraction,
        "expected_throttled_fraction": expected_fraction,
        "min_flops_mult": min_flops_mult,
        "network_mult_std": network_std,
        "requested_network_std": float(meta["network_drift_std"]),
    }

    if float(meta["magnitude"]) > 0 and float(meta["event_rate"]) > 0:
        assert min_flops_mult <= 1.0 - 0.5 * float(meta["magnitude"]), (
            f"throttle depth too shallow: min_mult={min_flops_mult}"
        )
        tol = max(0.15, 0.6 * expected_fraction)
        assert abs(throttled_fraction - expected_fraction) <= tol, (
            f"event coverage off: got {throttled_fraction}, "
            f"expected ~{expected_fraction}"
        )
    if float(meta["network_drift_std"]) > 0:
        assert network_std >= 0.3 * float(meta["network_drift_std"]), (
            f"network drift too small: std={network_std}"
        )
    return stats


def parse_within_batch_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_trace.py within-batch",
        description="Generate per-device within-batch time-varying "
        "compute/network multiplier traces for the C2 experiment",
    )
    parser.add_argument("--num-devices", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--magnitude", type=float, default=0.5)
    parser.add_argument("--event-rate", type=float, default=3.0)
    parser.add_argument("--duration-frac", type=float, default=0.15)
    parser.add_argument("--network-drift-std", type=float, default=0.15)
    parser.add_argument("--hurst", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default under results/within_batch/traces/)",
    )
    return parser.parse_args(argv)


def main_within_batch(argv: list[str]) -> None:
    args = parse_within_batch_args(argv)

    if args.num_devices <= 0 or args.num_bins <= 0:
        raise ValueError("--num-devices and --num-bins must be > 0")
    if not (0.0 <= args.magnitude <= 1.0):
        raise ValueError("--magnitude must be in [0, 1]")
    if not (0.0 < args.hurst < 1.0):
        raise ValueError("--hurst must be in (0, 1)")
    if not (0.0 <= args.duration_frac <= 1.0):
        raise ValueError("--duration-frac must be in [0, 1]")

    trace = generate_within_batch_traces(
        num_devices=args.num_devices,
        num_bins=args.num_bins,
        magnitude=args.magnitude,
        event_rate=args.event_rate,
        duration_frac=args.duration_frac,
        network_drift_std=args.network_drift_std,
        hurst=args.hurst,
        seed=args.seed,
    )
    stats = validate_within_batch(trace)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(
            "results/within_batch/traces/"
            f"within_batch_seed{args.seed}_mag{args.magnitude}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(trace, f)

    print(f"Devices: {args.num_devices}, bins: {args.num_bins}")
    print(
        f"Throttled bin fraction: {stats['throttled_bin_fraction']:.3f} "
        f"(expected ~{stats['expected_throttled_fraction']:.3f})"
    )
    print(f"Min flops mult: {stats['min_flops_mult']:.3f}")
    print(
        f"Network mult std: {stats['network_mult_std']:.3f} "
        f"(requested {stats['requested_network_std']:.3f})"
    )
    print(f"Wrote: {out_path}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "within-batch":
        main_within_batch(sys.argv[2:])
    else:
        main_ran()


if __name__ == "__main__":
    main()
