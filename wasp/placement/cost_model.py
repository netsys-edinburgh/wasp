"""Shared DP+PP cost helpers for edge-training baselines.

Canonical source: dtfm_solver.py:59-98.
Used by: asteroid_solver.py, confident_solver.py.
NOT used by: alpa_dag_solver.py (intentionally different TP-aware cost model).

Units convention (matching arguments.py __post_init__):
  - DEVICE_FLOPS: TFLOPS (e.g. 5-7)  [note: dtfm_solver receives args.device_flops * 1000]
  - DL_BW / UL_BW: MB/ms (e.g. 0.01-0.1)
  - DL_LATENCY / UL_LATENCY: ms (but typically 0 in current configs)
  - layer_size: GB
  - activation_size: GB
  - All returned costs: seconds
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wasp.placement.constants import KB, MB, GB, TB, BYTE_SIZE


def compute_layer_flops(B: int, S: int, d_model: int, d_ffn: int) -> float:
    """Per-layer FLOPS including forward + backward (3x forward).

    Breakdown:
      4 * B*S * d_model^2  = Q, K, V, O projections
      2 * B*S * d_model * d_ffn = MLP fc1 + fc2
      2 * B * d_model * S^2 = attention score matmul
      * 3 = forward + grad_input + grad_weight
    """
    NUM_TOKENS = B * S
    return float(
        (4 * NUM_TOKENS * d_model ** 2
         + 2 * NUM_TOKENS * d_model * d_ffn
         + 2 * B * d_model * S ** 2) * 3
    )


def compute_model_flops(B: int, S: int, d_model: int, d_ffn: int, n_layer: int) -> float:
    """Total model FLOPS including activation recompute (1.5x).
    Source: dtfm_solver.py:65."""
    return compute_layer_flops(B, S, d_model, d_ffn) * n_layer * 1.5


def build_peer_matrices(
    num_devices: int,
    UL_BW: NDArray[np.floating],
    DL_BW: NDArray[np.floating],
    UL_LATENCY: NDArray[np.floating],
    DL_LATENCY: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert asymmetric per-device DL/UL arrays to NxN peer matrices.

    Returns:
        delay: (N, N) in ms  (dtfm_solver.py:94: delay * 1000)
        bandwidth: (N, N) in Gbps  (dtfm_solver.py:95: bandwidth * 8)
    """
    delay = np.zeros((num_devices, num_devices))
    bandwidth = np.zeros((num_devices, num_devices))

    for i in range(num_devices):
        for j in range(i, num_devices):
            delay[i, j] = delay[j, i] = UL_LATENCY[i] + DL_LATENCY[j]
            bandwidth[i, j] = min(UL_BW[i], DL_BW[j])
            bandwidth[j, i] = min(UL_BW[j], DL_BW[i])

    delay_ms = delay * 1000
    bandwidth_gbps = bandwidth * 8
    return delay_ms, bandwidth_gbps


def compute_activation_size_gb(
    B: int, S: int, d_model: int, byte_size: int = BYTE_SIZE,
) -> float:
    """Inter-stage activation tensor size in GB.
    Source: dtfm_solver.py:74-76 (send_activation_size)."""
    return (B * S * d_model * byte_size) / GB


def compute_gradient_size_gb(
    layer_size_gb: float, n_assigned_layers: int, dp_size: int,
) -> float:
    """DP gradient payload per AllReduce in GB.
    Source: dtfm_solver.py:77-78 (send_gradient_size = layer_size * n_layer / way)."""
    if dp_size <= 1:
        return 0.0
    return layer_size_gb * n_assigned_layers / dp_size


def ring_allreduce_cost_s(
    data_size_gb: float,
    min_bandwidth_gbps: float,
    group_size: int,
) -> float:
    """Ring AllReduce time in seconds.
    ring_factor = 2*(N-1)/N, transfer = data / bandwidth."""
    if group_size <= 1 or data_size_gb <= 0.0:
        return 0.0
    ring_factor = 2.0 * (group_size - 1) / group_size
    bw = max(min_bandwidth_gbps, 1e-12)
    return ring_factor * data_size_gb / bw


def p2p_transfer_cost_s(
    data_size_gb: float,
    link_bandwidth_gbps: float,
    link_delay_ms: float,
) -> float:
    """Point-to-point transfer time in seconds."""
    bw = max(link_bandwidth_gbps, 1e-12)
    return data_size_gb / bw + link_delay_ms / 1000.0


def estimate_stage_memory_gb(
    layer_size_gb: float,
    n_layers: int,
    activation_size_gb_per_layer: float,
    micro_batch_size: int,
    stage_idx: int,
    num_stages: int,
) -> float:
    """Estimate per-stage peak memory in GB.

    Mem = weights + optimizer_states + K_p * activations
    where K_p = max(1, 2*(P - p) - 1) for 1F1B schedule.
    Source: asteroid_strategy.py:424-432 (_memory_footprint).
    """
    weights_gb = layer_size_gb * n_layers
    optimizer_gb = weights_gb * 2.0
    k_p = max(1, 2 * (num_stages - stage_idx) - 1)
    act_gb = activation_size_gb_per_layer * n_layers * micro_batch_size
    return weights_gb + optimizer_gb + k_p * act_gb


_OFFLINE_MODEL_REGISTRY: dict[str, dict] = {
    "facebook/opt-125m":  {"model_type": "opt",   "d_model": 768,   "d_ffn": 3072,   "n_head": 12,  "n_layer": 12,  "n_kv_head": 0},
    "facebook/opt-350m":  {"model_type": "opt",   "d_model": 1024,  "d_ffn": 4096,   "n_head": 16,  "n_layer": 24,  "n_kv_head": 0},
    "facebook/opt-1.3b":  {"model_type": "opt",   "d_model": 2048,  "d_ffn": 8192,   "n_head": 32,  "n_layer": 24,  "n_kv_head": 0},
    "facebook/opt-6.7b":  {"model_type": "opt",   "d_model": 4096,  "d_ffn": 16384,  "n_head": 32,  "n_layer": 32,  "n_kv_head": 0},
    "facebook/opt-13b":   {"model_type": "opt",   "d_model": 5120,  "d_ffn": 20480,  "n_head": 40,  "n_layer": 40,  "n_kv_head": 0},
    "facebook/opt-30b":   {"model_type": "opt",   "d_model": 7168,  "d_ffn": 28672,  "n_head": 56,  "n_layer": 48,  "n_kv_head": 0},
    "facebook/opt-66b":   {"model_type": "opt",   "d_model": 9216,  "d_ffn": 36864,  "n_head": 72,  "n_layer": 64,  "n_kv_head": 0},
    "meta-llama/Llama-2-7b-hf":  {"model_type": "llama", "d_model": 4096,  "d_ffn": 11008,  "n_head": 32,  "n_layer": 32,  "n_kv_head": 32},
    "meta-llama/Llama-2-13b-hf": {"model_type": "llama", "d_model": 5120,  "d_ffn": 13824,  "n_head": 40,  "n_layer": 40,  "n_kv_head": 40},
    "meta-llama/Llama-2-70b-hf": {"model_type": "llama", "d_model": 8192,  "d_ffn": 28672,  "n_head": 64,  "n_layer": 80,  "n_kv_head": 8},
}


def get_model_dims(model_name: str) -> dict:
    """Extract model dimensions. Uses offline registry first, falls back to HuggingFace."""
    if model_name in _OFFLINE_MODEL_REGISTRY:
        return dict(_OFFLINE_MODEL_REGISTRY[model_name])
    from wasp.placement.model_utils import parse_model_meta
    return parse_model_meta(model_name)


def get_layer_size_gb(model_name: str) -> float:
    """Get per-layer static memory in GB.
    Analytical: (4*d² + 2*d*d_ff) * byte_size / GB.
    Source: dtfm_solver.py:59."""
    meta = get_model_dims(model_name)
    d, d_ff = int(meta["d_model"]), int(meta["d_ffn"])
    return (4 * d * d + 2 * d * d_ff) * BYTE_SIZE / GB
