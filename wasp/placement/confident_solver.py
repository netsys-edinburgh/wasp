from collections.abc import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray

from wasp.placement.constants import TB
from wasp.placement.cost_model import (
    build_peer_matrices,
    compute_activation_size_gb,
    compute_layer_flops,
    get_model_dims,
    p2p_transfer_cost_s,
)


def solve_Confident(
    model: str,
    num_devices: int,
    B: int,
    S: int,
    DL_BW: Union[Sequence[float], NDArray[np.float64]],
    UL_BW: Union[Sequence[float], NDArray[np.float64]],
    DEVICE_FLOPS: Union[Sequence[float], NDArray[np.float64]],
    DL_LATENCY: Union[Sequence[float], NDArray[np.float64]],
    UL_LATENCY: Union[Sequence[float], NDArray[np.float64]],
) -> float:

    model_meta = get_model_dims(model)
    d_model = model_meta["d_model"]
    d_ffn = model_meta["d_ffn"]
    num_layers = max(1, int(model_meta["n_layer"]))

    num_stages = min(int(num_devices), num_layers)
    if num_stages <= 0:
        return 0.0

    delay_ms, bandwidth_gbps = build_peer_matrices(
        int(num_devices),
        np.asarray(UL_BW, dtype=np.float64),
        np.asarray(DL_BW, dtype=np.float64),
        np.asarray(UL_LATENCY, dtype=np.float64),
        np.asarray(DL_LATENCY, dtype=np.float64),
    )

    activation_size_gb = compute_activation_size_gb(int(B), int(S), int(d_model))

    layer_flops = compute_layer_flops(int(B), int(S), int(d_model), int(d_ffn))
    stage_prefix = np.zeros((num_stages, num_layers + 1), dtype=np.float64)
    for stage_idx in range(num_stages):
        stage_device = stage_idx
        denom = max(float(DEVICE_FLOPS[stage_device]) * TB, 1e-12)
        layer_compute_time_s = layer_flops / denom
        stage_prefix[stage_idx, 1:] = np.cumsum(
            np.full(num_layers, layer_compute_time_s, dtype=np.float64)
        )

    def range_cost(stage_idx: int, start_layer: int, end_layer: int) -> float:
        return float(stage_prefix[stage_idx, end_layer + 1] - stage_prefix[stage_idx, start_layer])

    dp = np.full((num_layers, num_stages), np.inf, dtype=np.float64)
    split = np.full((num_layers, num_stages), -1, dtype=np.int64)

    for end in range(num_layers):
        dp[end, 0] = range_cost(0, 0, end)

    for stage_idx in range(1, num_stages):
        left_device = stage_idx - 1
        right_device = stage_idx
        comm_s = p2p_transfer_cost_s(
            activation_size_gb,
            float(bandwidth_gbps[left_device, right_device]),
            float(delay_ms[left_device, right_device]),
        )

        for end in range(stage_idx, num_layers):
            best = np.inf
            best_cut = -1
            for cut in range(stage_idx - 1, end):
                stage_time = range_cost(stage_idx, cut + 1, end)
                candidate = max(float(dp[cut, stage_idx - 1]), stage_time + comm_s)
                if candidate < best:
                    best = candidate
                    best_cut = cut
            dp[end, stage_idx] = best
            split[end, stage_idx] = best_cut

    partition_points: list[int] = []
    end = num_layers - 1
    stage_idx = num_stages - 1
    while stage_idx > 0:
        cut = int(split[end, stage_idx])
        if cut < 0:
            partition_points = []
            break
        partition_points.append(cut)
        end = cut
        stage_idx -= 1
    partition_points.reverse()

    bottleneck_s = float(dp[num_layers - 1, num_stages - 1])
    if not np.isfinite(bottleneck_s):
        return float("inf")

    num_microbatches = max(1, int(B) // 4)
    total_cost_s = (num_microbatches + num_stages - 1) * bottleneck_s
    return float(total_cost_s)


if __name__ == "__main__":
    from wasp.placement.arguments import parse_args

    args = parse_args()
    cost = solve_Confident(
        args.model,
        args.num_devices,
        args.batch_size,
        args.seq_length,
        args.dl_bw,
        args.ul_bw,
        args.device_flops * 1000,
        args.dl_lat,
        args.ul_lat,
    )
    print("Objective value: ", cost, "s")
