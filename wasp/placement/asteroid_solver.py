from importlib import import_module
from typing import Any

import numpy as np

from wasp.placement.constants import TB


def _to_array(x, n):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    if arr.shape[0] != n:
        raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
    return arr


def _group_min_bw_gbps(group, bandwidth_gbps):
    if len(group) <= 1:
        return np.inf
    min_bw = np.inf
    for i in group:
        for j in group:
            if i == j:
                continue
            min_bw = min(min_bw, float(bandwidth_gbps[i, j]))
    return min_bw


def _inter_stage_link(group_a, group_b, delay_ms, bandwidth_gbps):
    min_bw = np.inf
    max_delay = 0.0
    for i in group_a:
        for j in group_b:
            min_bw = min(min_bw, float(bandwidth_gbps[i, j]))
            max_delay = max(max_delay, float(delay_ms[i, j]))
    return min_bw, max_delay


def _device_stage_exec_s(n_layers, local_bs, S, d_model, d_ffn, device_flops, compute_layer_flops_fn):
    if local_bs <= 0 or n_layers <= 0:
        return 0.0
    layer_s = compute_layer_flops_fn(local_bs, S, d_model, d_ffn) / max(float(device_flops) * TB, 1e-12)
    return float(layer_s * n_layers)


def _alloc_microbatches_greedy(
    stage_idx,
    num_stages,
    device_group,
    start_l,
    end_l,
    micro_bs,
    S,
    d_model,
    d_ffn,
    device_flops,
    layer_size_gb,
    act_size_per_sample_gb,
    memory_budget_mb,
    estimate_stage_memory_gb_fn,
    compute_layer_flops_fn,
):
    if not device_group:
        return None, np.inf

    n_layers = max(0, end_l - start_l)
    if n_layers <= 0:
        return None, np.inf

    alloc = {did: 0 for did in device_group}
    remaining = int(max(1, micro_bs))
    memory_budget_gb = float(memory_budget_mb) / 1024.0

    while remaining > 0:
        best_id = None
        best_cost = np.inf
        for did in device_group:
            next_bs = alloc[did] + 1
            mem_gb = estimate_stage_memory_gb_fn(
                layer_size_gb,
                n_layers,
                act_size_per_sample_gb,
                next_bs,
                stage_idx,
                num_stages,
            )
            if mem_gb > memory_budget_gb:
                continue
            projected = _device_stage_exec_s(
                n_layers,
                next_bs,
                S,
                d_model,
                d_ffn,
                device_flops[did],
                compute_layer_flops_fn,
            )
            if projected < best_cost:
                best_cost = projected
                best_id = did

        if best_id is None:
            break
        alloc[best_id] += 1
        remaining -= 1

    if remaining > 0:
        return None, np.inf

    active = {did: bs for did, bs in alloc.items() if bs > 0}
    if not active:
        return None, np.inf

    straggler = max(
        _device_stage_exec_s(
            n_layers,
            bs,
            S,
            d_model,
            d_ffn,
            device_flops[did],
            compute_layer_flops_fn,
        )
        for did, bs in active.items()
    )
    return alloc, float(straggler)


def solve_Asteroid(
    model,
    num_devices,
    B,
    S,
    DL_BW,
    UL_BW,
    DEVICE_FLOPS,
    DL_LATENCY,
    UL_LATENCY,
    memory_budget_mb=32768,
    num_microbatches=None,
):
    cost_model = import_module("solver.cost_model")
    build_peer_matrices = cost_model.build_peer_matrices
    compute_activation_size_gb = cost_model.compute_activation_size_gb
    compute_gradient_size_gb = cost_model.compute_gradient_size_gb
    compute_layer_flops = cost_model.compute_layer_flops
    estimate_stage_memory_gb = cost_model.estimate_stage_memory_gb
    get_layer_size_gb = cost_model.get_layer_size_gb
    get_model_dims = cost_model.get_model_dims
    p2p_transfer_cost_s = cost_model.p2p_transfer_cost_s
    ring_allreduce_cost_s = cost_model.ring_allreduce_cost_s

    model_meta = get_model_dims(model)
    d_model = int(model_meta["d_model"])
    d_ffn = int(model_meta["d_ffn"])
    num_layers = max(1, int(model_meta["n_layer"]))

    n = int(num_devices)
    if n <= 0:
        return 0.0

    device_flops = _to_array(DEVICE_FLOPS, n)
    ul_bw = _to_array(UL_BW, n)
    dl_bw = _to_array(DL_BW, n)
    ul_lat = _to_array(UL_LATENCY, n)
    dl_lat = _to_array(DL_LATENCY, n)

    delay_ms, bandwidth_gbps = build_peer_matrices(n, ul_bw, dl_bw, ul_lat, dl_lat)

    if num_microbatches is None:
        num_microbatches = max(1, int(B) // 4)
    num_microbatches = int(max(1, num_microbatches))
    micro_bs = int(max(1, int(np.ceil(float(B) / float(num_microbatches)))))

    layer_size_gb = float(get_layer_size_gb(model))
    act_size_per_sample_gb = float(compute_activation_size_gb(1, int(S), d_model))

    device_ids = list(range(n))
    max_stages = min(8, n, num_layers)
    inf = float("inf")
    best_total_cost = inf

    for num_stages in range(1, max_stages + 1):
        latency = np.full((num_layers + 1, n + 1, num_stages + 1), inf, dtype=np.float64)
        config: list[list[list[Any]]] = [
            [[None for _ in range(num_stages + 1)] for _ in range(n + 1)]
            for _ in range(num_layers + 1)
        ]

        for layers_tail in range(1, num_layers + 1):
            for devices_tail in range(1, n + 1):
                group = device_ids[n - devices_tail :]
                start = num_layers - layers_tail
                end = num_layers
                alloc, stage_exec = _alloc_microbatches_greedy(
                    stage_idx=num_stages - 1,
                    num_stages=num_stages,
                    device_group=group,
                    start_l=start,
                    end_l=end,
                    micro_bs=micro_bs,
                    S=int(S),
                    d_model=d_model,
                    d_ffn=d_ffn,
                    device_flops=device_flops,
                    layer_size_gb=layer_size_gb,
                    act_size_per_sample_gb=act_size_per_sample_gb,
                    memory_budget_mb=memory_budget_mb,
                    estimate_stage_memory_gb_fn=estimate_stage_memory_gb,
                    compute_layer_flops_fn=compute_layer_flops,
                )
                if alloc is None:
                    continue

                grad_size_gb = compute_gradient_size_gb(layer_size_gb, end - start, len(group))
                min_bw = _group_min_bw_gbps(group, bandwidth_gbps)
                allreduce = ring_allreduce_cost_s(grad_size_gb, min_bw, len(group))
                stage_cost = stage_exec + allreduce

                latency[layers_tail, devices_tail, 1] = stage_cost
                config[layers_tail][devices_tail][1] = {
                    "ranges": [(start, end)],
                    "groups": [group],
                    "allocs": [alloc],
                }

        for stages_tail in range(2, num_stages + 1):
            for layers_tail in range(stages_tail, num_layers + 1):
                for devices_tail in range(stages_tail, n + 1):
                    for prev_layers_tail in range(stages_tail - 1, layers_tail):
                        for prev_devices_tail in range(stages_tail - 1, devices_tail):
                            prev_latency = float(latency[prev_layers_tail, prev_devices_tail, stages_tail - 1])
                            if not np.isfinite(prev_latency):
                                continue

                            group = device_ids[n - devices_tail : n - prev_devices_tail]
                            if not group:
                                continue

                            start = num_layers - layers_tail
                            end = num_layers - prev_layers_tail
                            stage_idx = num_stages - stages_tail

                            alloc, stage_exec = _alloc_microbatches_greedy(
                                stage_idx=stage_idx,
                                num_stages=num_stages,
                                device_group=group,
                                start_l=start,
                                end_l=end,
                                micro_bs=micro_bs,
                                S=int(S),
                                d_model=d_model,
                                d_ffn=d_ffn,
                                device_flops=device_flops,
                                layer_size_gb=layer_size_gb,
                                act_size_per_sample_gb=act_size_per_sample_gb,
                                memory_budget_mb=memory_budget_mb,
                                estimate_stage_memory_gb_fn=estimate_stage_memory_gb,
                                compute_layer_flops_fn=compute_layer_flops,
                            )
                            if alloc is None:
                                continue

                            prev_cfg = config[prev_layers_tail][prev_devices_tail][stages_tail - 1]
                            if prev_cfg is None:
                                continue
                            next_group = prev_cfg["groups"][0]

                            data_size_gb = compute_activation_size_gb(int(sum(alloc.values())), int(S), d_model)
                            link_bw, link_delay = _inter_stage_link(group, next_group, delay_ms, bandwidth_gbps)
                            inter_stage = p2p_transfer_cost_s(data_size_gb, link_bw, link_delay)

                            grad_size_gb = compute_gradient_size_gb(layer_size_gb, end - start, len(group))
                            min_bw = _group_min_bw_gbps(group, bandwidth_gbps)
                            allreduce = ring_allreduce_cost_s(grad_size_gb, min_bw, len(group))

                            stage_cost = stage_exec + inter_stage + allreduce
                            total = max(prev_latency, stage_cost)

                            if total < latency[layers_tail, devices_tail, stages_tail]:
                                latency[layers_tail, devices_tail, stages_tail] = total
                                config[layers_tail][devices_tail][stages_tail] = {
                                    "ranges": [(start, end)] + prev_cfg["ranges"],
                                    "groups": [group] + prev_cfg["groups"],
                                    "allocs": [alloc] + prev_cfg["allocs"],
                                }

        best_bottleneck = float(latency[num_layers, n, num_stages])
        if not np.isfinite(best_bottleneck):
            continue

        total_cost = (num_microbatches + num_stages - 1) * best_bottleneck
        if total_cost < best_total_cost:
            best_total_cost = total_cost

    return float(best_total_cost)


if __name__ == "__main__":
    from wasp.placement.arguments import parse_args

    args = parse_args()
    cost = solve_Asteroid(
        args.model,
        args.num_devices,
        args.batch_size,
        args.seq_length,
        args.dl_bw,
        args.ul_bw,
        args.device_flops * 1000,
        args.dl_lat,
        args.ul_lat,
        memory_budget_mb=getattr(args, "memory_budget_mb", 32768),
    )
    print("Objective value: ", cost, "s")
