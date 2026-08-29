import os
import heuristic_evolutionary_solver
from heuristic_evolutionary_solver.scheduler import (
    GCMA,
    compute_data_parallel_cost,
    compute_pipeline_parallel_cost,
    get_pipelines,
    # compute_data_parallel_cost_ring,
)

import time
import numpy as np
from wasp.placement.transformers_utils import *
from wasp.placement.constants import *
from wasp.placement.plot_utils import plot_heatmap
from wasp.placement.arguments import parse_args
from wasp.placement.model_utils import parse_model_meta

def solve_DTFM(
    model,
    num_devices,
    B,
    S,
    DL_BW,
    UL_BW,
    DEVICE_FLOPS,
    DL_LATENCY,
    UL_LATENCY,
):

    NUM_TOKENS = B * S
            
    all_divisors = get_divisors(num_devices)
    
    config = HFUnifiedConfig(model)
    
    n_layer = config.num_layers
    d_model = config.hidden_size
    d_ffn = config.mlp_size

    p_idx = 0
    partition_size = all_divisors[p_idx]
    # way = num_devices
    while num_devices // partition_size > n_layer:
        p_idx += 1
        partition_size = all_divisors[p_idx]
    while True:
        try:
            way = num_devices // partition_size
            dp_table = np.full(shape=(way, pow(2, way)), fill_value=np.inf)
            break
        except MemoryError:
            p_idx += 1
            partition_size = all_divisors[p_idx]

    print(f"{way=} {partition_size=} {n_layer=}")
    assert way * partition_size == num_devices

    layer_size = static_layer_memory_usage(config) / GB
    layer_flops = (
        4 * NUM_TOKENS * d_model**2
        + 2 * NUM_TOKENS * d_model * d_ffn
        + 2 * B * d_model * S**2
    ) * 3 # include backprop
    model_flops = layer_flops * n_layer * 1.5  # include activation recompute

    print(f"model: {model_flops / TB} TFLOPS")
    print(f"layer: {layer_flops / TB} TFLOPS")
    # exit()
    heuristic_evolutionary_solver.scheduler.num_devices = num_devices
    heuristic_evolutionary_solver.scheduler.way = way
    heuristic_evolutionary_solver.scheduler.partition_size = partition_size
    heuristic_evolutionary_solver.scheduler.layer_size = layer_size
    heuristic_evolutionary_solver.scheduler.send_activation_size = (
        (NUM_TOKENS * d_model * BYTE_SIZE) / GB / partition_size # way
    )
    heuristic_evolutionary_solver.scheduler.send_gradient_size = send_gradient_size = (
        layer_size * n_layer / way # layer_size * n_layer / way * partition_size
    )
    print(f"{layer_size=} {n_layer=}")
    print(f"{heuristic_evolutionary_solver.scheduler.send_gradient_size=}")

    def simulate_7_edge_distributed(nodes=64):
        print("Simulate case 7: edge distributed")

        delay = np.zeros((nodes, nodes))
        bandwidth = np.zeros((nodes, nodes))

        for i in range(nodes):
            for j in range(i, nodes):
                delay[i, j] = delay[j, i] = UL_LATENCY[i] + DL_LATENCY[j]
                bandwidth[i, j] = min(UL_BW[i], DL_BW[j])
                bandwidth[j, i] = min(UL_BW[j], DL_BW[i])
        delay = delay * 1000
        bandwidth = bandwidth * 8
        print("delay(ms):", delay)
        print("bandwidth(Gbps):", bandwidth)
        return delay, bandwidth, None

    simulate_cases = [(7, simulate_7_edge_distributed)]

    repetition = 0
    for _, (case_idx, simulate_case) in enumerate(simulate_cases):
        peer_delay, peer_bandwidth, regions = simulate_case(num_devices)

        # change variable in lib
        heuristic_evolutionary_solver.scheduler.peer_delay = peer_delay
        heuristic_evolutionary_solver.scheduler.peer_bandwidth = peer_bandwidth
        heuristic_evolutionary_solver.scheduler.num_devices = num_devices

        start = time.perf_counter()
        min_total_cost = float("inf")
        candidate_partition = None
        data_parallel_cost = None
        pipeline_parallel_cost = None
        pipeline_parallel_path = None
        pipeline_parallel_match = None

        candidate_partitions, all_cost_records, min_cost_records = GCMA(
            nodes=list(range(num_devices)), population_size=1, trails=1, mode="default"
        )
        candidate_partition_idx = np.argmin(all_cost_records)
        candidate_partition = [
            candidate_partitions[candidate_partition_idx][i : i + partition_size]
            for i in range(0, num_devices, partition_size)
        ]
        
        
        data_parallel_cost = compute_data_parallel_cost(
            candidate_partition=candidate_partition
        )

        pipeline_parallel_cost, pipeline_parallel_path, pipeline_parallel_match = (
            compute_pipeline_parallel_cost(candidate_partition)
        )

        per_way_bs = B / way
        dp_comm_cost_total = data_parallel_cost # / 2
        pp_comm_cost_total = (
            pipeline_parallel_cost / per_way_bs * (per_way_bs - 1)
            + pipeline_parallel_cost / per_way_bs * partition_size
        ) * 2
        comp_cost_total = (
            model_flops / TB / num_devices / np.min(DEVICE_FLOPS)
        )  # * (per_way_bs-1 + partition_size)
        print(f"{np.mean(DEVICE_FLOPS)=}")
        min_total_cost = dp_comm_cost_total + pp_comm_cost_total + comp_cost_total

        end = time.perf_counter()
        print(
            "run time("
            + str(len(all_cost_records))
            + " candidates): "
            + str(end - start)
            + " seconds"
        )
        print("candidate partition: " + str(candidate_partition))
        print("pipeline parallel path: " + str(pipeline_parallel_path))
        print("total cost: " + str(min_total_cost))
        print("data parallel cost: " + str(data_parallel_cost))
        print("pipeline parallel cost: " + str(pipeline_parallel_cost))
        print("data parallel communication cost: " + str(dp_comm_cost_total))
        print("pipeline parallel communication cost: " + str(pp_comm_cost_total))
        print("computation cost: " + str(comp_cost_total))
        
        return min_total_cost

args = parse_args()
model_meta = parse_model_meta(args.model)

cost = solve_DTFM(
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