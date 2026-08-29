import re
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from wasp.placement.constants import *
from wasp.placement.arguments import parse_args


args = parse_args()
model_meta = args.model_meta

d_model = model_meta["d_model"]
d_ffn = model_meta["d_ffn"]
n_head = model_meta["n_head"]
n_layer = model_meta["n_layer"]

test_cycles = {
    "mlp-gate": (args.num_tokens, d_model, d_ffn),
    "mlp-proj": (args.num_tokens, d_ffn, d_model),
    "qkvo": (args.num_tokens, d_model, d_model),
    "attn": (args.num_tokens * n_head, d_model // n_head, d_model // n_head),
    "attn_proj": (args.num_tokens * n_head, args.seq_length, d_model // n_head),
}

def solve_Alpa(name, in_dim, h_dim, out_dim, num_devices, dl_bw, ul_bw, device_flops, dl_latency, ul_latency, cache_input=False):
    input_matrix_size = in_dim * h_dim
    weight_matrix_size = h_dim * out_dim
    
    def device_lat(M, N):
        dl_lat = (M * h_dim + N * h_dim) * args.byte_size / MB / args.dl_bw
        if cache_input:
            dl_lat = N * h_dim * args.byte_size / MB / args.dl_bw
        ul_lat = N * M * args.byte_size / MB / args.ul_bw
        device_lat = 2 * M * N * h_dim / TB / args.device_flops
                
        return  np.max(dl_lat + ul_lat + device_lat) / 1000
    
    divider = np.sqrt(num_devices)
    M = in_dim / divider
    N = out_dim / divider
    lat_tp = device_lat(M, N)
    
    M = in_dim / num_devices
    N = out_dim
    lat_input_replica = device_lat(M, N)
    
    M = in_dim
    N = out_dim / num_devices
    lat_weight_replica = device_lat(M, N)
    
    return min(lat_tp, lat_input_replica, lat_weight_replica)

test_results = {}
for cycle, (in_dim, h_dim, out_dim) in test_cycles.items():
    print(f"Cycle {cycle}")
    
    cache_input = "gate" in cycle or "qkvo" in cycle or "weight" in cycle
    device_latency = solve_Alpa(
        cycle,
        in_dim,
        h_dim,
        out_dim,
        args.num_devices,
        args.dl_bw,
        args.ul_bw,
        args.device_flops,
        args.dl_lat,
        args.ul_lat,
        cache_input=cache_input,
    )
    
    test_results[cycle] = np.max(device_latency)

qkv_parallel = 0    
model_forward_cost = 0
for cycle, cost in test_results.items():
    if cycle == "qkvo":
        model_forward_cost += cost * (4 - qkv_parallel)
    # elif cycle == "mlp-proj" and model_type == "llama":
    #     model_forward_cost += cost * (2 - mlp_parallel)
    else:
        model_forward_cost += cost
        
print(f"{model_forward_cost=}")
model_forward_cost *= n_layer
model_bachward_cost = model_forward_cost * 2
model_step_cost = model_forward_cost + model_bachward_cost

print(f"{model_step_cost=}s")