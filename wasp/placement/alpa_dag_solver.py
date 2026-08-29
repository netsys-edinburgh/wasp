from wasp.placement.transformers_utils import *
from wasp.placement.constants import *
import numpy as np
from wasp.placement.arguments import parse_args
from wasp.placement.model_utils import parse_model_meta

def solve_Alpa_DAG(
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
    
    config = HFUnifiedConfig(model)
    
    n_layer = config.num_layers
    d_model = config.hidden_size
    d_ffn = config.mlp_size
    hidden_size = config.hidden_size
    
    layer_size = static_layer_memory_usage(config) / MB
    layer_flops = (
        4 * NUM_TOKENS * d_model**2
        + 2 * NUM_TOKENS * d_model * d_ffn
        + 2 * B * d_model * S**2
        + 4 * B * S * d_ffn * hidden_size
    ) * 3 # include backprop
    model_flops = layer_flops * n_layer * 1.5  # include activation recompute
    model_size = layer_size * n_layer
    
    print(f"model: {model_flops / TB} TFLOPS")
    print(f"layer: {layer_flops / TB} TFLOPS")
    
    comp_cost_total = (
        model_flops / TB / num_devices / np.min(DEVICE_FLOPS)
    ) / 1000
    
    min_latency = np.inf
    for num_dp in range(1, num_devices + 1):
        if num_devices % num_dp == 0:
            num_way = num_devices // num_dp
        else:
            continue
        
        if num_way < 16:
            continue
        
        print(f"{num_dp=} {num_way=}")
    
        input_size = NUM_TOKENS * hidden_size * BYTE_SIZE / MB / num_dp
        dl_bw = (DL_BW.reshape((num_dp, num_way)) * 1000).copy()
        ul_bw = (UL_BW.reshape((num_dp, num_way)) * 1000).copy()
        
        
        tp_cost = np.zeros(num_dp)
        for i in range(num_dp):
            tp_cost[i] = np.max(2 * input_size * (num_way - 1) / num_way / ul_bw[i, :])
        
        model_cost = np.max(tp_cost) * n_layer * 4 + comp_cost_total * 3
        if num_dp > 1:
            partition_lat = np.zeros(num_way)
            for i in range(num_way):
                partition_lat[i] = np.max(2 * (model_size / num_way) * (num_dp - 1) / num_dp /  ul_bw[:, i])
            model_cost += np.max(partition_lat)
            
            print("partition_lat cost: " + str(partition_lat))
         
        if min_latency > model_cost:
            min_latency = model_cost
            
            print("computation cost: " + str(comp_cost_total))
            print("tp cost: " + str(tp_cost))
            print("model cost: " + str(model_cost))
    
    return min_latency


args = parse_args()
model_meta = parse_model_meta(args.model)

cost = solve_Alpa_DAG(
    args.model,
    args.num_devices,
    args.batch_size,
    args.seq_length,
    args.dl_bw,
    args.ul_bw,
    args.device_flops,
    args.dl_lat,
    args.ul_lat,
)

print("Objective value: ", cost, "s")