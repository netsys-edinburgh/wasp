
from wasp.placement.transformers_utils import *

MODELS = [
    "facebook/opt-13b",
]

GB = 1024 ** 3

BS = 128
SL = 1024

model_data = []
for model in MODELS:
    config = HFUnifiedConfig(model)
    
    num_layers = config.num_layers
    hidden_size = config.hidden_size
    
    print(f"{num_layers=}, {hidden_size=}")
    
    static_mem = static_layer_memory_usage(config) / GB * num_layers
    dynamic_mem = dynamic_layer_memory_usage(config, BS, SL) / GB * num_layers
    attn_score_mem = attn_score_memory_usage(config, BS, SL) / GB * num_layers
    optimizer_mem = optimizer_layer_memory_usage(config) / GB * num_layers
    
    dynamic_mem = (dynamic_mem - attn_score_mem) + attn_score_mem / 20
    
    inter_layer_mem = BS * hidden_size * SL / GB
    
    print(f"{dynamic_mem/128=}")
    print(f"{(static_mem + inter_layer_mem)/16=}")  