from .config import HFUnifiedConfig
import torch

dtype_size = {
    'bfloat16': 2,  # 16-bit bfloat16
    'float16': 2,  # 16-bit floating-point
    'float32': 4,  # 32-bit floating-point (default)
    'float64': 8,  # 64-bit floating-point
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8
}

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3
TB = 1024 ** 4

def static_layer_memory_usage(config: HFUnifiedConfig):    
    # Get size of each parameter in bytes
    param_size = dtype_size[config.torch_dtype]
    
    if config.model_type == "llama":
        # Parameters in attention layers
        attn_params = (
            config.hidden_size * config.hidden_size +        # Q matrices
            2 * (config.hidden_size * (config.hidden_size // config.num_attn_heads * config.num_kv_heads)) +  # K, V matrices
            (config.hidden_size * config.hidden_size)       # Output projection matrix
        )
        
        # MLP parameters
        mlp_params = (
            (config.hidden_size * config.mlp_size) * 2 +  # First linear transformation
            (config.mlp_size * config.hidden_size)    # Second linear transformation
        )
    else:
        # Parameters in attention layers
        attn_params = config.hidden_size * config.hidden_size * 4
        
        # MLP parameters
        mlp_params = (
            (config.hidden_size * config.mlp_size) +  # First linear transformation
            (config.mlp_size * config.hidden_size)    # Second linear transformation
        )
    
    # LayerNorm and other parameters (approximation)
    layernorm_params = 2 * config.hidden_size  # Two LayerNorms per layer
    
    total_params = attn_params + mlp_params + layernorm_params
    
    # Convert total number of parameters to bytes
    total_memory_bytes = total_params * param_size
    
    return total_memory_bytes
    
    # # Convert bytes to megabytes
    # total_memory_mb = total_memory_bytes / (1024 ** 2)
    
    # return total_memory_mb
    
def optimizer_layer_memory_usage(config: HFUnifiedConfig):
    # Get size of each parameter in bytes
    param_size = dtype_size[config.torch_dtype]
    
    static_memory = static_layer_memory_usage(config)
    
    # assume Adam optimizer
    adam_memory = static_memory / param_size * 8
    
    return adam_memory    


def attn_score_memory_usage(config: HFUnifiedConfig, batch_size: int, seq_length: int):
    # Get size of each parameter in bytes
    param_size = dtype_size[config.torch_dtype]
    
    # Memory required for Q, K, V matrices per head
    qkv_memory = 3 * (batch_size * seq_length * config.hidden_size  * param_size)
    
    # Attention scores memory: considering each head handles a part of the sequence length if num_kv_heads is used
    attn_scores_memory = batch_size * seq_length * seq_length * config.hidden_size * param_size
    # print(f"attn_scores_memory={attn_scores_memory/GB}GB")
    
    return qkv_memory + attn_scores_memory

def dynamic_layer_memory_usage(config: HFUnifiedConfig, batch_size: int, seq_length: int):
    # Get size of each parameter in bytes
    param_size = dtype_size[config.torch_dtype]
    
    # Memory required for Q, K, V matrices per head
    qkv_memory = 3 * (batch_size * seq_length * config.hidden_size  * param_size)

    # Attention scores memory: considering each head handles a part of the sequence length if num_kv_heads is used
    attn_scores_memory = batch_size * seq_length * seq_length * config.hidden_size * param_size
    # print(f"attn_scores_memory={attn_scores_memory/GB}GB")
    
    # Output memory for each attention head
    attn_output_memory = batch_size * seq_length * config.hidden_size * param_size
    
    # Memory for the output of the MLP (two stages)
    mlp_output_memory = 2 * (batch_size * seq_length * config.hidden_size * param_size) + (batch_size * seq_length * config.mlp_size * param_size)
    
    # Summing memory across all layers
    total_dynamic_memory_bytes = (
        qkv_memory + attn_output_memory +
        attn_scores_memory + mlp_output_memory
    )
    
    return total_dynamic_memory_bytes

def static_optimizer_memory_usage(config: HFUnifiedConfig):
    # Get size of each parameter in bytes
    param_size = dtype_size[config.torch_dtype]
    
    static_memory = static_layer_memory_usage(config)
    
    # assume Adam optimizer
    adam_memory = static_memory / param_size * 8
    
    return adam_memory

# compute peak memory usage for a transformers layer
def layer_memory_usage(config: HFUnifiedConfig, batch_size:int, seq_length: int):
    return static_layer_memory_usage(config) + dynamic_layer_memory_usage(config, batch_size, seq_length)    