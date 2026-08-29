from transformers import AutoConfig

def parse_model_meta(model_name) -> dict:
    config = AutoConfig.from_pretrained(model_name)
    
    model_type = config.architectures[0].split("F")[0].lower()
    
    if model_type == "opt":
        d_model = config.hidden_size
        d_ffn = config.ffn_dim
        n_head = config.num_attention_heads
        n_layer = config.num_hidden_layers
        n_kv_head = 0
    
    elif model_type == "llama":
        d_model = config.hidden_size
        d_ffn = config.intermediate_size
        n_head = config.num_attention_heads
        n_layer = config.num_hidden_layers
        n_kv_head = config.num_key_value_heads
        
    else:
        raise ValueError(f"Model type not supported {model_type}")
    
    return {
        "model_type": model_type,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_kv_head": n_kv_head
    }