from .config import HFUnifiedConfig


def forward_layer_flops(config: HFUnifiedConfig, batch_size: int, seq_length: int) -> float:
    """Approximate per-layer forward FLOPs for transformer training.

    Matches the cost model already used in the solver stack:
      - Attention projections (Q/K/V/O)
      - MLP projections
      - Attention score/value matmuls

    Returns raw FLOP count (not scaled by TB/GB).
    """
    num_tokens = batch_size * seq_length
    d_model = config.hidden_size
    d_ffn = config.mlp_size

    if config.model_type in ("llama", "gemma2"):
        # Llama/Gemma-style attention can use grouped KV heads.
        head_dim = d_model // config.num_attn_heads
        kv_dim = head_dim * config.num_kv_heads
        attn_proj_flops = (
            num_tokens * d_model * d_model
            + 2 * num_tokens * d_model * kv_dim
            + num_tokens * d_model * d_model
        )
        # Gated MLP has three projections: gate/up/down.
        mlp_flops = 3 * num_tokens * d_model * d_ffn
    else:
        # OPT-style attention and 2-layer MLP.
        attn_proj_flops = 4 * num_tokens * d_model * d_model
        mlp_flops = 2 * num_tokens * d_model * d_ffn

    # Attention score + weighted-value GEMMs.
    attn_score_flops = 2 * batch_size * d_model * (seq_length ** 2)

    return float(attn_proj_flops + mlp_flops + attn_score_flops)


def backward_layer_flops(config: HFUnifiedConfig, batch_size: int, seq_length: int) -> float:
    """Approximate backward FLOPs as 2x forward FLOPs."""
    return 2.0 * forward_layer_flops(config, batch_size, seq_length)
