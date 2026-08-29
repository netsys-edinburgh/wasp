from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass
class HFUnifiedConfig:
    model_name: str
    model_type: str
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    hidden_size: int
    torch_dtype: str
    mlp_size: int
    
    def __init__(self, model_name: str):
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        
        if self.config.model_type == "llama" or self.config.model_type == "gemma2":
            self._parse_llama_config()
            
        if self.config.model_type == "opt":
            self._parse_opt_config()

    
    def _parse_opt_config(self):
        self.model_type = self.config.model_type
        self.num_layers = self.config.num_hidden_layers
        self.num_attn_heads = self.config.num_attention_heads
        self.num_kv_heads = 1
        self.hidden_size = self.config.hidden_size
        self.torch_dtype = self.config.torch_dtype
        self.mlp_size = self.config.ffn_dim
        
    def _parse_llama_config(self):
        self.model_type = self.config.model_type
        self.num_layers = self.config.num_hidden_layers
        self.num_attn_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.torch_dtype = self.config.torch_dtype
        self.mlp_size = self.config.intermediate_size
        
    # def _parse_gemma_config(self):
    #     self.model_type = self.config.model_type
    #     self.num_layers = self.config.num_hidden_layers
    #     self.num_attn_heads = self.config.num_attention_heads
    #     self.num_kv_heads = self.config.num_key_value_heads
    #     self.hidden_size = self.config.hidden_size
    #     self.torch_dtype = self.config.torch_dtype
    #     self.mlp_size = self.config.intermediate_size
    
    