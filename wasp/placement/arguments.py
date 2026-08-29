from transformers import HfArgumentParser
from dataclasses import dataclass, field
import numpy as np
from .model_utils import parse_model_meta
from transformers import set_seed

@dataclass
class SolverArguments:
    model: str = field(default="facebook/opt-13b", metadata={"help": "Huggingface model name"})
    num_devices: int = field(default=256, metadata={"help": "Number of devices"})
    batch_size: int = field(default=128, metadata={"help": "Batch size"})
    seq_length: int = field(default=1024, metadata={"help": "Sequence length"})
    flops_lb: float = field(default=5, metadata={"help": "Lower bound of device TFLOPS"})
    flops_ub: float = field(default=7, metadata={"help": "Upper bound of device TFLOPS"})
    ul_bw_lb: float = field(default=5, metadata={"help": "Lower bound of device uplink bandwidth (MB/s)"})
    ul_bw_ub: float = field(default=10, metadata={"help": "Upper bound of device uplink bandwidth (MB/s)"})
    dl_bw_lb: float = field(default=10, metadata={"help": "Lower bound of device downlink bandwidth (MB/s)"})
    dl_bw_ub: float = field(default=100, metadata={"help": "Upper bound of device downlink bandwidth (MB/s)"})
    ul_latency_lb: float = field(default=0, metadata={"help": "Lower bound of device uplink latency"})
    ul_latency_ub: float = field(default=0.5, metadata={"help": "Upper bound of device uplink latency"})
    dl_latency_lb: float = field(default=0, metadata={"help": "Lower bound of device downlink latency"})
    dl_latency_ub: float = field(default=0.5, metadata={"help": "Upper bound of device downlink latency"})
    
    straggler_ratio: float = field(default=-1, metadata={"help": "Straggler ratio"})
    straggler_num: int = field(default=-1, metadata={"help": "Number of stragglers"})
    straggler_ul_scale: float = field(default=0.1, metadata={"help": "Straggler uplink scale"})
    straggler_dl_scale: float = field(default=0.1, metadata={"help": "Straggler downlink scale"})
    straggler_flops_scale: float = field(default=0.1, metadata={"help": "Strafficking FLOPS scale"})
    memory_budget_mb: float = field(default=32768, metadata={"help": "Per-device memory budget in MB (default 32GB relaxed, paper: 512MB mobile)"})

    def __post_init__(self):
        # all upper bounds must be greater than lower bounds
        assert self.flops_ub > self.flops_lb, "Upper bound of device FLOPS must be greater than lower bound"
        assert self.ul_bw_ub > self.ul_bw_lb, "Upper bound of device uplink bandwidth must be greater than lower bound"
        assert self.dl_bw_ub > self.dl_bw_lb, "Upper bound of device downlink bandwidth must be greater than lower bound"
        assert self.ul_latency_ub > self.ul_latency_lb, "Upper bound of device uplink latency must be greater than lower bound"
        assert self.dl_latency_ub > self.dl_latency_lb, "Upper bound of device downlink latency must be greater than lower bound"

        set_seed(42)
        self.device_flops = np.random.rand(self.num_devices) * (self.flops_ub - self.flops_lb) + self.flops_lb
        self.ul_bw = np.random.randint(self.ul_bw_lb, self.ul_bw_ub, self.num_devices)
        self.dl_bw = np.random.randint(self.dl_bw_lb, self.dl_bw_ub, self.num_devices)
        self.ul_lat = np.zeros(self.num_devices)
        self.dl_lat = np.zeros(self.num_devices)
        
        self.byte_size = 2
        self.num_tokens = self.batch_size * self.seq_length
        
        self.device_flops = self.device_flops / 1000    # TFLOPS / ms
        self.ul_bw = self.ul_bw / 1000                  # MB / ms
        self.dl_bw = self.dl_bw / 1000                  # MB / ms
        self.ul_lat = self.ul_lat * 1000                # ms
        self.dl_lat = self.dl_lat * 1000                # ms
        
        self.model_meta = parse_model_meta(self.model)
        
        # straggler_ratio and straggler_num can only be one of them positive
        # assert self.straggler_ratio >= 0 or self.straggler_num >= 0, "Only one of straggler_ratio and straggler_num can be positive"
        self.num_stragglers = 0
        if self.straggler_ratio > 0:
            self.num_stragglers = int(np.ceil(self.num_devices * self.straggler_ratio))
        if self.straggler_num > 0:
            self.num_stragglers = self.straggler_num
        print(f"Number of stragglers: {self.num_stragglers}, out of {self.num_devices} devices")
        if self.num_stragglers > 0:
            self.straggler_idx = np.random.choice(self.num_devices, self.num_stragglers, replace=False)
            self.ul_bw[self.straggler_idx] = self.ul_bw[self.straggler_idx] * self.straggler_ul_scale
            self.dl_bw[self.straggler_idx] = self.dl_bw[self.straggler_idx] * self.straggler_dl_scale
            self.device_flops[self.straggler_idx] = self.device_flops[self.straggler_idx] * self.straggler_flops_scale
        
        # np.random.seed(None) # reset random seed for stochasticity in solver

def parse_args():
    parser = HfArgumentParser((SolverArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    return args

def get_batch_gemm_sizes(args: SolverArguments):
    model_meta = args.model_meta

    n_head = model_meta["n_head"]
    model_type = model_meta["model_type"]
    
    multipliers = {
        "mlp-gate": 1,
        "mlp-proj": 1,
        "qkvo": 4 * args.batch_size,
        "attn": n_head * args.batch_size,
        "attn_proj": n_head * args.batch_size,
        "bk_input_mlp-gate": 1,
        "bk_weight_mlp-gate": 1,
        "bk_input_mlp-proj": 1,
        "bk_weight_mlp-proj": 1,
        "bk_input_qkvo": 4 * args.batch_size,
        "bk_weight_qkvo": 4 * args.batch_size,
        "bk_input_attn": n_head * args.batch_size,
        "bk_weight_attn": n_head * args.batch_size,
        "bk_input_attn_proj": n_head * args.batch_size,
        "bk_weight_attn_proj": n_head * args.batch_size,
    }
    
    if model_type == "llama":
        multipliers["mlp-gate"] = args.batch_size
        multipliers["mlp-proj"] = args.batch_size
        multipliers["bk_input_mlp-gate"] = args.batch_size
        multipliers["bk_weight_mlp-gate"] = args.batch_size
        multipliers["bk_input_mlp-proj"] = args.batch_size
        multipliers["bk_weight_mlp-proj"] = args.batch_size
        multipliers["mlp-gate1"] = args.batch_size
        multipliers["bk_input_mlp-gate1"] = args.batch_size
        multipliers["bk_weight_mlp-gate1"] = args.batch_size
    
        
    return multipliers

def get_gemm_shapes(args: SolverArguments):
    model_meta = args.model_meta

    d_model = model_meta["d_model"]
    d_ffn = model_meta["d_ffn"]
    n_head = model_meta["n_head"]
    # n_layer = model_meta["n_layer"]
    model_type = model_meta["model_type"]
    
    gemm_shapes = {
        "mlp-gate": (args.num_tokens, d_model, d_ffn),
        "mlp-proj": (args.num_tokens, d_ffn, d_model),
        # "qkvo": (args.num_tokens, d_model, d_model),
        "qkvo": (args.seq_length, d_model, d_model),
        "attn": (args.seq_length, d_model // n_head, d_model // n_head),
        "attn_proj": (args.seq_length, args.seq_length, d_model // n_head),
        
        # "attn": (args.num_tokens * n_head, d_model // n_head, d_model // n_head),
        # "attn_proj": (args.num_tokens * n_head, args.seq_length, d_model // n_head),
        
        "bk_input_mlp-proj": (args.num_tokens, d_model, d_ffn),
        "bk_weight_mlp-proj": (d_model, args.num_tokens, d_ffn),
        
        "bk_input_mlp-gate": (args.num_tokens, d_ffn, d_model),
        "bk_weight_mlp-gate": (d_ffn, args.num_tokens, d_model),
        
        "bk_input_qkvo":  (args.seq_length, d_model, d_model),
        "bk_weight_qkvo": (d_model, args.seq_length, d_model),
        
        "bk_input_attn": (args.seq_length, args.seq_length, d_model // n_head),
        "bk_weight_attn": (args.seq_length, args.seq_length, d_model // n_head),
        
        "bk_input_attn_proj": (args.seq_length, d_model // n_head, args.seq_length),
        "bk_weight_attn_proj": (d_model // n_head, args.seq_length, args.seq_length),
    }
    
    if model_type == "llama":
        gemm_shapes["mlp-proj"] = (args.seq_length, d_ffn, d_model)
        gemm_shapes["bk_input_mlp-proj"] = (args.seq_length, d_model, d_ffn)
        gemm_shapes["bk_weight_mlp-proj"] = (d_model, args.seq_length, d_ffn)
        
        gemm_shapes["mlp-gate"] = (args.seq_length, d_model, d_ffn)
        gemm_shapes["bk_input_mlp-gate"] = (args.seq_length, d_ffn, d_model)
        gemm_shapes["bk_weight_mlp-gate"] = (d_ffn, args.seq_length, d_model)
        
        gemm_shapes["mlp-gate1"] = (args.seq_length, d_model, d_ffn)
        gemm_shapes["bk_input_mlp-gate1"] = (args.seq_length, d_ffn, d_model)
        gemm_shapes["bk_weight_mlp-gate1"] = (d_ffn, args.seq_length, d_model)
        
    return gemm_shapes