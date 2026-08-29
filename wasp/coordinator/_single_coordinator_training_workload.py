"""Training workloads shared by the single-coordinator evaluation launcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn.functional as F

from morphling.api import LinearFunction
from morphling.api import PhaseRecorder, track_phase

_LARGE_DIM = 10000
_orig_linear_forward = torch.nn.Linear.forward


@dataclass(frozen=True, slots=True)
class TokenBatchSpec:
    seed: int
    batch_size: int
    sequence_length: int
    vocab_size: int


def make_token_batch(
    spec: TokenBatchSpec,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    return torch.randint(
        low=0,
        high=spec.vocab_size,
        size=(spec.batch_size, spec.sequence_length),
        dtype=torch.long,
        generator=generator,
    ).to(device)


def extract_loss(outputs: Any, input_ids: torch.Tensor) -> torch.Tensor:
    loss = getattr(outputs, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    return F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))


def restricted_linear_forward(self: torch.nn.Linear, inp: torch.Tensor):
    if self.out_features >= _LARGE_DIM or self.in_features >= _LARGE_DIM:
        return _orig_linear_forward(self, inp)
    return LinearFunction.apply(inp, self.weight.t(), self.bias)


def apply_restricted_hook() -> None:
    """Dispatch transformer Linear GEMMs while keeping wide operations native."""
    torch.nn.Linear.forward = restricted_linear_forward
    print(
        f"Restricted Linear dispatch hook applied (skip dims >= {_LARGE_DIM}); "
        "attention bmm/matmul and lm_head stay native.",
        flush=True,
    )


def remove_restricted_hook() -> None:
    torch.nn.Linear.forward = _orig_linear_forward


def infer_vocab_size(model: torch.nn.Module, tokenizer: Any) -> int:
    vs = getattr(getattr(model, "config", object()), "vocab_size", None)
    if isinstance(vs, int):
        return vs
    tvs = getattr(tokenizer, "vocab_size", None)
    if isinstance(tvs, int):
        return tvs
    raise ValueError("Unable to infer vocab_size")


def train_loop(
    model: torch.nn.Module,
    *,
    steps: int,
    warmup_steps: int = 0,
    seed: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
    vocab_size: int,
    phase_recorder: PhaseRecorder | None = None,
) -> List[float]:
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    losses: List[float] = []
    for step_index in range(warmup_steps + steps):
        is_warmup = step_index < warmup_steps
        measured_recorder = None if is_warmup else phase_recorder
        phase_name = "warmup_iteration" if is_warmup else "iteration_total"
        with track_phase(phase_recorder, phase_name):
            input_ids = make_token_batch(
                TokenBatchSpec(
                    seed=seed + step_index,
                    batch_size=batch_size,
                    sequence_length=seq_length,
                    vocab_size=vocab_size,
                ),
                device=device,
            )
            optimizer.zero_grad()
            with track_phase(
                measured_recorder, "forward_device_dispatch_aggregation"
            ):
                outputs = model(input_ids=input_ids, labels=input_ids)
            loss = extract_loss(outputs, input_ids)
            with track_phase(
                measured_recorder, "backward_device_dispatch_aggregation"
            ):
                loss.backward()
            losses.append(float(loss.detach().item()))
            with track_phase(measured_recorder, "optimizer"):
                optimizer.step()
    return losses


def make_tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 4))


def train_loop_tiny(
    model: torch.nn.Module,
    *,
    steps: int,
    warmup_steps: int = 0,
    seed: int,
    device: torch.device,
    phase_recorder: PhaseRecorder | None = None,
) -> List[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(16, 8, generator=gen).to(device)
    y = torch.randn(16, 4, generator=gen).to(device)
    losses: List[float] = []
    for step_index in range(warmup_steps + steps):
        is_warmup = step_index < warmup_steps
        measured_recorder = None if is_warmup else phase_recorder
        phase_name = "warmup_iteration" if is_warmup else "iteration_total"
        with track_phase(phase_recorder, phase_name):
            optimizer.zero_grad()
            with track_phase(
                measured_recorder, "forward_device_dispatch_aggregation"
            ):
                out = model(x)
            loss = F.mse_loss(out, y)
            with track_phase(
                measured_recorder, "backward_device_dispatch_aggregation"
            ):
                loss.backward()
            losses.append(float(loss.detach().item()))
            with track_phase(measured_recorder, "optimizer"):
                optimizer.step()
    return losses
