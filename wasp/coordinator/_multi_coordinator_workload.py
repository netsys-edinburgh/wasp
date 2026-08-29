"""Tiny and OPT rank workloads sharing the single-coordinator model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from morphling.api import PhaseRecorder, track_phase
from wasp.coordinator._runtime_common import load_model_and_tokenizer
from wasp.coordinator._single_coordinator_training_workload import (
    TokenBatchSpec,
    apply_restricted_hook,
    extract_loss,
    infer_vocab_size,
    make_token_batch,
    remove_restricted_hook,
)
from wasp.scaling.multi_coordinator_scaling_config import ScalingConfig

LossFunction = Callable[[torch.nn.Module], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ModelStep:
    model: torch.nn.Module
    loss_function: LossFunction


def _optimizer(
    config: ScalingConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def _native_losses(
    config: ScalingConfig,
    step_function: ModelStep,
) -> tuple[float, ...]:
    optimizer = _optimizer(config, step_function.model)
    losses: list[float] = []
    for step in range(config.warmup_iterations + config.measured_iterations):
        optimizer.zero_grad()
        loss = step_function.loss_function(step_function.model)
        loss.backward()
        if step >= config.warmup_iterations:
            losses.append(float(loss.detach()))
        optimizer.step()
    return tuple(losses)


def _distributed_losses(
    config: ScalingConfig,
    step_function: ModelStep,
    recorder: PhaseRecorder,
) -> tuple[float, ...]:
    optimizer = _optimizer(config, step_function.model)
    losses: list[float] = []
    for step in range(config.warmup_iterations + config.measured_iterations):
        warmup = step < config.warmup_iterations
        measured_recorder = None if warmup else recorder
        phase_name = "warmup_iteration" if warmup else "iteration_total"
        with track_phase(recorder, phase_name):
            optimizer.zero_grad()
            with track_phase(
                measured_recorder, "forward_device_dispatch_aggregation"
            ):
                loss = step_function.loss_function(step_function.model)
            with track_phase(
                measured_recorder, "backward_device_dispatch_aggregation"
            ):
                loss.backward()
            with track_phase(measured_recorder, "gradient_sync"):
                for parameter in step_function.model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad /= config.coordinators
                global_loss = loss.detach().clone()
                dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
                global_loss /= config.coordinators
            if not warmup:
                losses.append(float(global_loss))
            with track_phase(measured_recorder, "optimizer"):
                optimizer.step()
    return tuple(losses)


def _tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def _tiny_inputs(
    seed: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        torch.randn(batch_size, 8, generator=generator),
        torch.randn(batch_size, 4, generator=generator),
    )


def _run_tiny(
    rank: int, config: ScalingConfig, recorder: PhaseRecorder
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    inputs, targets = _tiny_inputs(42, config.global_batch)
    golden = _native_losses(
        config,
        ModelStep(
            _tiny_model(42),
            lambda model: F.mse_loss(model(inputs), targets),
        ),
    )
    lower = rank * config.local_batch
    local_inputs = inputs[lower : lower + config.local_batch]
    local_targets = targets[lower : lower + config.local_batch]
    dispatched = _distributed_losses(
        config,
        ModelStep(
            _tiny_model(42),
            lambda model: F.mse_loss(model(local_inputs), local_targets),
        ),
        recorder,
    )
    return dispatched, golden


def _load_full_model(config: ScalingConfig) -> tuple[torch.nn.Module, int]:
    torch.manual_seed(42)
    model, tokenizer = load_model_and_tokenizer(
        config.model_name, dtype=torch.float32
    )
    return model.to(torch.device("cpu")), infer_vocab_size(model, tokenizer)


def _run_full(
    rank: int, config: ScalingConfig, recorder: PhaseRecorder
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    golden_model, vocab_size = _load_full_model(config)
    tokens = make_token_batch(
        TokenBatchSpec(
            seed=42,
            batch_size=config.global_batch,
            sequence_length=config.sequence_length,
            vocab_size=vocab_size,
        ),
        device=torch.device("cpu"),
    )
    golden = _native_losses(
        config,
        ModelStep(
            golden_model,
            lambda model: extract_loss(
                model(input_ids=tokens, labels=tokens), tokens
            ),
        ),
    )
    del golden_model
    dispatched_model, _ = _load_full_model(config)
    lower = rank * config.local_batch
    local_tokens = tokens[lower : lower + config.local_batch]
    apply_restricted_hook()
    try:
        dispatched = _distributed_losses(
            config,
            ModelStep(
                dispatched_model,
                lambda model: extract_loss(
                    model(input_ids=local_tokens, labels=local_tokens),
                    local_tokens,
                ),
            ),
            recorder,
        )
    finally:
        remove_restricted_hook()
    return dispatched, golden


def run_workload(
    rank: int, config: ScalingConfig, recorder: PhaseRecorder
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if config.tiny:
        return _run_tiny(rank, config, recorder)
    return _run_full(rank, config, recorder)
