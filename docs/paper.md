# Wasp: Foundation Model Training System for the Edge

> Paper companion. Canonical place for paper metadata, the abstract, and
> figure-reproduction notes. The README links here from its Citation section.

## System

Wasp is a coordinator-driven runtime for exact synchronous foundation-model
training over decentralized edge resources. It decomposes GEMM-dominated
training into row/column sub-GEMM shards and elevates each shard to a
first-class runtime object with explicit placement (selective hybrid tensor
parallelism), backpressure-aware dispatch, shard caching, and shard-granular
recovery, preserving cloud-style synchronous update semantics.

## Authors

- Leyang Xue (The University of Edinburgh)
- Yufeng Xia (The University of Edinburgh)
- Myungjin Lee (Cisco Research)
- Mahesh K. Marina (The University of Edinburgh)

## Built on Morphling

Large-scale results use the measurement-calibrated Morphling emulator
(EdgeSys '26, <https://github.com/netsys-edinburgh/morphling>), consumed through
the public `morphling.api` surface. Please cite Morphling when using the
emulated-scale results.

## Reproduction scope

This repository ships a reference implementation and a runnable OPT-125M
emulated example (`examples/opt125m_emulated/`). Baseline reimplementations and
full paper-figure reproduction are seeded from the original artifact but are not
a gated artifact-evaluation pipeline. The placement solver defaults to the
license-free heuristic backend; the exact Gurobi MIP backend is optional.
