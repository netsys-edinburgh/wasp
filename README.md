# Wasp

**Coordinator-driven runtime for exact synchronous edge foundation-model
training.**

Wasp decomposes GEMM-dominated training into row/column sub-GEMM shards and
elevates each shard to a first-class runtime object with explicit placement,
transport-commit, cache, and recovery state. It preserves cloud-style
synchronous update semantics while running on heterogeneous, volatile edge
devices.

## Built on Morphling

Wasp consumes the [Morphling](https://github.com/netsys-edinburgh/morphling)
runtime (EdgeSys '26) through its public `morphling.api` surface and runs its
examples on Morphling's measurement-calibrated emulator. The canonical
environment is a Docker image layered on Morphling's image.

## Quick start

See [`examples/opt125m_emulated/`](examples/opt125m_emulated/) for a runnable
OPT-125M end-to-end run on emulated devices. The default placement backend is
the license-free heuristic solver; the exact Gurobi MIP backend is optional
(`pip install "wasp[solver]"`).

## Citation

If you use Wasp, please cite the paper (see [`CITATION.cff`](CITATION.cff) and
[`docs/paper.md`](docs/paper.md)).

## License

Apache-2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
