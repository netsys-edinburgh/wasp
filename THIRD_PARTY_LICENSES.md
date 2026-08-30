# Third-Party Licenses

Wasp depends on the following components. Their licenses govern their use.

- [Morphling](https://github.com/netsys-edinburgh/morphling) — Apache-2.0.
  Provides the runtime substrate consumed via `morphling.api`.
- [NetworkX](https://networkx.org/) — BSD-3-Clause. DAG construction for the
  placement solver.
- [NumPy](https://numpy.org/) — BSD-3-Clause.
- [Gurobi](https://www.gurobi.com/) (optional) — commercial license required;
  only used when the exact MIP solver backend is selected.
- [Matplotlib](https://matplotlib.org/) (optional) / [pandas](https://pandas.pydata.org/)
  (optional) — result plotting.

## Vendored real-device client (`wasp/clients/android/`)

Vendored from [Training-On-The-Edge](https://github.com/ismaeelbashir03/Training-On-The-Edge),
Copyright (c) Ismaeel Bashir (a Wasp co-author). Upstream declares no license;
confirm with the author before redistribution. Its runtime dependencies:

- [LiteRT / TensorFlow Lite](https://ai.google.dev/edge/litert) (Google AI Edge) — Apache-2.0. On-device GEMM execution (CPU + GPU).
- [Protocol Buffers](https://protobuf.dev/) — BSD-3-Clause. Wire-protocol stubs.
- [AndroidX](https://developer.android.com/jetpack/androidx) / [Material Components](https://github.com/material-components/material-components-android) — Apache-2.0.
- [AndroidAsync](https://github.com/koush/AndroidAsync) — MIT. Socket transport.
- [Jackson](https://github.com/FasterXML/jackson) — Apache-2.0.
