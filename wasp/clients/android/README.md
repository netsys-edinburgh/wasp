# Wasp real-device client (Android)

The real-device counterpart to Wasp's emulated path: the on-device worker that
executes coordinator-dispatched GEMM shards, matching the paper's Snapdragon
testbed. Vendored from
[Training-On-The-Edge](https://github.com/ismaeelbashir03/Training-On-The-Edge)
by Ismaeel Bashir (a Wasp co-author).

## Layout

- `AndroidWASPClient/` — the on-device worker app
  (`com.trainingontheedge.androidwaspclient`). `runner/CpuRunner.kt` and
  `runner/GpuRunner.kt` execute assigned GEMM shards via LiteRT (CPU and GPU
  paths); `runner/RunnerCache.kt` is the shard cache; `MainActivity.kt` handles
  device registration and the socket link to the coordinator.
- `mock_server/` — a Python server that speaks the paper's wire protocol, so the
  worker can be exercised end to end without the full coordinator.
- `protocol/` — the canonical coordinator protocol at tag `wasp-api-v2`
  (`global_api.proto`, `morphling.proto`). See [`PROTOCOL.md`](PROTOCOL.md) to
  reconcile the client against it.

## Build (Android Studio + a physical device)

Requires Android Studio and a physical Android device (per upstream).

- `compileSdk 35`, `minSdk 24`, `targetSdk 34`.
- On-device ML via **LiteRT** (Google AI Edge, formerly TensorFlow Lite),
  including `litert-gpu` for the GPU path (Adreno/QNN on Snapdragon).
- Protobuf stubs are generated at build time by the protobuf-gradle plugin from
  `AndroidWASPClient/app/src/main/proto/`.

```bash
# open AndroidWASPClient/ in Android Studio, or from its directory:
./gradlew :app:assembleDebug      # build
./gradlew :app:installDebug       # install on a plugged-in device
```

Standalone test: run `mock_server/mock_server.py` on a host the device can reach
and point the app at it.

## Status

Vendored as published and **not built in this environment** (no Android SDK; the
app needs a physical device). The Android build and the `wasp-api-v2` protocol
reconciliation ([`PROTOCOL.md`](PROTOCOL.md)) should be completed in Android
Studio.

## Attribution and license

Copyright (c) Ismaeel Bashir. Vendored from
`ismaeelbashir03/Training-On-The-Edge`. The upstream repository does not currently
declare a license; confirm licensing with the author before redistribution.
Runtime dependencies (LiteRT, protobuf, AndroidX, androidasync, Jackson) are
under their own licenses.
