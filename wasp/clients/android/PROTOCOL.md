# Protocol reconciliation: AndroidWASPClient <-> Wasp coordinator

The vendored Android worker and `mock_server/` are a **self-consistent pair**:
both speak the protocol captured in `AndroidWASPClient/app/src/main/proto/`
(`global_api.proto`, `morphling.proto`), so the on-device path can be exercised
end to end against the mock server without the full coordinator.

To drive the worker from the **live Wasp coordinator**, the client protos must be
reconciled with the coordinator's, mirrored here under `protocol/`
(`global_api.proto` + `morphling.proto` at tag `wasp-api-v2`). The canonical
protocol is the coordinator's; the vendored client is behind.

## Delta (client -> wasp-api-v2)

### morphling.proto
- The client declares an extra `service MemoryManager { ScheduleGemmSync,
  GetModelParam }` that the coordinator no longer exposes. Drop it (or leave it
  unused).

### global_api.proto (breaking: enum/field renumbering)
- `MessageType` enum renumbered and extended:
  - `COMPUTE_GEMM_REQUEST` 102 -> **101**, `COMPUTE_GEMM_RESPONSE` 103 -> **102**
  - new `COMPUTE_GEMM_DATA = 103`
  - `DEVICE_REGISTER = 104` -> `DEVICE_REGISTER_REQUEST = 105`
  - new `DEVICE_PROFILE_DATA = 106`, `PROBE_{LATENCY,BANDWIDTH,FLOPS}_{REQUEST,RESPONSE} = 107..112`
- `Body` oneof fields: `register_device = 103` -> `device_resgister_request = 105`
  (note the upstream spelling), plus new `compute_gemm_data = 103`,
  `device_profile_data = 106`, and `probe_* = 107..112`.
- `DeviceProfileData`: `required uint64 backend = 8` -> `optional uint64
  measured_flops = 8`, `measured_flops_verified = 9`, `measured_lat_ns = 10`, ...
  (devices now report measured/verified profiling instead of a static backend id).

## Migration steps (require Android Studio)

1. Replace `AndroidWASPClient/app/src/main/proto/{global_api,morphling}.proto`
   with the versions under `protocol/`.
2. Rebuild — the protobuf-gradle plugin regenerates the Java stubs.
3. Update the Kotlin that consumes the regenerated classes: the `COMPUTE_GEMM_*`
   enum values and message dispatch in `runner/Runner.kt`, and registration /
   profiling in `MainActivity.kt` and `utils/Parser.kt` (the renamed
   `device_resgister_request` and the new profiling/probe fields).
4. If you keep the mock server, regenerate its `*_pb2.py` from the updated protos.

Not attempted here: this environment has no Android SDK and the worker needs a
physical device, so the proto swap, Kotlin fixes, and rebuild are left for Android
Studio. The vendored copy is byte-for-byte the working pair as published.
