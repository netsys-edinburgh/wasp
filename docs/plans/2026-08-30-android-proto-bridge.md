# Android Proto Bridge to wasp-api-v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `wasp/clients/android/AndroidWASPClient` speak the `wasp-api-v2` protocol (the coordinator's gold standard) so it interoperates with the live Wasp coordinator: adopt the canonical protos, read GEMM matrices from the wire, and add the `message_type` dispatch plus the device-register and probe handlers the coordinator drives.

**Architecture:** Two phases. Phase A swaps the client + mock-server protos for the canonical copies under `wasp/clients/android/protocol/` and moves the compute path from eval-mode disk loads to the on-wire `matrix_a`/`matrix_b` payloads — the compute extension (`compute_gemm_request` #101, fields `row`/`col`/`h_dim`) is already aligned, so this is small. Phase B adds `Head.message_type` routing and the `DEVICE_REGISTER_REQUEST`, `PROBE_LATENCY`, `PROBE_BANDWIDTH`, and `PROBE_FLOPS` handlers that a real worker must answer.

**Tech Stack:** Kotlin, protobuf-java/kotlin (proto2) via the protobuf-gradle plugin, koushikdutta AndroidAsync sockets, LiteRT/TFLite `InterpreterApi`. Build + run in Android Studio on a physical Android device; a Python `mock_server` drives round-trips.

**Cannot be verified in CI/headless:** every build/run step here requires Android Studio and a plugged-in device (per the client README). Treat the `./gradlew` / device steps as the implementer's local verification.

**Canonical protocol (source of truth):** `wasp/clients/android/protocol/{global_api,morphling}.proto` (tag `wasp-api-v2`). Key facts this plan relies on:
- Envelope `morphling.UMessage { required Head head=1; required Body body=2 }`; `Head.message_type` (int32, field 6) routes; `Body` has `extensions 100 to max`.
- `MessageType`: `COMPUTE_GEMM_REQUEST=101`, `DEVICE_REGISTER_REQUEST=105`, `DEVICE_PROFILE_DATA=106`, `PROBE_LATENCY_REQUEST=107`/`RESPONSE=108`, `PROBE_BANDWIDTH_REQUEST=109`/`RESPONSE=110`, `PROBE_FLOPS_REQUEST=111`/`RESPONSE=112`.
- Body extensions: `compute_gemm_request=101`, `compute_gemm_response=102`, `device_resgister_request=105` (upstream spelling), `device_profile_data=106`, `probe_*=107..112`.
- `ComputeGemmRequest`: `row=2`, `col=3`, `pivot=4`, `h_dim=5`, `dev_id=6`, `oid=7`, `timestamp=8`, `matrix_a`(Payload #10), `matrix_b`(Payload #20). `Payload{ int64 offset=1; int64 size=2 }`.
- `DeviceProfileData`: required `uuid=1, flops=2, memory=3, ul_bw=4, dl_bw=5, ul_lat=6, dl_lat=7`; optional `measured_*=8..12`.
- `ProbeFlopsRequest`: `probe_id=1, seed=2, m=3, n=4, k=5, bytes matrix_a=10, bytes matrix_b=11`; `ProbeFlopsResponse`: `probe_id=1, bytes matrix_c=10`.
- Wire framing (from `MainActivity.kt`): each message is `[protoSize:int32 BE][tensorSize:int32 BE]` then `protoSize` bytes of `UMessage` then `tensorSize` bytes of tensor blob; matrices are float32 little-endian.

---

## Phase A — Adopt the gold-standard protos

### Task A1: Replace client + mock-server protos with the canonical copies

**Files:**
- Modify: `wasp/clients/android/AndroidWASPClient/app/src/main/proto/global_api.proto`
- Modify: `wasp/clients/android/AndroidWASPClient/app/src/main/proto/morphling.proto`
- Modify: `wasp/clients/android/mock_server/protos/*.proto` + regenerated `*_pb2.py`

- [ ] **Step 1: Copy the canonical protos over the client + mock-server copies**

```bash
cd wasp/clients/android
cp protocol/global_api.proto  AndroidWASPClient/app/src/main/proto/global_api.proto
cp protocol/morphling.proto   AndroidWASPClient/app/src/main/proto/morphling.proto
cp protocol/global_api.proto protocol/morphling.proto mock_server/protos/
```

- [ ] **Step 2: Confirm the client protos now match the canonical ones**

Run:
```bash
cd wasp/clients/android
diff AndroidWASPClient/app/src/main/proto/global_api.proto protocol/global_api.proto && \
diff AndroidWASPClient/app/src/main/proto/morphling.proto  protocol/morphling.proto && \
echo "protos aligned"
```
Expected: `protos aligned` (no diff output). This drops the client's extra `service MemoryManager` and renumbers the `MessageType` enum / device-register / probe fields to `wasp-api-v2`.

- [ ] **Step 3: Regenerate the mock-server Python stubs**

Run (needs `protoc` + `grpcio-tools` or system protoc):
```bash
cd wasp/clients/android/mock_server
python3 -m grpc_tools.protoc -I protos --python_out=. protos/morphling.proto protos/global_api.proto
python3 -c "import morphling_pb2, global_api_pb2; print('mock_server stubs OK')"
```
Expected: `mock_server stubs OK`. (`morphling_pb2.py` and `global_api_pb2.py` regenerate next to `mock_server.py`.)

- [ ] **Step 4: Commit**

```bash
git add wasp/clients/android/AndroidWASPClient/app/src/main/proto wasp/clients/android/mock_server
git commit -m "feat(android): adopt wasp-api-v2 protos in the client and mock server"
```

---

### Task A2: Verify the compute path builds against the new protos

**Files:**
- Read: `AndroidWASPClient/app/src/main/java/com/trainingontheedge/androidwaspclient/MainActivity.kt`

- [ ] **Step 1: Rebuild — the protobuf-gradle plugin regenerates the Java stubs**

Run (in Android Studio or from `AndroidWASPClient/`):
```bash
./gradlew :app:assembleDebug
```
Expected: BUILD SUCCESSFUL. The compute path is unchanged because `GlobalApi.computeGemmRequest` is still extension #101 and `.row`/`.col`/`.hDim` (from `row`/`col`/`h_dim`) still resolve. If the build fails on `getExtension(GlobalApi.computeGemmRequest)`, the extension registry line in `MainActivity.onCreate` (`registry.add(GlobalApi.computeGemmRequest)`) already matches — re-sync gradle and rebuild.

- [ ] **Step 2: Confirm no reference to removed symbols**

Run:
```bash
grep -rnE "MemoryManager|register_device\b|ScheduleGemmSync|GetModelParam" \
  AndroidWASPClient/app/src/main/java || echo "no stale symbols"
```
Expected: `no stale symbols` (the Kotlin never used them; this guards against drift).

---

### Task A3: Read GEMM matrices from the wire instead of disk (eval → real)

**Files:**
- Modify: `MainActivity.kt` — `onDataAvailable` (lines ~177-206) and `computeMatrix` (lines ~261-293)

The current worker runs an evaluation harness: the `if (false)` branch means it *loads A/B from disk* (`loadTensor`) rather than the wire, and never consumes the `tensorSize` bytes. For real integration, consume the tensor blob and slice it by the request's `matrix_a`/`matrix_b` `Payload{offset,size}`.

- [ ] **Step 1: Consume the tensor blob and slice by payload offsets**

Replace the eval block (the `if (false) { ... } else { ... }` in `onDataAvailable`, ~lines 179-206) with a straight wire read:

```kotlin
// Full frame is available: proto already parsed into uMessage.
val tensorBytes = ByteArray(tensorSize)
incomingBuffer.get(tensorBytes, 0, tensorSize)

val gemmReq = uMessage.body.getExtension(GlobalApi.computeGemmRequest)
val aOff = gemmReq.matrixA.offset.toInt(); val aLen = gemmReq.matrixA.size.toInt()
val bOff = gemmReq.matrixB.offset.toInt(); val bLen = gemmReq.matrixB.size.toInt()

val aBytes = tensorBytes.copyOfRange(aOff, aOff + aLen)
val bBytes = tensorBytes.copyOfRange(bOff, bOff + bLen)
val result: ByteArray = computeMatrix(gemmReq, aBytes, bBytes)
```

- [ ] **Step 2: Point `computeMatrix` at the sliced buffers**

Change `computeMatrix` to take the pre-sliced A/B and the request (drop the disk `loadTensor`/`storeTensor` calls from the hot path):

```kotlin
private fun computeMatrix(gemmReq: GlobalApi.ComputeGemmRequest, aBytes: ByteArray, bBytes: ByteArray): ByteArray {
    val m = gemmReq.row.toInt(); val n = gemmReq.col.toInt(); val k = gemmReq.hDim.toInt()
    val aBuf = ByteBuffer.wrap(aBytes).order(ByteOrder.LITTLE_ENDIAN)
    val bBuf = ByteBuffer.wrap(bBytes).order(ByteOrder.LITTLE_ENDIAN)
    val result = cpuRunner.get(aBuf, bBuf, m, k, n)               // FloatArray, m*n
    val out = ByteBuffer.allocate(result.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (v in result) out.putFloat(v)
    return out.array()
}
```

- [ ] **Step 3: Round-trip against the mock server**

Update `mock_server/mock_server.py` to send a `COMPUTE_GEMM_REQUEST` whose `matrix_a`/`matrix_b` payload offsets index into an appended float32 tensor blob (A then B), framed as `[protoSize][tensorSize] + proto + tensor`. Then, from `AndroidWASPClient/`:
```bash
./gradlew :app:installDebug     # onto a plugged-in device
python3 mock_server/mock_server.py --host 0.0.0.0 --port 9999 --m 64 --k 64 --n 64
```
Point the app at the host IP/9999. Expected: the app logs `Received message with proto size ...`, computes, and the server receives `[resultSize][result]` with `resultSize == m*n*4`; server-side A@B matches the returned C within float tolerance.

- [ ] **Step 4: Commit**

```bash
git add wasp/clients/android/AndroidWASPClient/app/src/main/java wasp/clients/android/mock_server
git commit -m "feat(android): compute GEMM from on-wire matrix_a/matrix_b payloads"
```

---

## Phase B — Full worker handshake (register + probes)

### Task B1: Route incoming messages on `Head.message_type`

**Files:**
- Modify: `MainActivity.kt` — inside `onDataAvailable`, after `UMessage.parseFrom`

- [ ] **Step 1: Register every extension the worker must read**

In `onCreate`, extend the registry (it currently holds only `computeGemmRequest`):

```kotlin
registry.add(GlobalApi.computeGemmRequest)
registry.add(GlobalApi.deviceResgisterRequest)   // upstream spelling of the field
registry.add(GlobalApi.probeLatencyRequest)
registry.add(GlobalApi.probeBandwidthRequest)
registry.add(GlobalApi.probeFlopsRequest)
```

- [ ] **Step 2: Dispatch on the message type**

After parsing `uMessage`, branch on `uMessage.head.messageType` instead of assuming compute:

```kotlin
when (uMessage.head.messageType) {
    GlobalApi.MessageType.COMPUTE_GEMM_REQUEST_VALUE   -> handleCompute(uMessage, tensorBytes, socket)
    GlobalApi.MessageType.DEVICE_REGISTER_REQUEST_VALUE -> handleRegister(uMessage, socket)
    GlobalApi.MessageType.PROBE_LATENCY_REQUEST_VALUE  -> handleProbeLatency(uMessage, socket)
    GlobalApi.MessageType.PROBE_BANDWIDTH_REQUEST_VALUE -> handleProbeBandwidth(uMessage, socket)
    GlobalApi.MessageType.PROBE_FLOPS_REQUEST_VALUE    -> handleProbeFlops(uMessage, socket)
    else -> Log.w("Dispatch", "Unhandled message_type ${uMessage.head.messageType}")
}
```

- [ ] **Step 3: Add a reply framer used by every handler**

```kotlin
private fun sendMessage(socket: AsyncSocket, msg: UMessage, tensor: ByteArray = ByteArray(0)) {
    val proto = msg.toByteArray()
    val header = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN)
    header.putInt(proto.size); header.putInt(tensor.size); header.flip()
    val out = ByteBufferList()
    out.add(header); out.add(ByteBuffer.wrap(proto))
    if (tensor.isNotEmpty()) out.add(ByteBuffer.wrap(tensor))
    socket.write(out)
}

private fun newHead(type: Int, flowNo: Int): morphling.Morphling.Head =
    morphling.Morphling.Head.newBuilder()
        .setVersion(1).setMagicFlag(0x12340987).setRandomNum(kotlin.random.Random.nextInt())
        .setFlowNo(flowNo).setSessionNo(System.nanoTime().toString()).setMessageType(type).build()
```

- [ ] **Step 4: Extract the A3 compute+reply into `handleCompute`**

The dispatch in Step 2 calls `handleCompute`; move the A3 inline compute/reply block into a named handler so every branch is symmetric:

```kotlin
private fun handleCompute(req: UMessage, tensorBytes: ByteArray, socket: AsyncSocket) {
    val gemmReq = req.body.getExtension(GlobalApi.computeGemmRequest)
    val aOff = gemmReq.matrixA.offset.toInt(); val aLen = gemmReq.matrixA.size.toInt()
    val bOff = gemmReq.matrixB.offset.toInt(); val bLen = gemmReq.matrixB.size.toInt()
    val result = computeMatrix(gemmReq,
        tensorBytes.copyOfRange(aOff, aOff + aLen),
        tensorBytes.copyOfRange(bOff, bOff + bLen))
    val header = ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN)
    header.putInt(result.size); header.flip()
    val out = ByteBufferList(); out.add(header); out.add(ByteBuffer.wrap(result))
    socket.write(out)   // raw [resultSize][C] reply, matching the existing mock server
}
```
If B6 shows the live coordinator expects a structured `COMPUTE_GEMM_RESPONSE` `UMessage` instead of the raw `[resultSize][C]` frame, wrap `result` in a `ComputeGemmResponse` (fields `row`/`col`/`h_dim` echoed, `matrix_c` = the bytes) and send it via `sendMessage`, mirroring the probe handlers.

- [ ] **Step 5: Build to confirm the dispatch compiles**

Run: `./gradlew :app:assembleDebug` — Expected: BUILD SUCCESSFUL (all handlers defined across B2-B4). Commit after B4.

---

### Task B2: Answer `DEVICE_REGISTER_REQUEST` with `DeviceProfileData`

**Files:**
- Modify: `MainActivity.kt`

- [ ] **Step 1: Build and send the profile**

The coordinator sends an empty `DEVICE_REGISTER_REQUEST`; the device replies with `DEVICE_PROFILE_DATA` carrying its static characteristics. Use conservative on-device measurements (memory from `ActivityManager`, a fixed FLOPS/bandwidth estimate refined by the probes in B3/B4).

```kotlin
private fun handleRegister(req: UMessage, socket: AsyncSocket) {
    val am = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
    val mem = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    val profile = GlobalApi.DeviceProfileData.newBuilder()
        .setUuid(deviceUuid())            // stable per-install id (Step 2)
        .setFlops(benchmarkFlops())       // one-shot local GEMM estimate (below)
        .setMemory(mem.totalMem)          // proto uint64 <- Kotlin Long
        .setUlBw(0L).setDlBw(0L).setUlLat(0L).setDlLat(0L) // refined by probes B3/B4
        .build()
    val body = morphling.Morphling.Body.newBuilder()
        .setExtension(GlobalApi.deviceProfileData, profile).build()
    sendMessage(socket, UMessage.newBuilder()
        .setHead(newHead(GlobalApi.MessageType.DEVICE_PROFILE_DATA_VALUE, req.head.flowNo))
        .setBody(body).build())
}

/** One-shot on-device FLOPS estimate: time a fixed 256^3 GEMM via the runner. */
private fun benchmarkFlops(): Long {
    val s = 256
    val a = ByteBuffer.allocate(s * s * 4).order(ByteOrder.LITTLE_ENDIAN)
    val b = ByteBuffer.allocate(s * s * 4).order(ByteOrder.LITTLE_ENDIAN)
    val t0 = System.nanoTime()
    cpuRunner.get(a, b, s, s, s)
    val secs = (System.nanoTime() - t0) / 1e9
    return (2.0 * s * s * s / secs).toLong()   // 2*m*n*k FLOPs / seconds
}
```

Note: `DeviceProfileData` numeric fields are `required uint64`; proto `uint64` maps to a Java/Kotlin `Long`, so pass `Long` values (`benchmarkFlops()` returns `Long`; `mem.totalMem` is `Long`). Bandwidth/latency fields start at `0L` and are refined by the probes in B3/B4.

- [ ] **Step 2: Add a stable device UUID helper**

```kotlin
private fun deviceUuid(): Long {
    val prefs = getSharedPreferences("wasp", MODE_PRIVATE)
    val existing = prefs.getLong("uuid", 0L)
    if (existing != 0L) return existing
    val id = java.util.UUID.randomUUID().mostSignificantBits and Long.MAX_VALUE
    prefs.edit().putLong("uuid", id).apply(); return id
}
```

- [ ] **Step 3: Round-trip register against the mock server**

Extend `mock_server.py` to first send `DEVICE_REGISTER_REQUEST` and assert it receives a `DEVICE_PROFILE_DATA` with a non-zero `uuid` and `memory`. Run `./gradlew :app:installDebug`, connect, and confirm the server logs the profile.

---

### Task B3: Answer `PROBE_LATENCY_REQUEST` and `PROBE_BANDWIDTH_REQUEST` (echo)

**Files:**
- Modify: `MainActivity.kt`

- [ ] **Step 1: Echo the payload back with the matching response type**

Both probes echo the request `payload` verbatim (the coordinator times the round-trip / throughput):

```kotlin
private fun handleProbeLatency(req: UMessage, socket: AsyncSocket) {
    val p = req.body.getExtension(GlobalApi.probeLatencyRequest)
    val resp = GlobalApi.ProbeLatencyResponse.newBuilder()
        .setProbeId(p.probeId).setPayload(p.payload).build()
    val body = morphling.Morphling.Body.newBuilder()
        .setExtension(GlobalApi.probeLatencyResponse, resp).build()
    sendMessage(socket, UMessage.newBuilder()
        .setHead(newHead(GlobalApi.MessageType.PROBE_LATENCY_RESPONSE_VALUE, req.head.flowNo))
        .setBody(body).build())
}

private fun handleProbeBandwidth(req: UMessage, socket: AsyncSocket) {
    val p = req.body.getExtension(GlobalApi.probeBandwidthRequest)
    val resp = GlobalApi.ProbeBandwidthResponse.newBuilder()
        .setProbeId(p.probeId).setPayload(p.payload).build()
    val body = morphling.Morphling.Body.newBuilder()
        .setExtension(GlobalApi.probeBandwidthResponse, resp).build()
    sendMessage(socket, UMessage.newBuilder()
        .setHead(newHead(GlobalApi.MessageType.PROBE_BANDWIDTH_RESPONSE_VALUE, req.head.flowNo))
        .setBody(body).build())
}
```

- [ ] **Step 2: Round-trip probes against the mock server**

Extend `mock_server.py` to send a 64-byte latency probe and a 4-MiB bandwidth probe and assert the returned `payload` bytes equal what was sent. `./gradlew :app:installDebug`, connect, verify equality server-side.

---

### Task B4: Answer `PROBE_FLOPS_REQUEST` (seeded GEMM recompute)

**Files:**
- Modify: `MainActivity.kt`

- [ ] **Step 1: Recompute the seeded GEMM and return `matrix_c`**

`ProbeFlopsRequest` carries `m,n,k` and `matrix_a`/`matrix_b` as float32 little-endian bytes; compute `C = A @ B` on-device (reuse `cpuRunner`) and return the bytes so the coordinator can verify FLOPS.

```kotlin
private fun handleProbeFlops(req: UMessage, socket: AsyncSocket) {
    val p = req.body.getExtension(GlobalApi.probeFlopsRequest)
    val m = p.m; val n = p.n; val k = p.k
    val aBuf = ByteBuffer.wrap(p.matrixA.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    val bBuf = ByteBuffer.wrap(p.matrixB.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    val c = cpuRunner.get(aBuf, bBuf, m, k, n)               // FloatArray m*n
    val cBytes = ByteBuffer.allocate(c.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (v in c) cBytes.putFloat(v)
    val resp = GlobalApi.ProbeFlopsResponse.newBuilder()
        .setProbeId(p.probeId)
        .setMatrixC(com.google.protobuf.ByteString.copyFrom(cBytes.array())).build()
    val body = morphling.Morphling.Body.newBuilder()
        .setExtension(GlobalApi.probeFlopsResponse, resp).build()
    sendMessage(socket, UMessage.newBuilder()
        .setHead(newHead(GlobalApi.MessageType.PROBE_FLOPS_RESPONSE_VALUE, req.head.flowNo))
        .setBody(body).build())
}
```

- [ ] **Step 2: Build all handlers and commit**

```bash
cd AndroidWASPClient && ./gradlew :app:assembleDebug   # BUILD SUCCESSFUL
git add ../../.. -- wasp/clients/android
git commit -m "feat(android): message_type dispatch + device-register and probe handlers"
```

---

### Task B5: Full round-trip in the mock server

**Files:**
- Modify: `wasp/clients/android/mock_server/mock_server.py`

- [ ] **Step 1: Script the coordinator handshake**

Extend `mock_server.py` to run the coordinator sequence against a connected device: (1) `DEVICE_REGISTER_REQUEST` → expect `DEVICE_PROFILE_DATA`; (2) latency + bandwidth + flops probes → expect matching echoes / a correct `matrix_c`; (3) one `COMPUTE_GEMM_REQUEST` with matrices → expect the correct `C`. Assert each and print a PASS summary.

- [ ] **Step 2: Run it end to end**

```bash
cd wasp/clients/android/AndroidWASPClient && ./gradlew :app:installDebug
python3 ../mock_server/mock_server.py --host 0.0.0.0 --port 9999 --handshake
```
Point the app at the host. Expected: `HANDSHAKE PASS: register, latency, bandwidth, flops, compute`.

- [ ] **Step 3: Commit**

```bash
git add wasp/clients/android/mock_server
git commit -m "test(android): mock-server drives the full register/probe/compute handshake"
```

---

### Task B6: Interop with the live coordinator

**Files:** none (integration verification).

- [ ] **Step 1: Point the worker at a running Wasp coordinator**

Start a coordinator (Morphling proxy at `wasp-api-v2`, default `0.0.0.0:39000`), install the app, and connect it to the coordinator host/port. Expected: the coordinator's device-registration log shows the phone's `DeviceProfileData`, the probes complete, and the device is admitted into a batch and returns shard outputs.

- [ ] **Step 2: Record the result in the client README**

Add a short "Verified against coordinator `<tag/commit>` on `<device model>`" line to `wasp/clients/android/README.md` and commit.

---

## Execution-order summary

```
Phase A: A1 protos -> A2 build compute -> A3 read matrices from wire
Phase B: B1 dispatch -> B2 register -> B3 latency/bandwidth -> B4 flops -> B5 mock handshake -> B6 live coordinator
```

## Out of scope (deliberately)

- The QNN/LiteRT native GPU path from `device_characteristics/EdgeClientTest` (the worker uses the LiteRT `Runner`; a native GPU runner is a separate optimization).
- Multi-shard scheduling / cache-eviction policy (the coordinator drives placement; the worker executes what it is sent).
- Background-service lifecycle (the app runs foreground per upstream; a foreground service is a later hardening step).

## Notes on the `flops` placeholder in B2

`DeviceProfileData.flops` is `required uint64`; the plan sets it to `0` at register time and relies on `PROBE_FLOPS` to establish measured FLOPS (`measured_flops`, `measured_flops_verified`). If the coordinator requires a non-zero static `flops` before probing, seed it from a one-shot local `cpuRunner` GEMM benchmark in `handleRegister` (compute a fixed `256x256x256` GEMM, time it, set `flops = 2*256^3 / seconds`).
