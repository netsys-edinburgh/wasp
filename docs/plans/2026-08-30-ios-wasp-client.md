# iOS Wasp Client (iPhone on-device worker) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `wasp/clients/ios/WaspWorker`, an iPhone on-device worker that executes coordinator-dispatched GEMM shards over TCP using the `wasp-api-v2` protocol — the iOS counterpart to `AndroidWASPClient`, matching the paper's iPhone 15 testbed.

**Architecture:** A SwiftUI app. `Network.framework` (`NWConnection`) provides the TCP link with the same `[protoSize:UInt32 BE][tensorSize:UInt32 BE]` + `UMessage` + tensor-blob framing the Android client uses. `apple/swift-protobuf` decodes the `wasp-api-v2` protos (proto2 extensions). Compute is Apple-native rather than LiteRT: a `Runner` protocol with `CpuRunner` (Accelerate `cblas_sgemm`) and `GpuRunner` (Metal Performance Shaders `MPSMatrixMultiplication`) — a plain float32 matmul, which is exactly what the shard worker needs and what MPS/Accelerate do best on iPhone. A `Dispatcher` routes on `Head.message_type` to compute / device-register / probe handlers mirroring the Android proto-bridge plan.

**Tech Stack:** Swift 5.9+, Xcode 15+, SwiftUI, `Network.framework`, `apple/swift-protobuf` (SwiftPM) + `protoc-gen-swift`, `Accelerate`, `MetalPerformanceShaders`, `XCTest`. Build and run on a **physical iPhone**; the Python `mock_server` under `wasp/clients/android/mock_server/` drives round-trips (protocol is identical).

**Cannot be verified in CI/headless:** every build/run step requires macOS + Xcode and a physical iPhone (Metal is unavailable in the simulator; sustained compute needs a real device). Treat `xcodebuild` / device steps as the implementer's local verification. The `XCTest` unit tests (Tasks 3-4) run on a Mac.

**Canonical protocol (source of truth):** `wasp/clients/android/protocol/{global_api,morphling}.proto` (tag `wasp-api-v2`) — the same protos the Android client targets. Relevant shapes:
- `Morphling_UMessage { head: Morphling_Head; body: Morphling_Body }`; `Head.messageType` (Int32) routes; `Body` has proto2 `extensions 100..max`.
- Compute: extension `compute_gemm_request` (#101) → `Morphling_GlobalApi_ComputeGemmRequest { row=2, col=3, pivot=4, h_dim=5, dev_id=6, oid=7, timestamp=8, matrix_a: Payload #10, matrix_b: Payload #20 }`; `Payload { offset, size }`. `row=m`, `col=n`, `h_dim=k`; A is m×k, B is k×n, C is m×n, float32 little-endian, row-major.
- Register: `DEVICE_REGISTER_REQUEST` (105, empty) → reply `DEVICE_PROFILE_DATA` (106) `Morphling_GlobalApi_DeviceProfileData { uuid, flops, memory, ul_bw, dl_bw, ul_lat, dl_lat, measured_*… }`.
- Probes (107-112): latency/bandwidth echo `payload`; `ProbeFlopsRequest { probe_id, seed, m, n, k, matrix_a: bytes, matrix_b: bytes }` → `ProbeFlopsResponse { probe_id, matrix_c: bytes }`.

**swift-protobuf naming** (from `protoc-gen-swift`): package `morphling` → prefix `Morphling_`; `morphling.global_api` → `Morphling_GlobalApi_`. Extensions are accessed as generated properties on the extended message (e.g. `body.Morphling_GlobalApi_computeGemmRequest`) and require passing the generated `Morphling_GlobalApi_Extensions` map when decoding. Confirm exact symbol names in the generated `*.pb.swift` (Task 1) and adjust casing if `protoc-gen-swift` differs.

---

## File structure

```
wasp/clients/ios/WaspWorker/
  Package.swift                      # SwiftPM: swift-protobuf dependency
  Sources/WaspWorker/
    Protocol/                        # generated: morphling.pb.swift, global_api.pb.swift
    Transport/Connection.swift       # NWConnection + framed read/write
    Compute/Runner.swift             # Runner protocol + Matrix helpers
    Compute/CpuRunner.swift          # Accelerate cblas_sgemm
    Compute/GpuRunner.swift          # MPSMatrixMultiplication
    Worker/Dispatcher.swift          # message_type routing + reply framing
    Worker/Handlers.swift            # compute / register / probe handlers
    Worker/DeviceProfile.swift       # uuid, memory, flops benchmark
    App/WaspWorkerApp.swift          # SwiftUI @main
    App/ContentView.swift            # IP/port UI
  Tests/WaspWorkerTests/
    GemmTests.swift                  # CPU/GPU correctness
    FramingTests.swift               # header parse/emit
  scripts/gen-proto.sh               # protoc-gen-swift invocation
```

---

### Task 1: Xcode project, swift-protobuf, and generated protos

**Files:**
- Create: `wasp/clients/ios/WaspWorker/Package.swift`, `scripts/gen-proto.sh`
- Create (generated): `Sources/WaspWorker/Protocol/{morphling,global_api}.pb.swift`

- [ ] **Step 1: Create the SwiftPM manifest**

Create `Package.swift`:

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "WaspWorker",
    platforms: [.iOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.28.0"),
    ],
    targets: [
        .target(name: "WaspWorker", dependencies: [
            .product(name: "SwiftProtobuf", package: "swift-protobuf"),
        ]),
        .testTarget(name: "WaspWorkerTests", dependencies: ["WaspWorker"]),
    ]
)
```

- [ ] **Step 2: Generate Swift protos from the canonical wasp-api-v2 protos**

Create `scripts/gen-proto.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Requires: brew install swift-protobuf  (installs protoc-gen-swift + protoc)
PROTO_DIR="../android/protocol"          # wasp/clients/android/protocol
OUT="Sources/WaspWorker/Protocol"
mkdir -p "$OUT"
protoc -I "$PROTO_DIR" --swift_out="$OUT" \
  "$PROTO_DIR/morphling.proto" "$PROTO_DIR/global_api.proto"
echo "generated $(ls "$OUT")"
```

Run:
```bash
cd wasp/clients/ios/WaspWorker && bash scripts/gen-proto.sh
```
Expected: `generated global_api.pb.swift morphling.pb.swift`.

- [ ] **Step 3: Confirm the generated symbols compile**

Run: `cd wasp/clients/ios/WaspWorker && swift build`
Expected: BUILD SUCCEEDED. Record the exact generated names (e.g. `Morphling_UMessage`, `Morphling_GlobalApi_ComputeGemmRequest`, `Morphling_GlobalApi_Extensions`) from `Protocol/*.pb.swift` — later tasks reference them.

- [ ] **Step 4: Commit**

```bash
git add wasp/clients/ios
git commit -m "feat(ios): scaffold WaspWorker SwiftPM package + generated wasp-api-v2 protos"
```

---

### Task 2: Framed TCP transport

**Files:**
- Create: `Sources/WaspWorker/Transport/Connection.swift`
- Test: `Tests/WaspWorkerTests/FramingTests.swift`

- [ ] **Step 1: Write the failing framing test**

Create `FramingTests.swift`:

```swift
import XCTest
@testable import WaspWorker

final class FramingTests: XCTestCase {
    func testHeaderRoundTrip() {
        let h = FrameHeader(protoSize: 42, tensorSize: 4096)
        let bytes = h.encoded()
        XCTAssertEqual(bytes.count, 8)
        let parsed = FrameHeader(bytes)
        XCTAssertEqual(parsed?.protoSize, 42)
        XCTAssertEqual(parsed?.tensorSize, 4096)
    }
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `swift test --filter FramingTests`
Expected: FAIL (`FrameHeader` undefined).

- [ ] **Step 3: Implement the connection + frame header**

Create `Connection.swift`:

```swift
import Foundation
import Network

struct FrameHeader {
    let protoSize: Int
    let tensorSize: Int
    init(protoSize: Int, tensorSize: Int) { self.protoSize = protoSize; self.tensorSize = tensorSize }
    init?(_ d: Data) {
        guard d.count >= 8 else { return nil }
        func be(_ o: Int) -> Int { Int(UInt32(d[o]) << 24 | UInt32(d[o+1]) << 16 | UInt32(d[o+2]) << 8 | UInt32(d[o+3])) }
        self.protoSize = be(0); self.tensorSize = be(4)
    }
    func encoded() -> Data {
        var d = Data(count: 8)
        func put(_ v: Int, _ o: Int) { d[o] = UInt8((v >> 24) & 0xff); d[o+1] = UInt8((v >> 16) & 0xff); d[o+2] = UInt8((v >> 8) & 0xff); d[o+3] = UInt8(v & 0xff) }
        put(protoSize, 0); put(tensorSize, 4); return d
    }
}

final class Connection {
    private let conn: NWConnection
    private var buffer = Data()
    var onMessage: ((_ proto: Data, _ tensor: Data) -> Void)?

    init(host: String, port: UInt16) {
        conn = NWConnection(host: .init(host), port: .init(rawValue: port)!, using: .tcp)
    }
    func start() {
        conn.stateUpdateHandler = { st in if case .ready = st { NSLog("TCP ready") } }
        conn.start(queue: .global(qos: .userInitiated))
        receive()
    }
    func send(proto: Data, tensor: Data = Data()) {
        var out = FrameHeader(protoSize: proto.count, tensorSize: tensor.count).encoded()
        out.append(proto); out.append(tensor)
        conn.send(content: out, completion: .contentProcessed { _ in })
    }
    func sendRaw(_ d: Data) { conn.send(content: d, completion: .contentProcessed { _ in }) }

    private func receive() {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self else { return }
            if let data, !data.isEmpty { self.buffer.append(data); self.drain() }
            if err == nil && !done { self.receive() }
        }
    }
    private func drain() {
        while true {
            guard let h = FrameHeader(buffer) else { return }
            let total = 8 + h.protoSize + h.tensorSize
            guard buffer.count >= total else { return }
            let proto = buffer.subdata(in: 8 ..< 8 + h.protoSize)
            let tensor = buffer.subdata(in: 8 + h.protoSize ..< total)
            buffer.removeSubrange(0 ..< total)
            onMessage?(proto, tensor)
        }
    }
}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `swift test --filter FramingTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add wasp/clients/ios/WaspWorker/Sources/WaspWorker/Transport wasp/clients/ios/WaspWorker/Tests
git commit -m "feat(ios): framed NWConnection transport"
```

---

### Task 3: CPU GEMM runner (Accelerate)

**Files:**
- Create: `Sources/WaspWorker/Compute/Runner.swift`, `Sources/WaspWorker/Compute/CpuRunner.swift`
- Test: `Tests/WaspWorkerTests/GemmTests.swift`

- [ ] **Step 1: Write the failing test (known 2x2 result)**

Create `GemmTests.swift`:

```swift
import XCTest
@testable import WaspWorker

final class GemmTests: XCTestCase {
    // A(2x3) @ B(3x2) row-major; hand-computed C(2x2).
    let a: [Float] = [1,2,3, 4,5,6]
    let b: [Float] = [7,8, 9,10, 11,12]
    let expected: [Float] = [58,64, 139,154]

    func testCpuGemm() {
        let c = CpuRunner().matmul(a, b, m: 2, k: 3, n: 2)
        XCTAssertEqual(c, expected)
    }
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `swift test --filter GemmTests/testCpuGemm`
Expected: FAIL (`CpuRunner` undefined).

- [ ] **Step 3: Implement the Runner protocol + Accelerate CPU runner**

Create `Runner.swift`:

```swift
protocol Runner {
    /// C(m×n) = A(m×k) @ B(k×n), row-major float32.
    func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float]
}
```

Create `CpuRunner.swift`:

```swift
import Accelerate

final class CpuRunner: Runner {
    func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(m), Int32(n), Int32(k),
                    1.0, a, Int32(k), b, Int32(n), 0.0, &c, Int32(n))
        return c
    }
}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `swift test --filter GemmTests/testCpuGemm`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add wasp/clients/ios/WaspWorker/Sources/WaspWorker/Compute wasp/clients/ios/WaspWorker/Tests/WaspWorkerTests/GemmTests.swift
git commit -m "feat(ios): Accelerate CPU GEMM runner"
```

---

### Task 4: GPU GEMM runner (Metal Performance Shaders)

**Files:**
- Create: `Sources/WaspWorker/Compute/GpuRunner.swift`
- Test: extend `Tests/WaspWorkerTests/GemmTests.swift`

- [ ] **Step 1: Write the failing test (GPU matches CPU on a random matmul)**

Append to `GemmTests.swift`:

```swift
    func testGpuMatchesCpu() throws {
        guard let gpu = GpuRunner() else { throw XCTSkip("no Metal device (simulator)") }
        let m = 64, k = 48, n = 32
        let a = (0..<m*k).map { _ in Float.random(in: -1...1) }
        let b = (0..<k*n).map { _ in Float.random(in: -1...1) }
        let cCpu = CpuRunner().matmul(a, b, m: m, k: k, n: n)
        let cGpu = gpu.matmul(a, b, m: m, k: k, n: n)
        for i in 0..<m*n { XCTAssertEqual(cGpu[i], cCpu[i], accuracy: 1e-2) }
    }
```

- [ ] **Step 2: Run it to see it fail**

Run (on a Mac; the guard skips if no Metal device): `swift test --filter GemmTests/testGpuMatchesCpu`
Expected: FAIL (`GpuRunner` undefined).

- [ ] **Step 3: Implement the MPS GPU runner**

Create `GpuRunner.swift`:

```swift
import Metal
import MetalPerformanceShaders

final class GpuRunner: Runner {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        device = d; queue = q
    }
    func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        let aBuf = device.makeBuffer(bytes: a, length: m*k*4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(bytes: b, length: k*n*4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: m*n*4, options: .storageModeShared)!
        let aM = MPSMatrix(buffer: aBuf, descriptor: .init(rows: m, columns: k, rowBytes: k*4, dataType: .float32))
        let bM = MPSMatrix(buffer: bBuf, descriptor: .init(rows: k, columns: n, rowBytes: n*4, dataType: .float32))
        let cM = MPSMatrix(buffer: cBuf, descriptor: .init(rows: m, columns: n, rowBytes: n*4, dataType: .float32))
        let mul = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                          resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
        let cmd = queue.makeCommandBuffer()!
        mul.encode(commandBuffer: cmd, leftMatrix: aM, rightMatrix: bM, resultMatrix: cM)
        cmd.commit(); cmd.waitUntilCompleted()
        let p = cBuf.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: m*n))
    }
}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `swift test --filter GemmTests/testGpuMatchesCpu`
Expected: PASS on a Metal-capable Mac (SKIP in the simulator).

- [ ] **Step 5: Commit**

```bash
git add wasp/clients/ios/WaspWorker/Sources/WaspWorker/Compute/GpuRunner.swift wasp/clients/ios/WaspWorker/Tests
git commit -m "feat(ios): Metal Performance Shaders GPU GEMM runner"
```

---

### Task 5: Message dispatch + reply helpers

**Files:**
- Create: `Sources/WaspWorker/Worker/Dispatcher.swift`

- [ ] **Step 1: Implement the dispatcher and reply framer**

Create `Dispatcher.swift`:

```swift
import Foundation
import SwiftProtobuf

final class Dispatcher {
    private let conn: Connection
    private let cpu = CpuRunner()
    private let gpu = GpuRunner()
    private let profile = DeviceProfile()

    init(conn: Connection) {
        self.conn = conn
        conn.onMessage = { [weak self] proto, tensor in self?.handle(proto, tensor) }
    }

    private func handle(_ proto: Data, _ tensor: Data) {
        guard let msg = try? Morphling_UMessage(serializedBytes: proto, extensions: Morphling_GlobalApi_Extensions) else {
            NSLog("proto parse failed"); return
        }
        switch msg.head.messageType {
        case 101: Handlers.compute(msg, tensor: tensor, runner: cpu, conn: conn)
        case 105: Handlers.register(msg, profile: profile, runner: cpu, conn: conn)
        case 107: Handlers.probeLatency(msg, conn: conn)
        case 109: Handlers.probeBandwidth(msg, conn: conn)
        case 111: Handlers.probeFlops(msg, runner: cpu, conn: conn)
        default: NSLog("unhandled message_type \(msg.head.messageType)")
        }
    }

    static func newHead(_ type: Int32, flowNo: UInt32) -> Morphling_Head {
        var h = Morphling_Head()
        h.version = 1; h.magicFlag = 0x12340987; h.randomNum = .random(in: .min ... .max)
        h.flowNo = flowNo; h.sessionNo = String(DispatchTime.now().uptimeNanoseconds); h.messageType = type
        return h
    }
}

extension Array where Element == Float {
    /// Little-endian float32 bytes.
    func leBytes() -> Data { self.withUnsafeBytes { Data($0) } }
}
extension Data {
    func floats() -> [Float] { withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
}
```

- [ ] **Step 2: Build**

Run: `swift build` — Expected: BUILD SUCCEEDED once `Handlers` and `DeviceProfile` exist (Tasks 6-8). Commit after Task 8.

---

### Task 6: Compute handler

**Files:**
- Create: `Sources/WaspWorker/Worker/Handlers.swift`

- [ ] **Step 1: Slice matrices by payload and reply**

Create `Handlers.swift` with the compute handler (raw `[resultSize][C]` reply, matching the mock server):

```swift
import Foundation

enum Handlers {
    static func compute(_ msg: Morphling_UMessage, tensor: Data, runner: Runner, conn: Connection) {
        let req = msg.body.Morphling_GlobalApi_computeGemmRequest
        let m = Int(req.row), n = Int(req.col), k = Int(req.hDim)
        let aOff = Int(req.matrixA.offset), aLen = Int(req.matrixA.size)
        let bOff = Int(req.matrixB.offset), bLen = Int(req.matrixB.size)
        let a = tensor.subdata(in: aOff ..< aOff + aLen).floats()
        let b = tensor.subdata(in: bOff ..< bOff + bLen).floats()
        let c = runner.matmul(a, b, m: m, k: k, n: n)
        let cBytes = c.leBytes()
        var out = Data(count: 4)                       // [resultSize:int32 BE]
        out[0] = UInt8((cBytes.count >> 24) & 0xff); out[1] = UInt8((cBytes.count >> 16) & 0xff)
        out[2] = UInt8((cBytes.count >> 8) & 0xff);  out[3] = UInt8(cBytes.count & 0xff)
        out.append(cBytes)
        conn.sendRaw(out)
    }
}
```

Note: if Task 10 shows the live coordinator expects a structured `COMPUTE_GEMM_RESPONSE` `UMessage` (fields `row`/`col`/`h_dim` echoed, `matrix_c` = `cBytes`), send that via `conn.send(proto:)` instead, mirroring the probe handlers below.

- [ ] **Step 2: Build**

Run: `swift build` — Expected: BUILD SUCCEEDED after Task 7-8 add the remaining handlers. Commit after Task 8.

---

### Task 7: Device profile + register handler

**Files:**
- Create: `Sources/WaspWorker/Worker/DeviceProfile.swift`
- Modify: `Sources/WaspWorker/Worker/Handlers.swift`

- [ ] **Step 1: Implement the profile (uuid, memory, benchmarked flops)**

Create `DeviceProfile.swift`:

```swift
import Foundation

final class DeviceProfile {
    let uuid: UInt64
    let memory: UInt64
    init() {
        let d = UserDefaults.standard
        let key = "wasp.uuid"
        let stored = UInt64(bitPattern: Int64(d.integer(forKey: key)))
        if stored != 0 {
            uuid = stored
        } else {
            let fresh = UInt64.random(in: 1 ... .max)
            d.set(Int64(bitPattern: fresh), forKey: key)   // Int is 64-bit on device
            uuid = fresh
        }
        memory = ProcessInfo.processInfo.physicalMemory
    }
    /// One-shot FLOPS estimate: time a fixed 256^3 GEMM.
    func benchmarkFlops(_ runner: Runner) -> UInt64 {
        let s = 256
        let a = [Float](repeating: 1, count: s*s), b = [Float](repeating: 1, count: s*s)
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = runner.matmul(a, b, m: s, k: s, n: s)
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        return UInt64(2.0 * Double(s*s*s) / max(secs, 1e-9))
    }
}
```

- [ ] **Step 2: Add the register handler**

Append to `Handlers.swift`:

```swift
extension Handlers {
    static func register(_ msg: Morphling_UMessage, profile: DeviceProfile, runner: Runner, conn: Connection) {
        var p = Morphling_GlobalApi_DeviceProfileData()
        p.uuid = profile.uuid
        p.flops = profile.benchmarkFlops(runner)
        p.memory = profile.memory
        p.ulBw = 0; p.dlBw = 0; p.ulLat = 0; p.dlLat = 0   // refined by probes
        var body = Morphling_Body()
        body.Morphling_GlobalApi_deviceProfileData = p
        var out = Morphling_UMessage()
        out.head = Dispatcher.newHead(106, flowNo: msg.head.flowNo)
        out.body = body
        if let d = try? out.serializedData() { conn.send(proto: d) }
    }
}
```

- [ ] **Step 3: Build**

Run: `swift build` — Expected: BUILD SUCCEEDED after Task 8. Commit after Task 8.

---

### Task 8: Probe handlers

**Files:**
- Modify: `Sources/WaspWorker/Worker/Handlers.swift`

- [ ] **Step 1: Latency + bandwidth (echo) and flops (recompute)**

Append to `Handlers.swift`:

```swift
extension Handlers {
    static func probeLatency(_ msg: Morphling_UMessage, conn: Connection) {
        let req = msg.body.Morphling_GlobalApi_probeLatencyRequest
        var r = Morphling_GlobalApi_ProbeLatencyResponse(); r.probeID = req.probeID; r.payload = req.payload
        var body = Morphling_Body(); body.Morphling_GlobalApi_probeLatencyResponse = r
        var out = Morphling_UMessage(); out.head = Dispatcher.newHead(108, flowNo: msg.head.flowNo); out.body = body
        if let d = try? out.serializedData() { conn.send(proto: d) }
    }
    static func probeBandwidth(_ msg: Morphling_UMessage, conn: Connection) {
        let req = msg.body.Morphling_GlobalApi_probeBandwidthRequest
        var r = Morphling_GlobalApi_ProbeBandwidthResponse(); r.probeID = req.probeID; r.payload = req.payload
        var body = Morphling_Body(); body.Morphling_GlobalApi_probeBandwidthResponse = r
        var out = Morphling_UMessage(); out.head = Dispatcher.newHead(110, flowNo: msg.head.flowNo); out.body = body
        if let d = try? out.serializedData() { conn.send(proto: d) }
    }
    static func probeFlops(_ msg: Morphling_UMessage, runner: Runner, conn: Connection) {
        let req = msg.body.Morphling_GlobalApi_probeFlopsRequest
        let c = runner.matmul(req.matrixA.floats(), req.matrixB.floats(),
                              m: Int(req.m), k: Int(req.k), n: Int(req.n))
        var r = Morphling_GlobalApi_ProbeFlopsResponse(); r.probeID = req.probeID; r.matrixC = c.leBytes()
        var body = Morphling_Body(); body.Morphling_GlobalApi_probeFlopsResponse = r
        var out = Morphling_UMessage(); out.head = Dispatcher.newHead(112, flowNo: msg.head.flowNo); out.body = body
        if let d = try? out.serializedData() { conn.send(proto: d) }
    }
}
```

- [ ] **Step 2: Build all handlers and commit**

Run: `swift build` — Expected: BUILD SUCCEEDED.
```bash
git add wasp/clients/ios/WaspWorker/Sources/WaspWorker/Worker
git commit -m "feat(ios): dispatcher + compute/register/probe handlers"
```

---

### Task 9: SwiftUI app shell

**Files:**
- Create: `Sources/WaspWorker/App/WaspWorkerApp.swift`, `Sources/WaspWorker/App/ContentView.swift`
- Note: wrap the SwiftPM target in an iOS app target in Xcode (File ▸ New ▸ Project ▸ App, add the local package).

- [ ] **Step 1: IP/port UI that starts the worker**

Create `ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    @State private var ip = "10.0.0.1"
    @State private var port = "9999"
    @State private var connected = false
    @State private var conn: Connection?
    @State private var dispatcher: Dispatcher?

    var body: some View {
        VStack(spacing: 16) {
            if connected {
                Text("Worker running — waiting for shards…")
            } else {
                TextField("Coordinator IP", text: $ip).textFieldStyle(.roundedBorder).autocapitalization(.none)
                TextField("Port", text: $port).textFieldStyle(.roundedBorder).keyboardType(.numberPad)
                Button("Connect") {
                    guard let p = UInt16(port) else { return }
                    let c = Connection(host: ip, port: p)
                    dispatcher = Dispatcher(conn: c)
                    c.start(); conn = c
                    UIApplication.shared.isIdleTimerDisabled = true   // keep computing
                    connected = true
                }.buttonStyle(.borderedProminent)
            }
        }.padding()
    }
}
```

Create `WaspWorkerApp.swift`:

```swift
import SwiftUI

@main
struct WaspWorkerApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
```

- [ ] **Step 2: Build the app for a device**

Run (Xcode or CLI, real device destination):
```bash
xcodebuild -scheme WaspWorker -destination 'generic/platform=iOS' build
```
Expected: BUILD SUCCEEDED (requires a signing team; set it in Xcode ▸ Signing & Capabilities).

- [ ] **Step 3: Commit**

```bash
git add wasp/clients/ios/WaspWorker/Sources/WaspWorker/App
git commit -m "feat(ios): SwiftUI shell to connect and run the worker"
```

---

### Task 10: End-to-end on a physical iPhone

**Files:** none (integration verification).

- [ ] **Step 1: Round-trip against the mock server**

Install on a plugged-in iPhone (Xcode ▸ Run), then drive it with the shared mock server:
```bash
python3 wasp/clients/android/mock_server/mock_server.py --host 0.0.0.0 --port 9999 --handshake
```
Point the app at the Mac's LAN IP / 9999. Expected: `HANDSHAKE PASS: register, latency, bandwidth, flops, compute` — the same script that validates the Android worker (protocol is identical).

- [ ] **Step 2: Interop with the live coordinator**

Start a Wasp coordinator (`wasp-api-v2`), connect the iPhone to it, and confirm registration (its `DeviceProfileData` appears in the coordinator log), the probes complete, and it is admitted into a batch and returns shard outputs. If the coordinator expects a structured `COMPUTE_GEMM_RESPONSE`, apply the note in Task 6.

- [ ] **Step 3: Record the result**

Add a "Verified against coordinator `<tag/commit>` on iPhone `<model, iOS version>`" line to `wasp/clients/ios/README.md` and commit.

---

## Execution-order summary

```
1 scaffold+protos -> 2 transport -> 3 CPU runner -> 4 GPU runner ->
5 dispatcher -> 6 compute -> 7 register -> 8 probes -> 9 UI -> 10 device E2E
```

## iOS-specific notes

- **Foreground compute:** iOS suspends background apps quickly; run the worker in the foreground with `isIdleTimerDisabled = true`. A `BGProcessingTask` can extend windows but is not a substitute for a foreground research run. For sustained fleets, keep the app foregrounded and the device charging.
- **Distribution:** for a research prototype, run via Xcode on a provisioned device or TestFlight; no App Store submission is needed.
- **Thermals:** watch `ProcessInfo.processInfo.thermalState`; under `.serious`/`.critical`, prefer `CpuRunner` (Accelerate) or throttle, since MPS GPU GEMM heats faster.
- **CPU vs GPU default:** the dispatcher uses `CpuRunner` by default (deterministic, matches the register benchmark); switch to `GpuRunner` for large shards where `guard GpuRunner() != nil` and the matmul dominates. A size threshold (e.g. `m*n*k > 2^20` → GPU) mirrors the Android CPU/GPU split.

## Out of scope (deliberately)

- LiteRT-for-iOS (`TensorFlowLiteSwift`): viable if exact parity with the Android LiteRT path is required, but unnecessary for a float32 GEMM worker — Accelerate/MPS are the native, lighter choice. Documented here as the alternative, not the plan.
- Multi-shard scheduling / cache policy (coordinator-driven).
- On-disk shard caching (the Android eval harness's `storeTensor`/`loadTensor`); add later if the coordinator relies on device-side input caching.
