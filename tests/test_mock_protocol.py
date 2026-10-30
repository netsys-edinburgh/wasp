"""Check the vendored protocol without starting a server or device."""

from pathlib import Path
import subprocess
import sys


def test_global_protocol_registers_dependency_in_fresh_process():
    # Import global_api first so the test cannot inherit a registered
    # morphling.proto descriptor from another test or module.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import global_api_pb2 as global_api
import morphling_pb2 as morphling

request = global_api.ComputeGemmRequest(
    version=1, row=2, col=3, pivot=0, h_dim=4,
    dev_id=5, oid=6, timestamp=7,
)
request.matrix_a.CopyFrom(morphling.Payload(offset=0, size=32))
body = morphling.Body()
body.Extensions[global_api.compute_gemm_request].CopyFrom(request)
decoded = morphling.Body.FromString(body.SerializeToString())
assert decoded.Extensions[global_api.compute_gemm_request] == request
""",
        ],
        cwd=Path(__file__).resolve().parents[1]
        / "wasp/clients/android/mock_server",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
