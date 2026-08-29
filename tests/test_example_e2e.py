"""End-to-end smoke: the OPT-125M emulated example runs to completion.

Requires the Morphling runtime, so this runs inside the Wasp Docker image.
"""

import subprocess
import sys


def test_example_runs_to_completion():
    out = subprocess.run(
        [sys.executable, "-m", "wasp.cli", "run", "--tiny", "--steps", "2"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert out.returncode == 0, (
        "wasp run failed:\n"
        + out.stdout[-3000:]
        + "\n--- stderr ---\n"
        + out.stderr[-3000:]
    )
