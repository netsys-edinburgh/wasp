import subprocess
import sys


def test_cli_help_lists_run_subcommand():
    out = subprocess.run(
        [sys.executable, "-m", "wasp.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "run" in out.stdout


def test_cli_reports_solver_backend():
    out = subprocess.run(
        [sys.executable, "-m", "wasp.cli", "info"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "solver backend:" in out.stdout
