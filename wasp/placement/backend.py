"""Placement backend selection.

The exact two-stage SHTP MIP uses Gurobi (commercial). When Gurobi is absent,
Wasp falls back to the bundled heuristic evolutionary solver so every example
runs without a license.
"""

from __future__ import annotations

import os


def gurobi_available() -> bool:
    """True if gurobipy imports (a Gurobi license may still be required)."""
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    return True


def select_backend(prefer: str | None = None) -> str:
    """Return ``'gurobi'`` or ``'heuristic'``.

    ``prefer`` overrides autodetection; the ``WASP_SOLVER`` environment variable
    is the next fallback; otherwise Gurobi is used when importable, else the
    license-free heuristic solver.
    """
    choice = prefer or os.environ.get("WASP_SOLVER")
    if choice in {"gurobi", "heuristic"}:
        if choice == "gurobi" and not gurobi_available():
            raise RuntimeError("WASP_SOLVER=gurobi but gurobipy is unavailable")
        return choice
    return "gurobi" if gurobi_available() else "heuristic"
