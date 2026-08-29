"""Smoke tests for the ported placement package.

The license-free core (constants, cost model, heuristic baselines) imports with
no Gurobi present; the exact two-stage SHTP MIP lives behind ``gurobipy`` and is
selected only when available.
"""

from wasp.placement.backend import select_backend


def test_backend_defaults_to_heuristic_without_gurobi(monkeypatch):
    monkeypatch.delenv("WASP_SOLVER", raising=False)
    monkeypatch.setattr(
        "wasp.placement.backend.gurobi_available", lambda: False
    )
    assert select_backend() == "heuristic"


def test_backend_respects_explicit_choice(monkeypatch):
    monkeypatch.setattr(
        "wasp.placement.backend.gurobi_available", lambda: True
    )
    assert select_backend("heuristic") == "heuristic"
    assert select_backend("gurobi") == "gurobi"


def test_backend_rejects_gurobi_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "wasp.placement.backend.gurobi_available", lambda: False
    )
    try:
        select_backend("gurobi")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when gurobi is unavailable")


def test_placement_core_imports_without_gurobi():
    import wasp.placement.constants  # noqa: F401
    import wasp.placement.cost_model  # noqa: F401
    import wasp.placement.asteroid_solver  # noqa: F401
    import wasp.placement.confident_solver  # noqa: F401
