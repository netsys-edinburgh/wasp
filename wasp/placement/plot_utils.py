"""Stub for the plotting helper referenced by the ported baseline solvers.

The original Cleave-AE artifact shipped heatmap plotting as an analysis-only
side effect. The Wasp reference implementation keeps the baseline solver logic
importable without it; swap in real plotting if you need the figures.
"""

from __future__ import annotations


def plot_heatmap(*args, **kwargs):
    """No-op stand-in for the original heatmap plot."""
    return None
