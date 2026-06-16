"""Pollution transport model.

Each infrastructure cell emits pollution. Optional storm/runoff multipliers turn
that fixed emission into a simple event load. Water (and the pollution it
carries) moves downstream along the board's flow field, accumulating from every
source it passes. Green cells along the path retain a fraction of whatever flows
through them. Whatever finally reaches a drain / water body, or runs off the grid
edge, counts as "delivered" pollution, which is the thing we want to minimize.

We trace each source independently and sum contributions at the sinks. Retention
is multiplicative per cell, and an optional effectiveness map lets robustness
experiments reduce a placed GI asset's function without changing the placement
itself. The flow field is acyclic by construction (see grid._flow_toward_drains),
so tracing always terminates.
"""

from __future__ import annotations

import numpy as np

from .grid import Cell


# fraction of throughput a green cell removes
DEFAULT_RETENTION = {
    Cell.RAIN_BARREL: 0.15,
    Cell.RAIN_GARDEN: 0.60,
    Cell.REDEVELOPED: 0.80,
}


def simulate(
    base: np.ndarray,
    placements: np.ndarray,
    emission: np.ndarray,
    flow: np.ndarray,
    retention: dict | None = None,
    edge_runoff_counts: bool = True,
    runoff_coeff: np.ndarray | None = None,
    storm_depth: float = 1.0,
    effectiveness: np.ndarray | None = None,
) -> float:
    """Return total pollution delivered to sinks (drains, water bodies, runoff).

    Args:
        base: (m, n) original cell types.
        placements: (m, n) green type per cell, or -1 where none.
        emission: (m, n) pollution emitted per cell.
        flow: (m, n, 2) downstream direction per cell.
        retention: per-green-type removal fraction. Defaults to DEFAULT_RETENTION.
        edge_runoff_counts: if True, pollution running off the grid edge still
            counts as delivered (it pollutes downstream water). If False, only
            pollution reaching DRAIN cells is counted.
        runoff_coeff: optional (m, n) multiplier for source runoff generation.
        storm_depth: scalar storm multiplier for source load.
        effectiveness: optional (m, n) multiplier for installed GI retention.
    """
    if retention is None:
        retention = DEFAULT_RETENTION
    m, n = base.shape
    if runoff_coeff is None:
        runoff_coeff = np.ones((m, n), dtype=float)
    if effectiveness is None:
        effectiveness = np.ones((m, n), dtype=float)
    if runoff_coeff.shape != base.shape:
        raise ValueError("runoff_coeff must match the grid shape")
    if effectiveness.shape != base.shape:
        raise ValueError("effectiveness must match the grid shape")

    # precompute the retention multiplier (fraction that PASSES) for every cell
    keep = np.ones((m, n), dtype=float)
    for r in range(m):
        for c in range(n):
            p = placements[r, c]
            if p >= 0 and Cell(p) in retention:
                keep[r, c] = 1.0 - retention[Cell(p)] * float(effectiveness[r, c])

    delivered = 0.0
    max_steps = m * n + 1  # acyclic flow guarantees we never exceed this

    sources = np.argwhere(emission > 0)
    for sr, sc in sources:
        load = (
            float(emission[sr, sc])
            * float(runoff_coeff[sr, sc])
            * float(storm_depth)
        )
        r, c = int(sr), int(sc)
        steps = 0
        while steps < max_steps:
            load *= keep[r, c]            # this cell retains some of the load
            if base[r, c] == Cell.DRAIN:  # reached a sink
                delivered += load
                break
            dr, dc = flow[r, c]
            if dr == 0 and dc == 0:       # stagnant cell, treat as local sink
                delivered += load
                break
            nr, nc = r + dr, c + dc
            if not (0 <= nr < m and 0 <= nc < n):  # ran off the grid edge
                if edge_runoff_counts:
                    delivered += load
                break
            r, c = nr, nc
            steps += 1
        else:
            # safety net; should be unreachable with an acyclic flow field
            delivered += load
    return delivered
