"""Grid primitives: cell types, the base board, and board generation.

The board is split into two layers:
  base       -> the original city layout (EMPTY / INFRA / DRAIN), fixed per episode
  placements -> the green modifications we add on top (barrel / garden / redevelop)

Keeping these separate makes it trivial to add, remove, or revert a placement
without losing the original layout, which is exactly what local search needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Cell(IntEnum):
    # base layer
    EMPTY = 0       # undeveloped land, can host a rain garden
    INFRA = 1       # built surface, emits pollution, can host a barrel or be redeveloped
    DRAIN = 2       # stormwater drain / local water body, a pollution sink

    # placement layer (overlaid on top of a base cell)
    RAIN_BARREL = 3   # on INFRA, cheap, small effect
    RAIN_GARDEN = 4   # on EMPTY, pricier, bigger effect
    REDEVELOPED = 5   # INFRA torn up and turned green, most expensive, best effect


GREEN_TYPES = (Cell.RAIN_BARREL, Cell.RAIN_GARDEN, Cell.REDEVELOPED)

# which base cell each green type is allowed to sit on
LEGAL_BASE = {
    Cell.RAIN_BARREL: Cell.INFRA,
    Cell.RAIN_GARDEN: Cell.EMPTY,
    Cell.REDEVELOPED: Cell.INFRA,
}


@dataclass
class Board:
    """A generated city layout plus its pollution emission field.

    Everything Shaurya's richer environment produces (extra feature channels,
    different emission models, etc.) can be attached here without touching the
    agents, because agents only ever talk to the env API, never to Board directly.
    """

    base: np.ndarray            # (m, n) int, values in {EMPTY, INFRA, DRAIN}
    emission: np.ndarray        # (m, n) float, pollution emitted per cell
    flow: np.ndarray            # (m, n, 2) int, downstream (dr, dc) per cell
    extra: dict = field(default_factory=dict)  # open slot for added feature maps

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    def copy(self) -> "Board":
        return Board(
            base=self.base.copy(),
            emission=self.emission.copy(),
            flow=self.flow.copy(),
            extra={k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in self.extra.items()},
        )


def _flow_toward_drains(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build a flow field where each cell points to a downstream neighbor.

    We run a multi-source BFS out from every drain to get each cell's distance
    to the nearest drain, then point each cell at whichever neighbor reduces
    that distance the most. Because flow always strictly decreases distance the
    field is acyclic, so pollution tracing can never loop forever. Cells with no
    path to a drain default to flowing south (off the grid edge).
    """
    m, n = base.shape
    INF = m * n + 1
    dist = np.full((m, n), INF, dtype=int)

    frontier = [(r, c) for r in range(m) for c in range(n) if base[r, c] == Cell.DRAIN]
    for r, c in frontier:
        dist[r, c] = 0

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while frontier:
        nxt = []
        for r, c in frontier:
            for dr, dc in neigh:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    nxt.append((nr, nc))
        frontier = nxt

    flow = np.zeros((m, n, 2), dtype=int)
    for r in range(m):
        for c in range(n):
            if base[r, c] == Cell.DRAIN:
                flow[r, c] = (0, 0)  # sink, water stops here
                continue
            best_dir, best_dist = (1, 0), dist[r, c]  # default: flow south
            options = neigh[:]
            rng.shuffle(options)  # break ties randomly so layouts vary
            for dr, dc in options:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and dist[nr, nc] < best_dist:
                    best_dist = dist[nr, nc]
                    best_dir = (dr, dc)
            flow[r, c] = best_dir
    return flow


def generate_board(
    m: int = 12,
    n: int = 12,
    n_infra: int | None = None,
    n_drains: int = 2,
    emission_low: float = 1.0,
    emission_high: float = 5.0,
    seed: int | None = None,
) -> Board:
    """Generate a randomized city board.

    n_infra defaults to ~45% of cells. Pollution sources (the infra emission
    values) are randomized in [emission_low, emission_high].
    """
    rng = np.random.default_rng(seed)
    total = m * n
    if n_infra is None:
        n_infra = int(0.45 * total)
    n_infra = max(0, min(n_infra, total - n_drains))

    base = np.full((m, n), Cell.EMPTY, dtype=int)
    flat = rng.permutation(total)

    drain_idx = flat[:n_drains]
    infra_idx = flat[n_drains:n_drains + n_infra]

    for idx in drain_idx:
        base[idx // n, idx % n] = Cell.DRAIN
    for idx in infra_idx:
        base[idx // n, idx % n] = Cell.INFRA

    emission = np.zeros((m, n), dtype=float)
    infra_mask = base == Cell.INFRA
    emission[infra_mask] = rng.uniform(emission_low, emission_high, size=infra_mask.sum())

    flow = _flow_toward_drains(base, rng)
    return Board(base=base, emission=emission, flow=flow)


# ascii glyphs for quick terminal rendering
GLYPH = {
    Cell.EMPTY: ".",
    Cell.INFRA: "#",
    Cell.DRAIN: "O",
    Cell.RAIN_BARREL: "b",
    Cell.RAIN_GARDEN: "g",
    Cell.REDEVELOPED: "G",
}


def render_ascii(base: np.ndarray, placements: np.ndarray) -> str:
    """Render the current board. Placements override base where present."""
    m, n = base.shape
    rows = []
    for r in range(m):
        line = []
        for c in range(n):
            p = placements[r, c]
            cell = Cell(p) if p >= 0 else Cell(base[r, c])
            line.append(GLYPH[cell])
        rows.append(" ".join(line))
    return "\n".join(rows)
