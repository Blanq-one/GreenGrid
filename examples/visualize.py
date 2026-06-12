"""Optional matplotlib visualization of a board + placements.

Kept separate so the core package has zero plotting dependency. If you want an
interactive pygame "grid game" later this is the natural place to grow it.
"""

from __future__ import annotations

import numpy as np

from greenspace import Cell
from greenspace.env import GreenSpaceEnv


_COLORS = {
    Cell.EMPTY: "#e8e8e0",
    Cell.INFRA: "#7a7a7a",
    Cell.DRAIN: "#2b6cb0",
    Cell.RAIN_BARREL: "#9ae6b4",
    Cell.RAIN_GARDEN: "#48bb78",
    Cell.REDEVELOPED: "#22543d",
}


def save_board_png(env: GreenSpaceEnv, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    m, n = env.board.shape
    rgb = np.zeros((m, n, 3))
    for r in range(m):
        for c in range(n):
            p = env.placements[r, c]
            cell = Cell(p) if p >= 0 else Cell(env.board.base[r, c])
            hexc = _COLORS[cell].lstrip("#")
            rgb[r, c] = [int(hexc[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"pollution={env.pollution():.1f}  spent={env.spent:.0f}")

    legend = [Patch(facecolor=_COLORS[t], label=t.name.lower()) for t in Cell]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
