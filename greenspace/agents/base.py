"""Base agent interface and shared utilities.

Agents operate on a GreenSpaceEnv. The contract is just `solve(env) -> SolveResult`.
The env passed in is left untouched; agents work on clones so callers can compare
fairly. This common shape is what lets us drop in a NN-based agent later without
changing the demo or evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env import GreenSpaceEnv


@dataclass
class SolveResult:
    env: GreenSpaceEnv               # the solved env (with placements applied)
    objective: float                 # final objective (lower is better)
    pollution: float
    spent: float
    history: list[float] = field(default_factory=list)  # objective over time
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"obj={self.objective:.2f}  pollution={self.pollution:.2f}  "
                f"spent={self.spent:.1f}  steps={len(self.history)}")


class Agent:
    name = "agent"

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        raise NotImplementedError

    @staticmethod
    def result_from(env: GreenSpaceEnv, history: list[float], **meta) -> SolveResult:
        return SolveResult(
            env=env,
            objective=env.current_objective(),
            pollution=env.pollution(),
            spent=env.spent,
            history=history,
            meta=meta,
        )


# ---- neighbor moves shared by the local search agents ---------------------
# A "neighbor" of the current board is reached by one of:
#   add    -> place a green space on a legal, affordable, empty cell
#   remove -> take an existing placement back off
#   swap   -> remove one placement and add a different one
# We sample moves rather than enumerate everything so it scales to big grids.

def sample_neighbor(env: GreenSpaceEnv, rng: np.random.Generator) -> GreenSpaceEnv | None:
    """Return a clone of env with one random move applied, or None if stuck."""
    current = env.current_placements()
    adds = env.valid_placements()

    moves = []
    if adds:
        moves.append("add")
    if current:
        moves.append("remove")
    if current and adds:
        moves.append("swap")
    if not moves:
        return None

    move = rng.choice(moves)
    nb = env.clone()
    if move == "add":
        r, c, g = adds[rng.integers(len(adds))]
        nb.place(r, c, g)
    elif move == "remove":
        r, c, _ = current[rng.integers(len(current))]
        nb.remove(r, c)
    else:  # swap
        r, c, _ = current[rng.integers(len(current))]
        nb.remove(r, c)
        adds2 = nb.valid_placements()
        if adds2:
            r2, c2, g2 = adds2[rng.integers(len(adds2))]
            nb.place(r2, c2, g2)
    return nb
