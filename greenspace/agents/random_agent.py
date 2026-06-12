"""Random baseline.

Places green spaces at random until the budget runs out. Not meant to be good,
it's the floor every real agent should beat, and a sanity check that placing
green infrastructure helps at all versus doing nothing.
"""

from __future__ import annotations

import numpy as np

from ..env import GreenSpaceEnv
from .base import Agent, SolveResult


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        cur = env.clone()
        history = [cur.current_objective()]
        while True:
            adds = cur.valid_placements()
            if not adds:
                break
            r, c, g = adds[self.rng.integers(len(adds))]
            cur.place(r, c, g)
            history.append(cur.current_objective())
        return self.result_from(cur, history)
