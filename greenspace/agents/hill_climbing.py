"""Hill climbing for green-space placement.

Supports two move-selection styles and random restarts:
  - steepest: scan a sample of neighbors, take the best improving one
  - first_choice: take the first improving neighbor found (cheaper per step)

Restarts re-run the climb from a fresh random start and keep the best result,
which is the standard fix for getting stuck in a poor local optimum.
"""

from __future__ import annotations

import numpy as np

from ..env import GreenSpaceEnv
from .base import Agent, SolveResult, sample_neighbor


class HillClimbingAgent(Agent):
    name = "hill_climbing"

    def __init__(
        self,
        variant: str = "steepest",      # "steepest" | "first_choice"
        neighbors_per_step: int = 40,    # how many neighbors to sample each step
        max_steps: int = 500,
        restarts: int = 4,
        seed: int | None = None,
    ):
        assert variant in ("steepest", "first_choice")
        self.variant = variant
        self.neighbors_per_step = neighbors_per_step
        self.max_steps = max_steps
        self.restarts = restarts
        self.rng = np.random.default_rng(seed)

    def _random_start(self, env: GreenSpaceEnv) -> GreenSpaceEnv:
        s = env.clone()
        # spend a random slice of the budget up front to diversify starts
        for _ in range(self.rng.integers(0, max(1, s.m * s.n // 8))):
            adds = s.valid_placements()
            if not adds:
                break
            r, c, g = adds[self.rng.integers(len(adds))]
            s.place(r, c, g)
        return s

    def _climb(self, start: GreenSpaceEnv) -> SolveResult:
        cur = start
        cur_obj = cur.current_objective()
        history = [cur_obj]

        for _ in range(self.max_steps):
            best, best_obj = None, cur_obj
            improved = False
            for _ in range(self.neighbors_per_step):
                nb = sample_neighbor(cur, self.rng)
                if nb is None:
                    continue
                obj = nb.current_objective()
                if obj < best_obj - 1e-9:
                    best, best_obj = nb, obj
                    if self.variant == "first_choice":
                        improved = True
                        break
            if best is not None and best_obj < cur_obj - 1e-9:
                cur, cur_obj = best, best_obj
                history.append(cur_obj)
                improved = True
            if not improved:
                break  # local optimum

        return self.result_from(cur, history, variant=self.variant)

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        best: SolveResult | None = None
        all_history: list[float] = []
        for _ in range(max(1, self.restarts)):
            res = self._climb(self._random_start(env))
            all_history.extend(res.history)
            if best is None or res.objective < best.objective:
                best = res
        best.history = all_history
        best.meta["restarts"] = self.restarts
        return best


class ConstructiveHillClimbingAgent(Agent):
    """Spec-literal local search: neighbors are reached only by PLACING a green
    space, and the search ends once the budget is fully utilized.

    Each step it scores every affordable, legal placement (add-only) and commits
    the single one that most reduces the objective, then repeats. It stops when no
    affordable placement remains, or when the best available placement would make
    the objective worse (relevant only when cost_weight > 0; with a pure-pollution
    objective it runs until the budget is spent). This is the construction the
    project brief describes; the general HillClimbingAgent above usually does
    better but explores a larger neighborhood than the brief specifies.
    """

    name = "constructive_hill_climbing"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        cur = env.clone()
        history = [cur.current_objective()]

        while True:
            candidates = cur.valid_placements()
            if not candidates:
                break  # budget fully utilized

            base_obj = cur.current_objective()
            best_move, best_obj = None, None
            for (r, c, g) in candidates:
                cur.place(r, c, g)
                obj = cur.current_objective()
                cur.remove(r, c)
                if best_obj is None or obj < best_obj:
                    best_move, best_obj = (r, c, g), obj

            if best_obj > base_obj + 1e-9:
                break  # nothing left that helps

            cur.place(*best_move)
            history.append(cur.current_objective())

        return self.result_from(cur, history, mode="constructive_greedy")
