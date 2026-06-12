"""Simulated annealing for green-space placement.

Like hill climbing but accepts a worse neighbor with probability exp(-delta / T),
where T cools geometrically over time. Early on it explores freely; as T drops it
behaves more and more like greedy hill climbing. Good at escaping the local optima
that trap plain hill climbing.
"""

from __future__ import annotations

import math

import numpy as np

from ..env import GreenSpaceEnv
from .base import Agent, SolveResult, sample_neighbor


class SimulatedAnnealingAgent(Agent):
    name = "simulated_annealing"

    def __init__(
        self,
        t_start: float = 5.0,
        t_end: float = 0.01,
        cooling: float = 0.995,    # T <- T * cooling each step
        max_steps: int = 4000,
        seed: int | None = None,
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.cooling = cooling
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        cur = env.clone()
        cur_obj = cur.current_objective()
        best, best_obj = cur, cur_obj
        history = [cur_obj]

        T = self.t_start
        for _ in range(self.max_steps):
            if T <= self.t_end:
                break
            nb = sample_neighbor(cur, self.rng)
            if nb is None:
                break
            obj = nb.current_objective()
            delta = obj - cur_obj
            if delta < 0 or self.rng.random() < math.exp(-delta / T):
                cur, cur_obj = nb, obj
                if cur_obj < best_obj:
                    best, best_obj = cur.clone(), cur_obj
            history.append(cur_obj)
            T *= self.cooling

        return self.result_from(best, history,
                                t_start=self.t_start, cooling=self.cooling)
