"""Neural placement agent (stub) + the scorer interface a trained model implements.

This is the seam for the learned model. The agent is a constructive policy: at
each step it asks a `PlacementScorer` to rank the valid placements and commits the
top one, until the budget is spent. The scorer is injected, so:

  - today, the default HeuristicScorer makes this runnable and gives a real
    baseline (it does one-step lookahead), and
  - later, a trained network implements the same `score(env, candidates)` method
    by running a forward pass over `env.observation()`, with no change to the agent.

Any feature maps Shaurya adds to `board.extra` automatically become extra channels
in `env.observation()`, so the model's input grows without touching this file.

A torch sketch of a learned scorer is at the bottom (commented, not imported, so
the package has no torch dependency until you add one).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..env import GreenSpaceEnv
from ..grid import Cell, GREEN_TYPES
from .base import Agent, SolveResult


@runtime_checkable
class PlacementScorer(Protocol):
    """Contract for anything that ranks candidate placements.

    Higher score = more desirable to place next. A learned model satisfies this
    by consuming env.observation() (a (C, m, n) float array) and the candidate
    list, and returning one score per candidate.
    """

    def score(
        self, env: GreenSpaceEnv, candidates: list[tuple[int, int, Cell]]
    ) -> list[float]:
        ...


class HeuristicScorer:
    """Runnable default, no learning. Scores each candidate by how much placing it
    would cut the objective (one-step lookahead). This is the placeholder the
    trained model replaces."""

    def score(self, env, candidates):
        base = env.current_objective()
        scores = []
        for (r, c, g) in candidates:
            env.place(r, c, g)
            scores.append(base - env.current_objective())  # bigger cut = higher
            env.remove(r, c)
        return scores


class NeuralPlacementAgent(Agent):
    """Constructive policy agent driven by a (swappable) PlacementScorer."""

    name = "neural"

    def __init__(
        self,
        scorer: PlacementScorer | None = None,
        stop_when_no_gain: bool = True,
        seed: int | None = None,
    ):
        self.scorer = scorer or HeuristicScorer()
        self.stop_when_no_gain = stop_when_no_gain
        self.rng = np.random.default_rng(seed)

    def solve(self, env: GreenSpaceEnv) -> SolveResult:
        cur = env.clone()
        history = [cur.current_objective()]

        while True:
            candidates = cur.valid_placements()
            if not candidates:
                break
            scores = self.scorer.score(cur, candidates)
            i = int(np.argmax(scores))
            if self.stop_when_no_gain and scores[i] <= 1e-9:
                break
            cur.place(*candidates[i])
            history.append(cur.current_objective())

        return self.result_from(cur, history, scorer=type(self.scorer).__name__)


# ---------------------------------------------------------------------------
# Sketch: how a trained network plugs in. Not imported, so no torch dependency
# until you choose to add one. Implements the same PlacementScorer.score contract.
#
# import torch
# import torch.nn as nn
#
# class CNNPlacementScorer(nn.Module):
#     """Reads the (C, m, n) observation, outputs a desirability map per green type."""
#     def __init__(self, n_channels: int, n_green: int = len(GREEN_TYPES)):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, n_green, 1),          # -> (n_green, m, n) logits
#         )
#
#     def forward(self, obs):                      # obs: (B, C, m, n)
#         return self.net(obs)
#
#     @torch.no_grad()
#     def score(self, env, candidates):
#         obs = torch.tensor(env.observation())[None]      # (1, C, m, n)
#         logits = self(obs)[0].cpu().numpy()              # (n_green, m, n)
#         return [float(logits[GREEN_TYPES.index(g), r, c]) for (r, c, g) in candidates]
#
# Training options:
#   - behavior cloning: imitate the placements the local search agents find
#   - REINFORCE / policy gradient: reward = objective reduction per episode
#   - value net: predict objective-to-go and use it to guide the existing search
# ---------------------------------------------------------------------------
