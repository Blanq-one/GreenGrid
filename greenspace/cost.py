"""Modular cost / objective definition.

Everything that defines "what is a good solution" lives here so it can be passed
around as a single config object. Shaurya's extra features just become more
fields on CostConfig (and another term in `objective`), without the agents or
the env needing to change.

Convention: LOWER objective is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import Cell


@dataclass
class CostConfig:
    # placement costs (budget units)
    barrel_cost: float = 50.0
    garden_cost: float = 10000.0
    redevelop_cost: float = 50000.0

    # objective weights
    pollution_weight: float = 1.0
    cost_weight: float = 0.0   # 0 -> pollution-only objective with budget as a hard cap
                               # >0 -> trade spending off against pollution directly

    # retention (fraction removed) per green type; None -> pollution.DEFAULT_RETENTION
    retention: dict | None = None

    # does pollution running off the grid edge count against us?
    edge_runoff_counts: bool = True

    # FUTURE FEATURES PLUG IN HERE. Each callable takes a context dict
    # (keys: pollution, spent, placements, board, env) and returns a value added
    # to the objective. Add a term without editing objective() or any agent, e.g.:
    #   cfg.extra_terms.append(lambda ctx: 2.0 * disruption_score(ctx["placements"]))
    extra_terms: list[Callable[[dict], float]] = field(default_factory=list)

    def placement_cost(self, green_type: Cell) -> float:
        return {
            Cell.RAIN_BARREL: self.barrel_cost,
            Cell.RAIN_GARDEN: self.garden_cost,
            Cell.REDEVELOPED: self.redevelop_cost,
        }[green_type]


def total_spent(placements: np.ndarray, cfg: CostConfig) -> float:
    spent = 0.0
    for green_type in (Cell.RAIN_BARREL, Cell.RAIN_GARDEN, Cell.REDEVELOPED):
        count = int((placements == green_type).sum())
        spent += count * cfg.placement_cost(green_type)
    return spent


def objective(pollution: float, spent: float, cfg: CostConfig,
              context: dict | None = None) -> float:
    """The scalar local search minimizes. Lower is better.

    Base objective is pollution + cost. Any callables in cfg.extra_terms are
    summed in as well (when a context dict is supplied), so new cost features
    extend the objective without changing this function or the agents.
    """
    value = cfg.pollution_weight * pollution + cfg.cost_weight * spent
    if cfg.extra_terms and context is not None:
        for term in cfg.extra_terms:
            value += term(context)
    return value
