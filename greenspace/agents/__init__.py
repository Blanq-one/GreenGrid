from .base import Agent, SolveResult, sample_neighbor
from .hill_climbing import HillClimbingAgent, ConstructiveHillClimbingAgent
from .simulated_annealing import SimulatedAnnealingAgent
from .random_agent import RandomAgent

# Registry so adding an agent = add one entry here; demos/evals can iterate it.
AGENTS = {
    cls.name: cls
    for cls in (
        RandomAgent,
        HillClimbingAgent,
        ConstructiveHillClimbingAgent,
        SimulatedAnnealingAgent,
    )
}

__all__ = [
    "Agent",
    "SolveResult",
    "sample_neighbor",
    "HillClimbingAgent",
    "ConstructiveHillClimbingAgent",
    "SimulatedAnnealingAgent",
    "RandomAgent",
    "AGENTS",
]
