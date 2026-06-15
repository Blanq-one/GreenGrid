"""Demo: generate a city, run every agent on the same board, compare results.

Run:  python -m examples.run_demo
      python -m examples.run_demo --save board.png    # also writes a picture

This is the "grid game" starting point: it shows the environment working end to
end and that the local search agents actually cut pollution versus doing nothing.
"""

from __future__ import annotations

import argparse
import numpy as np

from greenspace import GreenSpaceEnv, CostConfig, generate_board, render_ascii
from greenspace.agents import (
    RandomAgent,
    HillClimbingAgent,
    ConstructiveHillClimbingAgent,
    SimulatedAnnealingAgent,
    NeuralPlacementAgent,
)

def run(seed: int = 7, save: str | None = None, layout: np.ndarray | None = None):
    board = generate_board(m=12, n=12, n_drains=2, seed=seed, layout=layout)
    cfg = CostConfig(pollution_weight=1.0, cost_weight=0.0)  # budget is the hard cap
    env = GreenSpaceEnv(board=board, budget=180000.0, cost_config=cfg, seed=seed)
    env.reset(seed=seed)

    print("=== starting board ===")
    print(render_ascii(env.board.base, env.placements))
    print(f"\nlegend: . empty   # infra   O drain/water   "
          f"b barrel   g garden   G redeveloped")
    print(f"\nbaseline pollution (no green infrastructure): "
          f"{env.baseline_pollution():.2f}")
    print(f"budget: {env.budget}\n")

    agents = [
        RandomAgent(seed=seed),
        ConstructiveHillClimbingAgent(seed=seed),
        HillClimbingAgent(variant="steepest", seed=seed),
        HillClimbingAgent(variant="first_choice", seed=seed),
        SimulatedAnnealingAgent(seed=seed),
        NeuralPlacementAgent(seed=seed),   # stub: heuristic scorer until a model is trained
    ]

    results = {}
    for agent in agents:
        res = agent.solve(env)
        label = f"{agent.name}/{agent.variant}" if hasattr(agent, "variant") else agent.name
        results[label] = res
        print(f"{label:28s} -> {res.summary()}")

    best_label = min(results, key=lambda k: results[k].objective)
    best = results[best_label]
    print(f"\n=== best: {best_label} ===")
    print(render_ascii(best.env.board.base, best.env.placements))
    cut = env.baseline_pollution() - best.pollution
    pct = 100.0 * cut / max(env.baseline_pollution(), 1e-9)
    print(f"\npollution cut: {cut:.2f}  ({pct:.1f}% reduction)  "
          f"for {best.spent:.1f} of {env.budget} budget")

    if save:
        try:
            from examples.visualize import save_board_png
            save_board_png(best.env, save)
            print(f"\nsaved visualization to {save}")
        except Exception as e:  # matplotlib missing or headless issue
            print(f"\n(could not save png: {e})")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", type=str, default=None, help="path to save a PNG")
    args = ap.parse_args()

    layout = np.array([
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 1, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    ])

    run(seed=args.seed, save=args.save)
    run(seed=args.seed, save='planned_' + args.save, layout=layout)
