"""Run a small paired-seed adversarial robustness study.

This script treats the upstream environment as the contract. It passes explicit
budgets and board dimensions instead of relying on defaults that changed with
the current cost scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np

from greenspace import Board, CostConfig, GreenSpaceEnv, generate_board
from greenspace.adversary import AttackResult, evaluate_static_adversary
from greenspace.agents import (
    ConstructiveHillClimbingAgent,
    HillClimbingAgent,
    RandomAgent,
    SimulatedAnnealingAgent,
)


def build_board(
    *,
    seed: int,
    rows: int,
    cols: int,
    layout: np.ndarray | None = None,
    runoff_coeff: float | None = None,
    storm_depth: float = 1.0,
) -> Board:
    """Build a board, deriving dimensions from supplied layouts."""

    if layout is not None:
        rows, cols = layout.shape
    board = generate_board(m=rows, n=cols, n_drains=2, seed=seed, layout=layout)
    if runoff_coeff is not None:
        board.extra["runoff_coeff"] = np.full(board.shape, float(runoff_coeff))
    if storm_depth != 1.0:
        board.extra["storm_depth"] = float(storm_depth)
    return board


def output_paths(output: str | None) -> tuple[Path | None, Path | None]:
    """Return CSV and JSON paths, or no paths when output is omitted."""

    if output is None:
        return None, None
    path = Path(output)
    if path.suffix:
        return path, path.with_suffix(".json")
    return path / "adversary_rows.csv", path / "adversary_summary.json"


def agents(seed: int):
    """The current comparison set."""

    return [
        ("random", RandomAgent(seed=seed)),
        ("constructive", ConstructiveHillClimbingAgent(seed=seed)),
        ("hill_climbing", HillClimbingAgent(seed=seed)),
        ("simulated_annealing", SimulatedAnnealingAgent(seed=seed)),
    ]


def _result_row(
    *,
    seed: int,
    agent_name: str,
    budget: float,
    runtime_seconds: float,
    attack_result: AttackResult,
    spent: float,
    placements: int,
) -> dict:
    return {
        "seed": seed,
        "agent": agent_name,
        "budget": budget,
        "spent": spent,
        "placements": placements,
        "runtime_seconds": runtime_seconds,
        "baseline_pollution": attack_result.baseline_pollution,
        "portfolio_pollution": attack_result.portfolio_pollution,
        "attacked_baseline_pollution": attack_result.attacked_baseline_pollution,
        "attacked_portfolio_pollution": attack_result.attacked_portfolio_pollution,
        "unopposed_reduction": attack_result.unopposed_reduction,
        "attacked_reduction": attack_result.attacked_reduction,
        "robustness_ratio": attack_result.robustness_ratio,
        "attacks": ";".join(
            f"{attack.kind}@{attack.cell[0]},{attack.cell[1]}"
            for attack in attack_result.attacks
        ),
    }


def run_study(
    *,
    seeds: Iterable[int],
    rows: int,
    cols: int,
    budget: float,
    attack_budget: int,
    source_multiplier: float = 1.5,
    runoff_coeff: float | None = None,
    storm_depth: float = 1.0,
) -> list[dict]:
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    rows_out: list[dict] = []
    cfg = CostConfig(pollution_weight=1.0, cost_weight=0.0)
    for seed in seeds:
        board = build_board(
            seed=int(seed),
            rows=rows,
            cols=cols,
            runoff_coeff=runoff_coeff,
            storm_depth=storm_depth,
        )
        for agent_name, agent in agents(int(seed)):
            env = GreenSpaceEnv(board=board, budget=budget, cost_config=cfg, seed=int(seed))
            env.reset(seed=int(seed))
            started = time.perf_counter()
            solved = agent.solve(env)
            runtime = time.perf_counter() - started
            attack_result = evaluate_static_adversary(
                solved.env,
                move_budget=attack_budget,
                source_multiplier=source_multiplier,
            )
            rows_out.append(
                _result_row(
                    seed=int(seed),
                    agent_name=agent_name,
                    budget=budget,
                    runtime_seconds=runtime,
                    attack_result=attack_result,
                    spent=solved.spent,
                    placements=len(solved.env.current_placements()),
                )
            )
    return rows_out


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for agent_name in sorted({row["agent"] for row in rows}):
        group = [row for row in rows if row["agent"] == agent_name]
        unopposed = [row["unopposed_reduction"] for row in group]
        attacked = [row["attacked_reduction"] for row in group]
        summary[agent_name] = {
            "n": len(group),
            "mean_unopposed_reduction": mean(unopposed),
            "mean_attacked_reduction": mean(attacked),
            "stdev_unopposed_reduction": stdev(unopposed) if len(unopposed) > 1 else 0.0,
            "stdev_attacked_reduction": stdev(attacked) if len(attacked) > 1 else 0.0,
        }
    return summary


def write_outputs(rows: list[dict], csv_path: Path | None, json_path: Path | None) -> None:
    if csv_path is None or json_path is None:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump({"summary": summarize(rows), "rows": rows}, f, indent=2)


def print_summary(rows: list[dict]) -> None:
    for agent_name, values in summarize(rows).items():
        print(
            f"{agent_name:20s} "
            f"unopposed={values['mean_unopposed_reduction']:.3f} "
            f"attacked={values['mean_attacked_reduction']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(20)))
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--budget", type=float, default=180000.0)
    parser.add_argument("--attack-budget", type=int, default=2)
    parser.add_argument("--source-multiplier", type=float, default=1.5)
    parser.add_argument("--runoff-coeff", type=float, default=None)
    parser.add_argument("--storm-depth", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = run_study(
        seeds=args.seeds,
        rows=args.rows,
        cols=args.cols,
        budget=args.budget,
        attack_budget=args.attack_budget,
        source_multiplier=args.source_multiplier,
        runoff_coeff=args.runoff_coeff,
        storm_depth=args.storm_depth,
    )
    csv_path, json_path = output_paths(args.output)
    write_outputs(rows, csv_path, json_path)
    print_summary(rows)
    if csv_path is not None and json_path is not None:
        print(f"saved rows to {csv_path}")
        print(f"saved summary to {json_path}")


if __name__ == "__main__":
    main()
