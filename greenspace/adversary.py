"""Static robustness checks for a completed GreenGrid portfolio.

This is deliberately not a turn-based adversary. The base environment is a
static placement model, so the adversary evaluates "what if" damage to a solved
portfolio without changing the construction process. Asset attacks reduce GI
effectiveness during evaluation; they do not remove the underlying placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import GreenSpaceEnv
from .metrics import percent_reduction, robustness_ratio
from .pollution import simulate


@dataclass(frozen=True)
class Attack:
    """One static perturbation applied during evaluation only."""

    kind: str
    cell: tuple[int, int]


@dataclass(frozen=True)
class AttackResult:
    attacks: tuple[Attack, ...]
    baseline_pollution: float
    portfolio_pollution: float
    attacked_baseline_pollution: float
    attacked_portfolio_pollution: float
    unopposed_reduction: float
    attacked_reduction: float
    robustness_ratio: float | None


def _pollution(
    env: GreenSpaceEnv,
    placements: np.ndarray,
    emission: np.ndarray,
    effectiveness: np.ndarray | None = None,
) -> float:
    return simulate(
        env.board.base,
        placements,
        emission,
        env.board.flow,
        retention=env.cfg.retention,
        edge_runoff_counts=env.cfg.edge_runoff_counts,
        runoff_coeff=env.board.extra.get("runoff_coeff"),
        storm_depth=float(env.board.extra.get("storm_depth", 1.0)),
        effectiveness=effectiveness,
    )


def _apply_attacks(
    env: GreenSpaceEnv, attacks: tuple[Attack, ...], source_multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    placements = env.placements.copy()
    emission = env.board.emission.copy()
    effectiveness = np.ones(env.placements.shape, dtype=float)
    for attack in attacks:
        r, c = attack.cell
        if attack.kind == "disable_gi":
            effectiveness[r, c] = 0.0
        elif attack.kind == "amplify_source":
            emission[r, c] *= float(source_multiplier)
        else:
            raise ValueError(f"unknown attack kind: {attack.kind}")
    return placements, emission, effectiveness


def evaluate_attacks(
    env: GreenSpaceEnv,
    attacks: list[Attack] | tuple[Attack, ...],
    *,
    source_multiplier: float = 1.5,
) -> AttackResult:
    """Evaluate a fixed attack list without mutating ``env``."""

    attack_tuple = tuple(attacks)
    attacked_placements, attacked_emission, effectiveness = _apply_attacks(
        env, attack_tuple, source_multiplier
    )
    empty = np.full(env.placements.shape, -1, dtype=int)

    baseline = _pollution(env, empty, env.board.emission)
    portfolio = _pollution(env, env.placements, env.board.emission)
    attacked_baseline = _pollution(env, empty, attacked_emission)
    attacked_portfolio = _pollution(
        env, attacked_placements, attacked_emission, effectiveness
    )
    unopposed = percent_reduction(baseline, portfolio)
    attacked = percent_reduction(attacked_baseline, attacked_portfolio)
    return AttackResult(
        attacks=attack_tuple,
        baseline_pollution=baseline,
        portfolio_pollution=portfolio,
        attacked_baseline_pollution=attacked_baseline,
        attacked_portfolio_pollution=attacked_portfolio,
        unopposed_reduction=unopposed,
        attacked_reduction=attacked,
        robustness_ratio=robustness_ratio(unopposed, attacked),
    )


def _candidate_attacks(env: GreenSpaceEnv, used: set[Attack]) -> list[Attack]:
    candidates: list[Attack] = []
    rows, cols = np.where(env.placements >= 0)
    for r, c in zip(rows, cols):
        attack = Attack("disable_gi", (int(r), int(c)))
        if attack not in used:
            candidates.append(attack)

    rows, cols = np.where(env.board.emission > 0.0)
    for r, c in zip(rows, cols):
        attack = Attack("amplify_source", (int(r), int(c)))
        if attack not in used:
            candidates.append(attack)

    return sorted(candidates, key=lambda item: (item.kind, item.cell[0], item.cell[1]))


def evaluate_static_adversary(
    env: GreenSpaceEnv,
    *,
    move_budget: int = 2,
    source_multiplier: float = 1.5,
) -> AttackResult:
    """Greedily choose attacks that lower the portfolio's normalized benefit."""

    if move_budget <= 0:
        return evaluate_attacks(env, (), source_multiplier=source_multiplier)
    attacks: list[Attack] = []
    used: set[Attack] = set()
    current = evaluate_attacks(env, (), source_multiplier=source_multiplier)

    for _ in range(int(move_budget)):
        best_attack: Attack | None = None
        best_result = current
        for attack in _candidate_attacks(env, used):
            trial = evaluate_attacks(
                env, [*attacks, attack], source_multiplier=source_multiplier
            )
            if trial.attacked_reduction < best_result.attacked_reduction - 1e-9:
                best_attack = attack
                best_result = trial
        if best_attack is None:
            break
        attacks.append(best_attack)
        used.add(best_attack)
        current = best_result

    return current
