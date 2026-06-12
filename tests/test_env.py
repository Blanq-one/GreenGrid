"""Lightweight tests. Run with: python -m pytest tests/  (or just python tests/test_env.py)

No external test framework required to run the smoke checks below.
"""

from __future__ import annotations

from greenspace import GreenSpaceEnv, CostConfig, generate_board, Cell
from greenspace.agents import (
    HillClimbingAgent,
    ConstructiveHillClimbingAgent,
    SimulatedAnnealingAgent,
    RandomAgent,
    NeuralPlacementAgent,
)
import numpy as np


def test_budget_respected():
    env = GreenSpaceEnv(budget=10.0, seed=1)
    env.reset(seed=1)
    res = RandomAgent(seed=1).solve(env)
    assert res.spent <= env.budget + 1e-6


def test_placements_legal():
    env = GreenSpaceEnv(budget=15.0, seed=2)
    env.reset(seed=2)
    res = HillClimbingAgent(seed=2).solve(env)
    for r, c, g in res.env.current_placements():
        if g == Cell.RAIN_GARDEN:
            assert res.env.board.base[r, c] == Cell.EMPTY
        else:  # barrel or redeveloped -> infra
            assert res.env.board.base[r, c] == Cell.INFRA


def test_green_helps():
    """A real agent should beat the do-nothing baseline."""
    env = GreenSpaceEnv(budget=20.0, seed=3)
    env.reset(seed=3)
    baseline = env.baseline_pollution()
    res = HillClimbingAgent(seed=3).solve(env)
    assert res.pollution <= baseline + 1e-6


def test_local_search_beats_random():
    env = GreenSpaceEnv(budget=20.0, seed=5)
    env.reset(seed=5)
    rand = RandomAgent(seed=5).solve(env)
    hc = HillClimbingAgent(seed=5).solve(env)
    sa = SimulatedAnnealingAgent(seed=5).solve(env)
    assert min(hc.objective, sa.objective) <= rand.objective + 1e-6


def test_gym_step_loop():
    """The standard gymnasium loop runs without error."""
    env = GreenSpaceEnv(budget=12.0, seed=4)
    obs, info = env.reset(seed=4)
    assert obs.shape == env.observation_space.shape
    steps = 0
    while steps < 50:
        valid = env.valid_placements()
        if not valid:
            break
        r, c, g = valid[0]
        # encode (r,c,g) back into the flat action index
        from greenspace.grid import GREEN_TYPES
        cell = r * env.n + c
        action = cell * len(GREEN_TYPES) + GREEN_TYPES.index(g)
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        if term:
            break
    assert info["spent"] <= env.budget + 1e-6


def test_constructive_respects_spec():
    """Constructive search only adds, and spends the budget under a pollution-only
    objective (the spec's 'end once budget fully utilized')."""
    env = GreenSpaceEnv(budget=15.0, seed=6)
    env.reset(seed=6)
    res = ConstructiveHillClimbingAgent(seed=6).solve(env)
    assert res.spent <= env.budget + 1e-6
    # with cost_weight 0, it should keep placing until little budget remains
    assert res.env.remaining_budget() < min(
        env.cfg.barrel_cost, env.cfg.garden_cost, env.cfg.redevelop_cost
    ) + 1e-6
    assert res.pollution <= env.baseline_pollution() + 1e-6


def test_neural_stub_runs_and_helps():
    env = GreenSpaceEnv(budget=18.0, seed=8)
    env.reset(seed=8)
    res = NeuralPlacementAgent(seed=8).solve(env)
    assert res.spent <= env.budget + 1e-6
    assert res.pollution <= env.baseline_pollution() + 1e-6


def test_extra_feature_adds_channel():
    """A feature map on board.extra shows up as an observation channel with no
    other code change."""
    board = generate_board(m=8, n=8, seed=9)
    board.extra["slope"] = np.ones(board.shape, dtype=float)  # a made-up feature
    env = GreenSpaceEnv(board=board, budget=10.0, seed=9)
    obs, _ = env.reset(seed=9)
    assert obs.shape[0] == 7  # 6 base channels + 1 extra
    assert obs.shape == env.observation_space.shape


def test_extra_cost_term_applied():
    """A callable in cfg.extra_terms changes the objective without editing core."""
    base_cfg = CostConfig()
    penalized = CostConfig(extra_terms=[lambda ctx: 100.0 * (ctx["placements"] >= 0).sum()])
    env_a = GreenSpaceEnv(budget=10.0, cost_config=base_cfg, seed=10); env_a.reset(seed=10)
    env_b = GreenSpaceEnv(budget=10.0, cost_config=penalized, seed=10); env_b.reset(seed=10)
    # place one identical green space in each
    r, c, g = env_a.valid_placements()[0]
    env_a.place(r, c, g); env_b.place(r, c, g)
    assert env_b.current_objective() > env_a.current_objective() + 50.0


if __name__ == "__main__":
    for fn in [
        test_budget_respected,
        test_placements_legal,
        test_green_helps,
        test_local_search_beats_random,
        test_gym_step_loop,
        test_constructive_respects_spec,
        test_neural_stub_runs_and_helps,
        test_extra_feature_adds_channel,
        test_extra_cost_term_applied,
    ]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall smoke tests passed")
