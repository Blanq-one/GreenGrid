import numpy as np

from greenspace import Cell, CostConfig, GreenSpaceEnv, generate_board
from greenspace.adversary import Attack, evaluate_attacks, evaluate_static_adversary
from greenspace.metrics import percent_reduction


def make_env() -> GreenSpaceEnv:
    layout = np.array(
        [
            [Cell.INFRA, Cell.INFRA, Cell.DRAIN],
            [Cell.EMPTY, Cell.EMPTY, Cell.EMPTY],
            [Cell.EMPTY, Cell.EMPTY, Cell.EMPTY],
        ],
        dtype=int,
    )
    board = generate_board(m=3, n=3, layout=layout, seed=1)
    env = GreenSpaceEnv(board=board, budget=20000.0, cost_config=CostConfig())
    env.reset(seed=1)
    assert env.place(0, 1, Cell.RAIN_BARREL)
    return env


def test_evaluate_attacks_uses_attacked_no_gi_baseline_for_source_changes():
    env = make_env()
    result = evaluate_attacks(
        env, [Attack("amplify_source", (0, 0))], source_multiplier=2.0
    )

    assert result.attacked_baseline_pollution > result.baseline_pollution
    assert result.attacked_reduction == percent_reduction(
        result.attacked_baseline_pollution,
        result.attacked_portfolio_pollution,
    )


def test_evaluate_attacks_uses_board_runoff_context():
    env = make_env()
    env.board.extra["runoff_coeff"] = np.full(env.board.shape, 0.5, dtype=float)
    env.board.extra["storm_depth"] = 3.0

    result = evaluate_attacks(env, [])

    assert result.baseline_pollution == env.baseline_pollution()
    assert result.portfolio_pollution == env.pollution()


def test_static_adversary_does_not_mutate_original_env():
    env = make_env()
    original_placements = env.placements.copy()
    original_emission = env.board.emission.copy()

    result = evaluate_static_adversary(env, move_budget=2, source_multiplier=2.0)

    assert result.attacks
    np.testing.assert_array_equal(env.placements, original_placements)
    np.testing.assert_array_equal(env.board.emission, original_emission)


def test_static_adversary_respects_move_budget():
    env = make_env()

    result = evaluate_static_adversary(env, move_budget=1, source_multiplier=2.0)

    assert len(result.attacks) <= 1


def test_static_adversary_is_deterministic():
    env = make_env()

    first = evaluate_static_adversary(env, move_budget=2, source_multiplier=2.0)
    second = evaluate_static_adversary(env, move_budget=2, source_multiplier=2.0)

    assert first.attacks == second.attacks
    assert first.attacked_reduction == second.attacked_reduction


def test_static_adversary_stops_when_no_attack_reduces_benefit():
    layout = np.full((3, 3), Cell.EMPTY, dtype=int)
    layout[0, 0] = Cell.DRAIN
    board = generate_board(m=3, n=3, layout=layout, seed=2)
    env = GreenSpaceEnv(board=board, budget=20000.0, cost_config=CostConfig())
    env.reset(seed=2)
    assert env.place(1, 1, Cell.RAIN_GARDEN)

    result = evaluate_static_adversary(env, move_budget=2, source_multiplier=2.0)

    assert result.attacks == ()
    assert result.unopposed_reduction == 0.0
