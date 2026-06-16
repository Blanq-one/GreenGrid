import numpy as np
import pytest

from greenspace import Cell, GreenSpaceEnv, generate_board
from greenspace.pollution import simulate


def one_cell_inputs():
    base = np.array([[Cell.INFRA]], dtype=int)
    placements = np.array([[-1]], dtype=int)
    emission = np.array([[10.0]], dtype=float)
    flow = np.array([[[0, 0]]], dtype=int)
    return base, placements, emission, flow


def test_runoff_scaling_changes_source_load():
    base, placements, emission, flow = one_cell_inputs()

    delivered = simulate(
        base,
        placements,
        emission,
        flow,
        runoff_coeff=np.array([[0.5]], dtype=float),
        storm_depth=3.0,
    )

    assert delivered == pytest.approx(15.0)


def test_effectiveness_map_scales_green_retention():
    base, placements, emission, flow = one_cell_inputs()
    placements[0, 0] = Cell.RAIN_BARREL
    retention = {Cell.RAIN_BARREL: 0.2}

    normal = simulate(base, placements, emission, flow, retention=retention)
    half_effective = simulate(
        base,
        placements,
        emission,
        flow,
        retention=retention,
        effectiveness=np.array([[0.5]], dtype=float),
    )
    blocked = simulate(
        base,
        placements,
        emission,
        flow,
        retention=retention,
        effectiveness=np.array([[0.0]], dtype=float),
    )

    assert normal == pytest.approx(8.0)
    assert half_effective == pytest.approx(9.0)
    assert blocked == pytest.approx(10.0)


def test_env_reads_runoff_and_storm_from_board_extra():
    board = generate_board(m=1, n=1, n_infra=1, n_drains=0, seed=1)
    board.emission[0, 0] = 10.0
    board.flow[0, 0] = (0, 0)
    board.extra["runoff_coeff"] = np.array([[0.5]], dtype=float)
    board.extra["storm_depth"] = 3.0
    env = GreenSpaceEnv(board=board, budget=1000.0)
    env.reset(seed=1)

    assert env.pollution() == pytest.approx(15.0)
