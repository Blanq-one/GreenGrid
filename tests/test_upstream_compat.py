import numpy as np

from examples.run_adversary_study import build_board, output_paths
from greenspace import Cell


def test_build_board_uses_supplied_layout_dimensions():
    layout = np.full((24, 24), Cell.INFRA, dtype=int)
    layout[0, 0] = Cell.DRAIN

    board = build_board(seed=1, rows=12, cols=12, layout=layout)

    assert board.base.shape == (24, 24)
    assert board.emission.shape == (24, 24)
    assert board.flow.shape == (24, 24, 2)


def test_build_board_can_attach_uniform_runoff_context():
    board = build_board(seed=1, rows=4, cols=4, runoff_coeff=0.5, storm_depth=3.0)

    assert board.extra["storm_depth"] == 3.0
    assert board.extra["runoff_coeff"].shape == (4, 4)
    assert np.all(board.extra["runoff_coeff"] == 0.5)


def test_output_paths_are_absent_when_output_is_omitted():
    assert output_paths(None) == (None, None)


def test_output_paths_create_csv_and_json_names_for_directory(tmp_path):
    csv_path, json_path = output_paths(str(tmp_path))

    assert csv_path == tmp_path / "adversary_rows.csv"
    assert json_path == tmp_path / "adversary_summary.json"
