from examples.run_demo import planned_save_path


def test_planned_save_path_adds_default_when_save_is_omitted():
    assert planned_save_path(None) == "planned_board.png"


def test_planned_save_path_uses_user_save_path_when_provided():
    assert planned_save_path("board.png") == "planned_board.png"


def test_planned_save_path_preserves_user_directory():
    assert planned_save_path("outputs/board.png") == "outputs/planned_board.png"
