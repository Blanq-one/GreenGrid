from greenspace.metrics import percent_reduction, robustness_ratio


def test_percent_reduction_uses_matching_baseline():
    assert percent_reduction(100.0, 60.0) == 0.4
    assert percent_reduction(200.0, 120.0) == 0.4


def test_percent_reduction_handles_zero_baseline():
    assert percent_reduction(0.0, 0.0) == 0.0
    assert percent_reduction(0.0, 5.0) == -1.0


def test_robustness_ratio_is_null_when_unopposed_has_no_gain():
    assert robustness_ratio(0.0, 0.2) is None
    assert robustness_ratio(-0.1, 0.2) is None
    assert robustness_ratio(0.5, 0.25) == 0.5
